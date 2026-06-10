import argparse
import random
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from multi_tree_model import MultiTreeConfig, MultiTreeModel, make_rays


ELEVATION_FOLDERS = {
    0.0: None,
    15.0: "15",
    30.0: "30",
    45.0: "45",
}


class TreeSequence:
    def __init__(self, root: Path, resolution: int):
        self.root = root
        self.resolution = resolution
        self.views = []
        self._scan()

    def _scan_ring(self, folder: Path, elevation: float, expected_count=None):
        files = sorted(folder.glob("*.png"))
        if not files:
            return

        names = [path.name for path in files]
        expected_names = [f"{index:04d}.png" for index in range(1, len(files) + 1)]
        if names != expected_names:
            raise ValueError(f"Non-contiguous filenames in {folder}")

        if expected_count is not None and len(files) != expected_count:
            raise ValueError(
                f"Expected {expected_count} PNG files in {folder}, found {len(files)}"
            )

        step = 360.0 / len(files)
        for index, path in enumerate(files):
            self.views.append(
                {
                    "path": path,
                    "azimuth": index * step,
                    "elevation": elevation,
                }
            )

    def _scan(self):
        root_files = sorted(self.root.glob("*.png"))
        if len(root_files) not in (359, 360):
            raise ValueError(
                f"{self.root} must contain 359 or 360 root PNG files, "
                f"found {len(root_files)}"
            )
        self._scan_ring(self.root, elevation=0.0)

        for elevation, folder_name in ELEVATION_FOLDERS.items():
            if elevation == 0.0:
                continue
            folder = self.root / folder_name
            if folder.exists():
                self._scan_ring(folder, elevation=elevation, expected_count=72)

        if len(self.views) < 16:
            raise ValueError(f"Not enough views in {self.root}")

    def __len__(self):
        return len(self.views)


class MultiTreeDataset:
    def __init__(self, roots, resolution):
        self.resolution = resolution
        self.trees = [TreeSequence(Path(root), resolution) for root in roots]
        print(f"Loaded {len(self.trees)} trees")
        for tree in self.trees:
            print(f"  {tree.root.name}: {len(tree)} views")

    def __len__(self):
        return len(self.trees)


@lru_cache(maxsize=4096)
def load_image_cached(path_string: str, resolution: int):
    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.ToTensor(),
        ]
    )
    with Image.open(path_string) as image:
        return transform(image.convert("RGB"))


def load_view(view, resolution, device):
    image = load_image_cached(str(view["path"]), resolution).clone().to(device)
    return image


def sample_episode(dataset, batch_size, min_context, max_context):
    episodes = []
    for _ in range(batch_size):
        tree_index = random.randrange(len(dataset.trees))
        tree = dataset.trees[tree_index]
        context_count = random.randint(min_context, max_context)
        selected = random.sample(range(len(tree.views)), context_count + 1)
        context_indices = selected[:-1]
        target_index = selected[-1]
        episodes.append((tree, context_indices, target_index))
    return episodes


def build_memory(model, episodes, device):
    batch_size = len(episodes)
    memory = model.initial_memory(batch_size, device)
    max_context = max(len(context_indices) for _, context_indices, _ in episodes)

    for context_step in range(max_context):
        images = []
        azimuths = []
        elevations = []
        active = []

        for batch_index, (tree, context_indices, _) in enumerate(episodes):
            if context_step < len(context_indices):
                view = tree.views[context_indices[context_step]]
                images.append(load_view(view, tree.resolution, device))
                azimuths.append(view["azimuth"])
                elevations.append(view["elevation"])
                active.append(batch_index)

        if not active:
            continue

        active_tensor = torch.tensor(active, dtype=torch.long, device=device)
        image_batch = torch.stack(images)
        azimuth_batch = torch.tensor(azimuths, dtype=torch.float32, device=device)
        elevation_batch = torch.tensor(elevations, dtype=torch.float32, device=device)

        updated = model.observe(
            image_batch,
            azimuth_batch,
            elevation_batch,
            memory.index_select(0, active_tensor),
        )
        memory = memory.index_copy(0, active_tensor, updated.to(memory.dtype))

    return memory


def sample_target_rays(episodes, resolution, rays_per_view, device):
    pixel_indices = torch.randint(
        0,
        resolution * resolution,
        (rays_per_view,),
        device=device,
    )

    targets = []
    masks = []
    azimuths = []
    elevations = []

    ys = torch.div(pixel_indices, resolution, rounding_mode="floor")
    xs = pixel_indices % resolution

    for tree, _, target_index in episodes:
        view = tree.views[target_index]
        image = load_view(view, resolution, device)
        target = image[:, ys, xs].permute(1, 0).contiguous()
        targets.append(target)
        masks.append((target.amax(dim=-1) > 0.02).float())
        azimuths.append(view["azimuth"])
        elevations.append(view["elevation"])

    return (
        torch.tensor(azimuths, dtype=torch.float32, device=device),
        torch.tensor(elevations, dtype=torch.float32, device=device),
        pixel_indices,
        torch.stack(targets),
        torch.stack(masks),
    )


def render_full(model, memory, azimuth, elevation, config, device, ray_chunk=4096):
    pixel_count = config.image_resolution * config.image_resolution
    outputs = []
    uncertainties = []

    for start in range(0, pixel_count, ray_chunk):
        end = min(start + ray_chunk, pixel_count)
        pixel_indices = torch.arange(start, end, device=device)
        origins, directions = make_rays(
            torch.tensor([azimuth], device=device),
            torch.tensor([elevation], device=device),
            config,
            device,
            pixel_indices,
        )
        rgb, _, uncertainty = model.render_rays(memory, origins, directions)
        outputs.append(rgb[0])
        uncertainties.append(uncertainty[0])

    rgb = torch.cat(outputs).reshape(
        config.image_resolution,
        config.image_resolution,
        3,
    )
    uncertainty = torch.cat(uncertainties).reshape(
        config.image_resolution,
        config.image_resolution,
    )
    return rgb, uncertainty


def save_validation(model, dataset, config, device, output_path):
    tree = dataset.trees[0]
    context_indices = [
        0,
        len(tree.views) // 4,
        len(tree.views) // 2,
        (3 * len(tree.views)) // 4,
    ]
    target_index = min(len(tree.views) - 1, len(tree.views) // 3)
    episode = [(tree, context_indices, target_index)]

    model.eval()
    with torch.inference_mode():
        memory = build_memory(model, episode, device)
        target_view = tree.views[target_index]
        prediction, uncertainty = render_full(
            model,
            memory,
            target_view["azimuth"],
            target_view["elevation"],
            config,
            device,
        )
        target = load_view(target_view, config.image_resolution, device)

        uncertainty = uncertainty / uncertainty.max().clamp_min(1e-6)
        uncertainty_rgb = uncertainty[..., None].expand(-1, -1, 3)
        panel = torch.stack(
            (
                target,
                prediction.permute(2, 0, 1),
                uncertainty_rgb.permute(2, 0, 1),
            )
        )
        save_image(panel.cpu(), output_path, nrow=3, value_range=(0.0, 1.0))
    model.train()


def discover_tree_roots(data_root, explicit_trees):
    if explicit_trees:
        return [Path(path) for path in explicit_trees]

    root = Path(data_root)
    trees = sorted(path for path in root.glob("orbit_tree_*") if path.is_dir())
    if not trees:
        raise FileNotFoundError(
            f"No orbit_tree_* directories found under {root}. "
            "Use --tree to provide explicit paths."
        )
    return trees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--tree", action="append", default=[])
    parser.add_argument("--out", default="runs/multi_tree")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-context", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=8)
    parser.add_argument("--rays-per-view", type=int, default=4096)
    parser.add_argument("--memory-dim", type=int, default=256)
    parser.add_argument("--plane-channels", type=int, default=16)
    parser.add_argument("--plane-resolution", type=int, default=32)
    parser.add_argument("--samples-per-ray", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.min_context < 1 or args.max_context < args.min_context:
        raise ValueError("Invalid context-view range")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    roots = discover_tree_roots(args.data_root, args.tree)
    dataset = MultiTreeDataset(roots, args.resolution)

    config = MultiTreeConfig(
        image_resolution=args.resolution,
        memory_dim=args.memory_dim,
        plane_channels=args.plane_channels,
        plane_resolution=args.plane_resolution,
        samples_per_ray=args.samples_per_ray,
    )
    model = MultiTreeModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out = Path(args.out)
    samples_dir = out / "samples"
    checkpoints_dir = out / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        episodes = sample_episode(
            dataset,
            args.batch_size,
            args.min_context,
            args.max_context,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            memory = build_memory(model, episodes, device)
            azimuths, elevations, pixel_indices, targets, masks = sample_target_rays(
                episodes,
                args.resolution,
                args.rays_per_view,
                device,
            )
            origins, directions = make_rays(
                azimuths,
                elevations,
                config,
                device,
                pixel_indices,
            )
            predicted_rgb, predicted_opacity, predicted_uncertainty = model.render_rays(
                memory,
                origins,
                directions,
            )
            rgb_loss = F.smooth_l1_loss(predicted_rgb, targets)

        opacity = predicted_opacity.float().clamp(1e-5, 1.0 - 1e-5)
        opacity_loss = F.binary_cross_entropy(opacity, masks.float())
        per_ray_error = (predicted_rgb.float() - targets.float()).abs().mean(dim=-1).detach()
        uncertainty_loss = F.smooth_l1_loss(
            predicted_uncertainty.float(),
            per_ray_error,
        )
        memory_regularization = memory.float().pow(2).mean()

        loss = (
            rgb_loss.float()
            + 0.25 * opacity_loss
            + 0.1 * uncertainty_loss
            + 1e-5 * memory_regularization
        )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 25 == 0:
            context_average = sum(len(item[1]) for item in episodes) / len(episodes)
            print(
                f"Step {step:06d}/{args.steps} "
                f"loss={loss.item():.6f} "
                f"rgb={rgb_loss.item():.6f} "
                f"opacity={opacity_loss.item():.6f} "
                f"uncertainty={uncertainty_loss.item():.6f} "
                f"context={context_average:.1f}"
            )

        if step == 1 or step % args.checkpoint_every == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "config": vars(config),
                    "tree_roots": [str(root) for root in roots],
                },
                checkpoints_dir / "latest.pt",
            )
            save_validation(
                model,
                dataset,
                config,
                device,
                samples_dir / f"step_{step:06d}.png",
            )

    torch.save(model.state_dict(), checkpoints_dir / "multi_tree_final.pt")


if __name__ == "__main__":
    main()
