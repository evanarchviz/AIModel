import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from tree_field import CameraConfig, DenseTreeField, make_rays


ELEVATION_FOLDERS = {
    0.0: None,
    15.0: "15",
    30.0: "30",
    45.0: "45",
}


class TreeFieldDataset:
    def __init__(self, root: str, resolution: int):
        self.root = Path(root)
        self.resolution = resolution
        self.samples = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), antialias=True),
                transforms.ToTensor(),
            ]
        )

        self._add_ring(self.root, elevation=0.0, count=360, step=1.0)
        for elevation, folder_name in ELEVATION_FOLDERS.items():
            if elevation == 0.0:
                continue
            self._add_ring(
                self.root / folder_name,
                elevation=elevation,
                count=72,
                step=5.0,
            )

        self.images = [
            self.transform(Image.open(sample["path"]).convert("RGB"))
            for sample in self.samples
        ]

    def _add_ring(self, folder: Path, elevation: float, count: int, step: float):
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        expected = [folder / f"{index:04d}.png" for index in range(1, count + 1)]
        missing = [path.name for path in expected if not path.exists()]
        if missing:
            raise ValueError(
                f"{folder} is missing {len(missing)} expected files. "
                f"First missing file: {missing[0]}"
            )

        png_files = sorted(folder.glob("*.png"))
        if len(png_files) != count:
            raise ValueError(
                f"Expected exactly {count} PNG files in {folder}, found {len(png_files)}"
            )

        for index, path in enumerate(expected):
            self.samples.append(
                {
                    "path": path,
                    "azimuth": index * step,
                    "elevation": elevation,
                }
            )

    def __len__(self):
        return len(self.samples)


def total_variation(grid: torch.Tensor) -> torch.Tensor:
    tv_x = (grid[..., 1:, :, :] - grid[..., :-1, :, :]).abs().mean()
    tv_y = (grid[..., :, 1:, :] - grid[..., :, :-1, :]).abs().mean()
    tv_z = (grid[..., :, :, 1:] - grid[..., :, :, :-1]).abs().mean()
    return tv_x + tv_y + tv_z


def sample_training_batch(dataset, batch_size, rays_per_view, device):
    view_indices = torch.randint(0, len(dataset), (batch_size,))
    pixel_count = dataset.resolution * dataset.resolution
    pixel_indices = torch.randint(0, pixel_count, (rays_per_view,))

    images = torch.stack(
        [dataset.images[index] for index in view_indices.tolist()]
    ).to(device)
    azimuths = torch.tensor(
        [dataset.samples[index]["azimuth"] for index in view_indices.tolist()],
        dtype=torch.float32,
        device=device,
    )
    elevations = torch.tensor(
        [dataset.samples[index]["elevation"] for index in view_indices.tolist()],
        dtype=torch.float32,
        device=device,
    )

    ys = torch.div(pixel_indices, dataset.resolution, rounding_mode="floor")
    xs = pixel_indices % dataset.resolution
    targets = images[:, :, ys, xs].permute(0, 2, 1).contiguous()
    masks = (targets.amax(dim=-1) > 0.02).float()

    return azimuths, elevations, pixel_indices, targets, masks


def render_validation_grid(model, config, device, output_path, ray_chunk):
    azimuths = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    elevations = [0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0]
    images = []

    model.eval()
    with torch.inference_mode():
        for elevation in elevations:
            for azimuth in azimuths:
                rgb, _, _ = model.render_image(
                    azimuth,
                    elevation,
                    config,
                    device,
                    ray_chunk=ray_chunk,
                )
                images.append(rgb.permute(2, 0, 1).cpu())

    save_image(
        torch.stack(images),
        output_path,
        nrow=len(azimuths),
        value_range=(0.0, 1.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="runs/tree_field")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rays-per-view", type=int, default=4096)
    parser.add_argument("--samples-per-ray", type=int, default=48)
    parser.add_argument("--field-resolution", type=int, default=64)
    parser.add_argument("--validation-ray-chunk", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--near", type=float, default=1.5)
    parser.add_argument("--far", type=float, default=4.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.field_resolution <= 0:
        raise ValueError("--field-resolution must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    dataset = TreeFieldDataset(args.data, args.resolution)
    print(
        f"Loaded {len(dataset)} calibrated views at "
        f"{args.resolution}x{args.resolution}"
    )
    print(
        f"Field resolution: {args.field_resolution}^3, "
        f"rays/view: {args.rays_per_view}, "
        f"samples/ray: {args.samples_per_ray}"
    )

    config = CameraConfig(
        width=args.resolution,
        height=args.resolution,
        fov_degrees=args.fov,
        radius=args.radius,
        near=args.near,
        far=args.far,
        samples_per_ray=args.samples_per_ray,
    )

    out = Path(args.out)
    samples_dir = out / "samples"
    checkpoints_dir = out / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model = DenseTreeField(args.field_resolution).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    model.train()
    for step in range(1, args.steps + 1):
        azimuths, elevations, pixel_indices, targets, masks = sample_training_batch(
            dataset,
            args.batch_size,
            args.rays_per_view,
            device,
        )

        origins, directions = make_rays(
            azimuths,
            elevations,
            config,
            device,
            pixel_indices=pixel_indices,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            predicted_rgb, predicted_opacity, _ = model.render_rays(
                origins,
                directions,
                config,
                randomized=True,
            )
            rgb_loss = F.smooth_l1_loss(predicted_rgb, targets)
            tv_loss = total_variation(model.density_grid) + 0.25 * total_variation(
                model.color_grid
            )

        predicted_opacity_f32 = predicted_opacity.float().clamp(1e-5, 1.0 - 1e-5)
        masks_f32 = masks.float()
        opacity_loss = F.binary_cross_entropy(predicted_opacity_f32, masks_f32)
        opacity_binary_loss = (
            predicted_opacity_f32 * (1.0 - predicted_opacity_f32)
        ).mean()

        loss = (
            rgb_loss.float()
            + 0.5 * opacity_loss
            + 0.05 * opacity_binary_loss
            + 1e-5 * tv_loss.float()
        )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 25 == 0:
            print(
                f"Step {step:05d}/{args.steps} "
                f"loss={loss.item():.6f} "
                f"rgb={rgb_loss.item():.6f} "
                f"opacity={opacity_loss.item():.6f} "
                f"binary={opacity_binary_loss.item():.6f}"
            )

        if step == 1 or step % args.checkpoint_every == 0:
            checkpoint_path = checkpoints_dir / "latest.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "field_resolution": args.field_resolution,
                    "camera_config": vars(config),
                    "training_resolution": args.resolution,
                },
                checkpoint_path,
            )

            render_validation_grid(
                model,
                config,
                device,
                samples_dir / f"step_{step:05d}.png",
                ray_chunk=args.validation_ray_chunk,
            )
            model.train()

    torch.save(model.state_dict(), checkpoints_dir / "tree_field_final.pt")


if __name__ == "__main__":
    main()
