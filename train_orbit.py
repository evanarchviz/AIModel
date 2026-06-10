import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


ELEVATION_FOLDERS = {
    15.0: "15",
    30.0: "30",
    45.0: "45",
}


class OrbitDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.samples = []

        self._add_ring(
            folder=self.root,
            elevation=0.0,
            count=360,
            azimuth_step=1.0,
        )

        for elevation, folder_name in ELEVATION_FOLDERS.items():
            self._add_ring(
                folder=self.root / folder_name,
                elevation=elevation,
                count=72,
                azimuth_step=5.0,
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _add_ring(
        self,
        folder: Path,
        elevation: float,
        count: int,
        azimuth_step: float,
    ):
        if not folder.exists():
            raise FileNotFoundError(f"Missing orbit folder: {folder}")

        expected = [folder / f"{index:04d}.png" for index in range(1, count + 1)]
        missing = [path.name for path in expected if not path.exists()]
        if missing:
            raise ValueError(
                f"{folder} is missing {len(missing)} expected files. "
                f"First missing file: {missing[0]}"
            )

        extras = sorted(folder.glob("*.png"))
        if len(extras) != count:
            raise ValueError(
                f"Expected exactly {count} PNG files in {folder}, found {len(extras)}"
            )

        for index, path in enumerate(expected):
            self.samples.append(
                {
                    "path": path,
                    "azimuth": index * azimuth_step,
                    "elevation": elevation,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.transform(Image.open(sample["path"]).convert("RGB"))
        code = make_camera_code(
            sample["azimuth"],
            sample["elevation"],
        )
        return code, image


class OrbitGenerator(nn.Module):
    def __init__(self, hidden=256, base=64):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * 4 * base * 16),
            nn.SiLU(),
        )

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            )

        self.base = base
        self.decoder = nn.Sequential(
            block(base * 16, base * 8),
            block(base * 8, base * 4),
            block(base * 4, base * 2),
            block(base * 2, base),
            block(base, base // 2),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(base // 2, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, camera_code):
        features = self.mapping(camera_code)
        features = features.view(camera_code.shape[0], self.base * 16, 4, 4)
        return self.to_rgb(self.decoder(features))


def make_camera_code(azimuth_degrees, elevation_degrees):
    azimuth = math.radians(float(azimuth_degrees))
    elevation = math.radians(float(elevation_degrees))
    return torch.tensor(
        [
            math.sin(azimuth),
            math.cos(azimuth),
            math.sin(elevation),
            math.cos(elevation),
        ],
        dtype=torch.float32,
    )


def make_camera_codes(azimuths, elevations, device):
    codes = [
        make_camera_code(azimuth, elevation)
        for elevation in elevations
        for azimuth in azimuths
    ]
    return torch.stack(codes).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="runs/orbit_tree_elevation")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    dataset = OrbitDataset(args.data)
    print(
        f"Loaded {len(dataset)} images: "
        "360 at 0 degrees and 72 each at +15, +30, +45 degrees"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    out = Path(args.out)
    samples_dir = out / "samples"
    checkpoints_dir = out / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model = OrbitGenerator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    sample_azimuths = list(range(0, 360, 45))
    sample_elevations = [0.0, 15.0, 30.0, 45.0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for codes, target in loader:
            codes = codes.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(codes)
                l1 = F.l1_loss(prediction, target)
                mse = F.mse_loss(prediction, target)
                loss = l1 + 0.25 * mse

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch:04d}/{args.epochs} loss={total_loss / len(loader):.6f}")

        if epoch == 1 or epoch % args.sample_every == 0:
            model.eval()
            with torch.inference_mode():
                codes = make_camera_codes(
                    sample_azimuths,
                    sample_elevations,
                    device,
                )
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    samples = model(codes).float()
                save_image(
                    samples,
                    samples_dir / f"epoch_{epoch:04d}.png",
                    nrow=len(sample_azimuths),
                    normalize=True,
                    value_range=(-1, 1),
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "camera_code_dimensions": 4,
                    "elevations": [0.0, 15.0, 30.0, 45.0],
                },
                checkpoints_dir / "latest.pt",
            )

    torch.save(model.state_dict(), checkpoints_dir / "orbit_generator_final.pt")


if __name__ == "__main__":
    main()
