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


class OrbitDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(Path(root).glob("*.png"))
        if len(self.files) != 360:
            raise ValueError(f"Expected exactly 360 PNG files, found {len(self.files)}")

        expected = [f"{index:04d}.png" for index in range(1, 361)]
        actual = [path.name for path in self.files]
        if actual != expected:
            raise ValueError("Files must be named 0001.png through 0360.png with no gaps")

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.files[index]).convert("RGB"))
        angle = math.radians(index)
        angle_code = torch.tensor(
            [math.sin(angle), math.cos(angle)],
            dtype=torch.float32,
        )
        return angle_code, image


class OrbitGenerator(nn.Module):
    def __init__(self, hidden=256, base=64):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(2, hidden),
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

    def forward(self, angle_code):
        features = self.mapping(angle_code)
        features = features.view(angle_code.shape[0], self.base * 16, 4, 4)
        return self.to_rgb(self.decoder(features))


def angle_code(degrees, device):
    radians = torch.deg2rad(degrees)
    return torch.stack((torch.sin(radians), torch.cos(radians)), dim=1).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="runs/orbit_tree")
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
    print("Loaded 360 orbit images: 0001.png -> 0 degrees, 0360.png -> 359 degrees")

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

    sample_degrees = torch.arange(0, 360, 15, dtype=torch.float32, device=device)

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
                codes = angle_code(sample_degrees, device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    samples = model(codes).float()
                save_image(
                    samples,
                    samples_dir / f"epoch_{epoch:04d}.png",
                    nrow=6,
                    normalize=True,
                    value_range=(-1, 1),
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                },
                checkpoints_dir / "latest.pt",
            )

    torch.save(model.state_dict(), checkpoints_dir / "orbit_generator_final.pt")


if __name__ == "__main__":
    main()
