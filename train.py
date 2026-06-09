import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


class TreeDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(
            path
            for path in Path(root).glob("tree_*.png")
            if not path.stem.endswith("_mask")
        )
        if not self.files:
            raise FileNotFoundError(f"No RGB tree images found in {root}")

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.025, 0.015),
                    scale=(0.94, 1.06),
                    fill=0,
                ),
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                    saturation=0.08,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.transform(Image.open(self.files[index]).convert("RGB"))


def weights_init(module):
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim=128, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 16, base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.ConvTranspose2d(base, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, base * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 16, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset train folder")
    parser.add_argument("--out", default="runs/tree_gan")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=25)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    out = Path(args.out)
    samples_dir = out / "samples"
    checkpoints_dir = out / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dataset = TreeDataset(args.data)
    print(f"Loaded {len(dataset)} RGB training images (mask files excluded)")
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)
    scaler_g = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        generator_loss_total = 0.0
        discriminator_loss_total = 0.0

        for real in loader:
            real = real.to(device, non_blocking=True)
            batch = real.size(0)

            optimizer_d.zero_grad(set_to_none=True)
            noise = torch.randn(batch, args.latent_dim, 1, 1, device=device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                fake = generator(noise)
                real_logits = discriminator(real)
                fake_logits = discriminator(fake.detach())
                real_targets = torch.full_like(real_logits, 0.9)
                fake_targets = torch.zeros_like(fake_logits)
                loss_d = criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            optimizer_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                fake_logits = discriminator(fake)
                loss_g = criterion(fake_logits, torch.ones_like(fake_logits))

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            generator_loss_total += loss_g.item()
            discriminator_loss_total += loss_d.item()

        print(
            f"Epoch {epoch:04d}/{args.epochs} "
            f"G={generator_loss_total / len(loader):.4f} "
            f"D={discriminator_loss_total / len(loader):.4f}"
        )

        if epoch == 1 or epoch % args.sample_every == 0:
            generator.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    samples = generator(fixed_noise).float()
                save_image(
                    samples,
                    samples_dir / f"epoch_{epoch:04d}.png",
                    nrow=8,
                    normalize=True,
                    value_range=(-1, 1),
                )
            generator.train()

            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "latent_dim": args.latent_dim,
                },
                checkpoints_dir / "latest.pt",
            )

    torch.save(generator.state_dict(), checkpoints_dir / "generator_final.pt")


if __name__ == "__main__":
    main()
