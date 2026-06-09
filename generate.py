import argparse

import torch
from torchvision.utils import save_image

from train import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="generated_trees.png")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(args.latent_dim).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("generator", checkpoint)
    model.load_state_dict(state)
    model.eval()

    noise = torch.randn(args.count, args.latent_dim, 1, 1, device=device)
    with torch.no_grad():
        images = model(noise)

    save_image(
        images,
        args.out,
        nrow=max(1, int(args.count**0.5)),
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
