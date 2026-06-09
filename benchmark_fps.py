import argparse
import time

import torch

from train import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    model = Generator(args.latent_dim).to(device)
    saved = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(saved.get("generator", saved))
    model.eval()

    noise = torch.randn(
        args.batch_size,
        args.latent_dim,
        1,
        1,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda",
            ):
                model(noise)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(args.frames):
            noise.normal_()
            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda",
            ):
                model(noise)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    count = args.frames * args.batch_size
    fps = count / elapsed
    latency_ms = elapsed * 1000.0 / count

    print()
    print(f"Generated: {count} images")
    print(f"Elapsed: {elapsed:.3f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Latency: {latency_ms:.3f} ms/image")


if __name__ == "__main__":
    main()
