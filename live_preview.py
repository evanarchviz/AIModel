import argparse
import time
import tkinter as tk

import torch
from PIL import Image, ImageTk

from train import Generator


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(-1, 1)
    image = ((image + 1.0) * 127.5).to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image, mode="RGB")


class LivePreview:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(torch.cuda.get_device_name(0))

        self.model = Generator(args.latent_dim).to(self.device)
        saved = torch.load(args.checkpoint, map_location=self.device)
        self.model.load_state_dict(saved.get("generator", saved))
        self.model.eval()

        self.noise = torch.randn(
            1,
            args.latent_dim,
            1,
            1,
            device=self.device,
        )

        self.root = tk.Tk()
        self.root.title("AIModel Live Tree Preview")
        self.root.configure(bg="black")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.label = tk.Label(self.root, bg="black")
        self.label.pack()

        self.info = tk.Label(
            self.root,
            text="Starting...",
            fg="white",
            bg="black",
            font=("Consolas", 12),
        )
        self.info.pack(fill="x")

        self.running = True
        self.frame_count = 0
        self.total_start = time.perf_counter()
        self.window_start = self.total_start
        self.window_frames = 0
        self.current_photo = None

        with torch.inference_mode():
            for _ in range(args.warmup):
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.device.type == "cuda",
                ):
                    self.model(self.noise)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self.root.after(0, self.update_frame)

    def close(self):
        self.running = False
        self.root.destroy()

    def update_frame(self):
        if not self.running:
            return

        frame_start = time.perf_counter()

        self.noise.normal_()
        with torch.inference_mode():
            with torch.amp.autocast(
                "cuda",
                enabled=self.device.type == "cuda",
            ):
                output = self.model(self.noise)[0]

        image = tensor_to_image(output)
        image = image.resize(
            (self.args.window_size, self.args.window_size),
            Image.Resampling.NEAREST,
        )

        self.current_photo = ImageTk.PhotoImage(image)
        self.label.configure(image=self.current_photo)

        self.frame_count += 1
        self.window_frames += 1

        now = time.perf_counter()
        window_elapsed = now - self.window_start
        total_elapsed = now - self.total_start

        if window_elapsed >= 0.5:
            live_fps = self.window_frames / window_elapsed
            average_fps = self.frame_count / total_elapsed
            frame_ms = (now - frame_start) * 1000.0

            self.info.configure(
                text=(
                    f"Live FPS: {live_fps:7.2f}   "
                    f"Average: {average_fps:7.2f}   "
                    f"Frame: {frame_ms:6.2f} ms   "
                    f"Generated: {self.frame_count}"
                )
            )
            self.root.title(f"AIModel Live Tree Preview - {live_fps:.1f} FPS")
            self.window_start = now
            self.window_frames = 0

        if self.args.max_fps > 0:
            target_seconds = 1.0 / self.args.max_fps
            elapsed = time.perf_counter() - frame_start
            delay_ms = max(1, int((target_seconds - elapsed) * 1000.0))
        else:
            delay_ms = 1

        self.root.after(delay_ms, self.update_frame)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--max-fps",
        type=float,
        default=0,
        help="0 means generate as fast as the window can display",
    )
    args = parser.parse_args()

    app = LivePreview(args)
    app.run()


if __name__ == "__main__":
    main()
