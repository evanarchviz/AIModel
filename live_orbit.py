import argparse
import math
import time
import tkinter as tk

import torch
from PIL import Image, ImageTk

from train_orbit import OrbitGenerator


def make_angle_code(angle_degrees: float, device: torch.device) -> torch.Tensor:
    radians = math.radians(angle_degrees % 360.0)
    return torch.tensor(
        [[math.sin(radians), math.cos(radians)]],
        dtype=torch.float32,
        device=device,
    )


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(-1, 1)
    image = ((image + 1.0) * 127.5).to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image, mode="RGB")


class OrbitPreview:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(torch.cuda.get_device_name(0))

        self.model = OrbitGenerator().to(self.device)
        saved = torch.load(args.checkpoint, map_location=self.device)
        self.model.load_state_dict(saved.get("model", saved))
        self.model.eval()

        self.angle = 0.0
        self.speed = args.speed
        self.playing = True
        self.running = True
        self.last_time = time.perf_counter()
        self.frame_count = 0
        self.fps_window_start = self.last_time
        self.fps_window_frames = 0
        self.live_fps = 0.0

        self.root = tk.Tk()
        self.root.title("AIModel Tree Orbit")
        self.root.configure(bg="black")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack()

        self.info_label = tk.Label(
            self.root,
            text="Starting...",
            fg="white",
            bg="black",
            font=("Consolas", 12),
        )
        self.info_label.pack(fill="x")

        self.root.bind("<Left>", lambda event: self.step_angle(-1.0))
        self.root.bind("<Right>", lambda event: self.step_angle(1.0))
        self.root.bind("<space>", lambda event: self.toggle_play())
        self.root.bind("<Button-1>", self.start_drag)
        self.root.bind("<B1-Motion>", self.drag)

        self.drag_x = None
        self.photo = None
        self.root.after(0, self.update)

    def close(self):
        self.running = False
        self.root.destroy()

    def toggle_play(self):
        self.playing = not self.playing

    def step_angle(self, amount: float):
        self.playing = False
        self.angle = (self.angle + amount) % 360.0

    def start_drag(self, event):
        self.playing = False
        self.drag_x = event.x

    def drag(self, event):
        if self.drag_x is None:
            self.drag_x = event.x
            return
        delta = event.x - self.drag_x
        self.drag_x = event.x
        self.angle = (self.angle + delta * self.args.drag_sensitivity) % 360.0

    def update(self):
        if not self.running:
            return

        now = time.perf_counter()
        delta_time = now - self.last_time
        self.last_time = now

        if self.playing:
            self.angle = (self.angle + self.speed * delta_time) % 360.0

        code = make_angle_code(self.angle, self.device)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                output = self.model(code)[0]

        image = tensor_to_image(output)
        image = image.resize(
            (self.args.window_size, self.args.window_size),
            Image.Resampling.NEAREST,
        )
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

        self.frame_count += 1
        self.fps_window_frames += 1
        fps_elapsed = now - self.fps_window_start
        if fps_elapsed >= 0.5:
            self.live_fps = self.fps_window_frames / fps_elapsed
            self.fps_window_start = now
            self.fps_window_frames = 0

        state = "PLAY" if self.playing else "PAUSE"
        self.info_label.configure(
            text=(
                f"Angle: {self.angle:7.2f} deg   "
                f"FPS: {self.live_fps:7.2f}   "
                f"Speed: {self.speed:6.1f} deg/s   "
                f"{state}"
            )
        )
        self.root.title(f"AIModel Tree Orbit - {self.angle:.1f} deg")
        self.root.after(1, self.update)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--speed", type=float, default=45.0)
    parser.add_argument("--drag-sensitivity", type=float, default=0.5)
    args = parser.parse_args()

    app = OrbitPreview(args)
    app.run()


if __name__ == "__main__":
    main()
