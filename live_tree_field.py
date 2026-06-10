import argparse
import time
import tkinter as tk

import torch
from PIL import Image, ImageTk

from tree_field import CameraConfig, DenseTreeField


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(image, mode="RGB")


class TreeFieldPreview:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(torch.cuda.get_device_name(0))
            torch.backends.cudnn.benchmark = True

        saved = torch.load(args.checkpoint, map_location=self.device)
        field_resolution = saved.get("field_resolution", 64)
        config_values = saved.get("camera_config", {})

        checkpoint_config = CameraConfig(**config_values)
        self.training_resolution = checkpoint_config.width

        preview_samples = args.samples_per_ray
        if preview_samples is None:
            preview_samples = min(checkpoint_config.samples_per_ray, 24)

        self.config = CameraConfig(
            width=args.render_resolution,
            height=args.render_resolution,
            fov_degrees=checkpoint_config.fov_degrees,
            radius=checkpoint_config.radius,
            near=checkpoint_config.near,
            far=checkpoint_config.far,
            samples_per_ray=preview_samples,
        )

        self.model = DenseTreeField(field_resolution).to(self.device)
        self.model.load_state_dict(saved.get("model", saved))
        self.model.eval()

        self.azimuth = 0.0
        self.elevation = 0.0
        self.playing = True
        self.running = True
        self.last_time = time.perf_counter()
        self.fps_start = self.last_time
        self.fps_frames = 0
        self.live_fps = 0.0

        self.root = tk.Tk()
        self.root.title("AIModel Shared 3D Tree Field")
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

        self.help_label = tk.Label(
            self.root,
            text="1: fast  2: balanced  3: quality  |  arrows/mouse: orbit  |  space: pause",
            fg="#bbbbbb",
            bg="black",
            font=("Consolas", 10),
        )
        self.help_label.pack(fill="x")

        self.root.bind("<Left>", lambda event: self.step_azimuth(-1.0))
        self.root.bind("<Right>", lambda event: self.step_azimuth(1.0))
        self.root.bind("<Up>", lambda event: self.step_elevation(1.0))
        self.root.bind("<Down>", lambda event: self.step_elevation(-1.0))
        self.root.bind("<Home>", lambda event: self.reset_camera())
        self.root.bind("<space>", lambda event: self.toggle_play())
        self.root.bind("<Key-1>", lambda event: self.set_quality(96, 12))
        self.root.bind("<Key-2>", lambda event: self.set_quality(128, 24))
        self.root.bind("<Key-3>", lambda event: self.set_quality(256, 32))
        self.root.bind("<Button-1>", self.start_drag)
        self.root.bind("<B1-Motion>", self.drag)

        self.drag_x = None
        self.drag_y = None
        self.photo = None
        self.root.after(0, self.update)

    def close(self):
        self.running = False
        self.root.destroy()

    def toggle_play(self):
        self.playing = not self.playing

    def set_quality(self, resolution, samples):
        self.config.width = resolution
        self.config.height = resolution
        self.config.samples_per_ray = samples
        print(f"Preview quality: {resolution}x{resolution}, {samples} samples/ray")

    def step_azimuth(self, amount):
        self.playing = False
        self.azimuth = (self.azimuth + amount) % 360.0

    def step_elevation(self, amount):
        self.playing = False
        self.elevation = max(0.0, min(45.0, self.elevation + amount))

    def reset_camera(self):
        self.azimuth = 0.0
        self.elevation = 0.0

    def start_drag(self, event):
        self.playing = False
        self.drag_x = event.x
        self.drag_y = event.y

    def drag(self, event):
        if self.drag_x is None or self.drag_y is None:
            self.drag_x = event.x
            self.drag_y = event.y
            return

        delta_x = event.x - self.drag_x
        delta_y = event.y - self.drag_y
        self.drag_x = event.x
        self.drag_y = event.y

        self.azimuth = (
            self.azimuth + delta_x * self.args.drag_sensitivity
        ) % 360.0
        self.elevation = max(
            0.0,
            min(
                45.0,
                self.elevation - delta_y * self.args.drag_sensitivity,
            ),
        )

    def update(self):
        if not self.running:
            return

        frame_start = time.perf_counter()
        delta_time = frame_start - self.last_time
        self.last_time = frame_start

        if self.playing:
            self.azimuth = (
                self.azimuth + self.args.speed * delta_time
            ) % 360.0

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                rgb, _, _ = self.model.render_image(
                    self.azimuth,
                    self.elevation,
                    self.config,
                    self.device,
                    ray_chunk=self.args.ray_chunk,
                )

        image = tensor_to_image(rgb)
        image = image.resize(
            (self.args.window_size, self.args.window_size),
            Image.Resampling.BILINEAR,
        )
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

        self.fps_frames += 1
        now = time.perf_counter()
        fps_elapsed = now - self.fps_start
        if fps_elapsed >= 0.5:
            self.live_fps = self.fps_frames / fps_elapsed
            self.fps_frames = 0
            self.fps_start = now

        state = "PLAY" if self.playing else "PAUSE"
        self.info_label.configure(
            text=(
                f"Azimuth: {self.azimuth:7.2f} deg   "
                f"Elevation: {self.elevation:6.2f} deg   "
                f"FPS: {self.live_fps:6.2f}   "
                f"Render: {self.config.width}²   "
                f"Samples: {self.config.samples_per_ray}   "
                f"{state}"
            )
        )
        self.root.title(
            f"AIModel 3D Tree Field - az {self.azimuth:.1f}, el {self.elevation:.1f}"
        )
        self.root.after(1, self.update)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument(
        "--render-resolution",
        type=int,
        default=128,
        help="Internal ray-marched resolution; output is upscaled to window size",
    )
    parser.add_argument("--speed", type=float, default=30.0)
    parser.add_argument("--drag-sensitivity", type=float, default=0.5)
    parser.add_argument("--ray-chunk", type=int, default=16384)
    parser.add_argument(
        "--samples-per-ray",
        type=int,
        default=None,
        help="Defaults to at most 24 for interactive preview",
    )
    args = parser.parse_args()

    app = TreeFieldPreview(args)
    app.run()


if __name__ == "__main__":
    main()
