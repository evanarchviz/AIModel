import argparse
import math
import time
import tkinter as tk
from pathlib import Path

import torch
from PIL import Image, ImageTk
from torchvision import transforms

from multi_tree_model import MultiTreeConfig, MultiTreeModel, make_rays


ELEVATION_FOLDERS = {
    0.0: None,
    15.0: "15",
    30.0: "30",
    45.0: "45",
}


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(image, mode="RGB")


def scan_views(tree_root: Path):
    views = []

    def add_ring(folder: Path, elevation: float):
        files = sorted(folder.glob("*.png"))
        if not files:
            return
        step = 360.0 / len(files)
        for index, path in enumerate(files):
            views.append(
                {
                    "path": path,
                    "azimuth": index * step,
                    "elevation": elevation,
                }
            )

    add_ring(tree_root, 0.0)
    for elevation, folder_name in ELEVATION_FOLDERS.items():
        if elevation == 0.0:
            continue
        folder = tree_root / folder_name
        if folder.exists():
            add_ring(folder, elevation)

    if not views:
        raise FileNotFoundError(f"No PNG views found in {tree_root}")
    return views


def choose_context_views(views, count):
    count = max(1, min(count, len(views)))
    if count == 1:
        return [views[0]]

    chosen = []
    for index in range(count):
        position = round(index * (len(views) - 1) / (count - 1))
        chosen.append(views[position])
    return chosen


class MultiTreePreview:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(torch.cuda.get_device_name(0))
            torch.backends.cudnn.benchmark = True

        saved = torch.load(args.checkpoint, map_location=self.device)
        config_values = saved.get("config")
        if config_values is None:
            raise ValueError("Checkpoint does not contain multi-tree config data")

        checkpoint_config = MultiTreeConfig(**config_values)
        self.config = MultiTreeConfig(**config_values)
        self.config.image_resolution = args.render_resolution
        if args.samples_per_ray is not None:
            self.config.samples_per_ray = args.samples_per_ray

        self.model = MultiTreeModel(checkpoint_config).to(self.device)
        self.model.load_state_dict(saved.get("model", saved))
        self.model.eval()

        self.tree_root = Path(args.tree)
        self.views = scan_views(self.tree_root)
        self.context_views = choose_context_views(self.views, args.context_views)
        print(
            f"Building memory from {len(self.context_views)} views in "
            f"{self.tree_root}"
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (checkpoint_config.image_resolution, checkpoint_config.image_resolution),
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )

        self.memory = self.build_memory(checkpoint_config)

        self.azimuth = 0.0
        self.elevation = 15.0
        self.playing = True
        self.running = True
        self.last_time = time.perf_counter()
        self.fps_start = self.last_time
        self.fps_frames = 0
        self.live_fps = 0.0
        self.show_uncertainty = False

        self.root = tk.Tk()
        self.root.title("AIModel Multi-Tree Memory Viewer")
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
            text="mouse/arrows: orbit | space: pause | U: uncertainty | 1/2/3/4: quality",
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
        self.root.bind("<Key-u>", lambda event: self.toggle_uncertainty())
        self.root.bind("<Key-U>", lambda event: self.toggle_uncertainty())
        self.root.bind("<Key-1>", lambda event: self.set_quality(96, 12))
        self.root.bind("<Key-2>", lambda event: self.set_quality(128, 24))
        self.root.bind("<Key-3>", lambda event: self.set_quality(256, 32))
        self.root.bind("<Key-4>", lambda event: self.set_quality(512, 48))
        self.root.bind("<Button-1>", self.start_drag)
        self.root.bind("<B1-Motion>", self.drag)

        self.drag_x = None
        self.drag_y = None
        self.photo = None
        self.root.after(0, self.update)

    def build_memory(self, checkpoint_config):
        memory = self.model.initial_memory(1, self.device)
        with torch.inference_mode():
            for index, view in enumerate(self.context_views, start=1):
                with Image.open(view["path"]) as image:
                    tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
                azimuth = torch.tensor([view["azimuth"]], device=self.device)
                elevation = torch.tensor([view["elevation"]], device=self.device)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    updated = self.model.observe(
                        tensor,
                        azimuth,
                        elevation,
                        memory,
                    )
                memory = updated.float()
                print(
                    f"Observed {index}/{len(self.context_views)}: "
                    f"az={view['azimuth']:.2f}, el={view['elevation']:.2f}"
                )
        return memory

    def close(self):
        self.running = False
        self.root.destroy()

    def toggle_play(self):
        self.playing = not self.playing

    def toggle_uncertainty(self):
        self.show_uncertainty = not self.show_uncertainty

    def set_quality(self, resolution, samples):
        self.config.image_resolution = resolution
        self.config.samples_per_ray = samples
        print(f"Quality: {resolution}x{resolution}, {samples} samples/ray")

    def step_azimuth(self, amount):
        self.playing = False
        self.azimuth = (self.azimuth + amount) % 360.0

    def step_elevation(self, amount):
        self.playing = False
        self.elevation = max(0.0, min(60.0, self.elevation + amount))

    def reset_camera(self):
        self.azimuth = 0.0
        self.elevation = 15.0

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
                60.0,
                self.elevation - delta_y * self.args.drag_sensitivity,
            ),
        )

    def render(self):
        resolution = self.config.image_resolution
        pixel_count = resolution * resolution
        rgb_parts = []
        uncertainty_parts = []

        for start in range(0, pixel_count, self.args.ray_chunk):
            end = min(start + self.args.ray_chunk, pixel_count)
            pixel_indices = torch.arange(start, end, device=self.device)
            origins, directions = make_rays(
                torch.tensor([self.azimuth], device=self.device),
                torch.tensor([self.elevation], device=self.device),
                self.config,
                self.device,
                pixel_indices,
            )
            rgb, _, uncertainty = self.model.render_rays(
                self.memory,
                origins,
                directions,
            )
            rgb_parts.append(rgb[0])
            uncertainty_parts.append(uncertainty[0])

        rgb = torch.cat(rgb_parts).reshape(resolution, resolution, 3)
        uncertainty = torch.cat(uncertainty_parts).reshape(resolution, resolution)
        return rgb, uncertainty

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
                rgb, uncertainty = self.render()

        if self.show_uncertainty:
            uncertainty = uncertainty.float()
            uncertainty = uncertainty / uncertainty.max().clamp_min(1e-6)
            display = uncertainty[..., None].expand(-1, -1, 3)
            mode = "UNCERTAINTY"
        else:
            display = rgb
            mode = "RGB"

        image = tensor_to_image(display)
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
                f"Context: {len(self.context_views)} views   "
                f"Azimuth: {self.azimuth:7.2f}   "
                f"Elevation: {self.elevation:6.2f}   "
                f"FPS: {self.live_fps:6.2f}   "
                f"Render: {self.config.image_resolution}²   "
                f"{mode}   {state}"
            )
        )
        self.root.title(
            f"AIModel Multi-Tree Viewer - step checkpoint - {self.live_fps:.1f} FPS"
        )
        self.root.after(1, self.update)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/multi_tree/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--tree",
        default="data/orbit_tree_002",
        help="Tree folder whose views are used to build recurrent memory",
    )
    parser.add_argument("--context-views", type=int, default=8)
    parser.add_argument("--render-resolution", type=int, default=128)
    parser.add_argument("--samples-per-ray", type=int, default=None)
    parser.add_argument("--ray-chunk", type=int, default=16384)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--speed", type=float, default=24.0)
    parser.add_argument("--drag-sensitivity", type=float, default=0.5)
    args = parser.parse_args()

    app = MultiTreePreview(args)
    app.run()


if __name__ == "__main__":
    main()
