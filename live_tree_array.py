import argparse
import math
import time
import tkinter as tk
from dataclasses import dataclass

import torch
from PIL import Image, ImageTk

from tree_field import CameraConfig, DenseTreeField, make_rays


@dataclass
class TreeInstance:
    position: tuple[float, float, float]
    yaw_degrees: float
    scale: float


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(image, mode="RGB")


def inverse_transform_rays(ray_origins, ray_directions, instance, device):
    position = torch.tensor(
        instance.position,
        dtype=ray_origins.dtype,
        device=device,
    )
    centered = ray_origins - position

    yaw = math.radians(instance.yaw_degrees)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)

    origin_x = cosine * centered[..., 0] - sine * centered[..., 2]
    origin_y = centered[..., 1]
    origin_z = sine * centered[..., 0] + cosine * centered[..., 2]

    direction_x = cosine * ray_directions[..., 0] - sine * ray_directions[..., 2]
    direction_y = ray_directions[..., 1]
    direction_z = sine * ray_directions[..., 0] + cosine * ray_directions[..., 2]

    local_origins = torch.stack((origin_x, origin_y, origin_z), dim=-1)
    local_directions = torch.stack(
        (direction_x, direction_y, direction_z),
        dim=-1,
    )

    local_origins = local_origins / instance.scale
    return local_origins, local_directions


def make_tree_grid(grid_size, spacing):
    instances = []
    offset = (grid_size - 1) * 0.5

    for row in range(grid_size):
        for column in range(grid_size):
            x = (column - offset) * spacing
            z = (row - offset) * spacing
            index = row * grid_size + column

            yaw = (index * 47.0) % 360.0
            scale = 0.85 + 0.075 * (index % 5)
            instances.append(
                TreeInstance(
                    position=(x, 0.0, z),
                    yaw_degrees=yaw,
                    scale=scale,
                )
            )

    return instances


class TreeArrayPreview:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(torch.cuda.get_device_name(0))
            torch.backends.cudnn.benchmark = True

        saved = torch.load(args.checkpoint, map_location=self.device)
        field_resolution = saved.get("field_resolution", 64)
        checkpoint_config = CameraConfig(**saved.get("camera_config", {}))

        self.model = DenseTreeField(field_resolution).to(self.device)
        self.model.load_state_dict(saved.get("model", saved))
        self.model.eval()

        self.config = CameraConfig(
            width=args.render_resolution,
            height=args.render_resolution,
            fov_degrees=args.fov or checkpoint_config.fov_degrees,
            radius=args.camera_radius,
            near=args.near,
            far=args.far,
            samples_per_ray=args.samples_per_ray,
        )

        self.instances = make_tree_grid(args.grid_size, args.spacing)
        print(f"Instances: {len(self.instances)}")

        self.azimuth = 0.0
        self.elevation = 18.0
        self.playing = True
        self.running = True
        self.last_time = time.perf_counter()
        self.fps_start = self.last_time
        self.fps_frames = 0
        self.live_fps = 0.0

        self.root = tk.Tk()
        self.root.title("AIModel Tree Array")
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
            text="1: 128²  2: 256²  3: 512²  |  arrows/mouse: orbit  |  space: pause",
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
        self.root.bind("<Key-1>", lambda event: self.set_quality(128, 20))
        self.root.bind("<Key-2>", lambda event: self.set_quality(256, 32))
        self.root.bind("<Key-3>", lambda event: self.set_quality(512, 48))
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
        print(f"Array quality: {resolution}x{resolution}, {samples} samples/ray")

    def step_azimuth(self, amount):
        self.playing = False
        self.azimuth = (self.azimuth + amount) % 360.0

    def step_elevation(self, amount):
        self.playing = False
        self.elevation = max(0.0, min(60.0, self.elevation + amount))

    def reset_camera(self):
        self.azimuth = 0.0
        self.elevation = 18.0

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

    def render_instance(self, world_origins, world_directions, instance):
        local_origins, local_directions = inverse_transform_rays(
            world_origins,
            world_directions,
            instance,
            self.device,
        )

        local_config = CameraConfig(
            width=self.config.width,
            height=self.config.height,
            fov_degrees=self.config.fov_degrees,
            radius=self.config.radius / instance.scale,
            near=self.config.near / instance.scale,
            far=self.config.far / instance.scale,
            samples_per_ray=self.config.samples_per_ray,
        )

        rgb_parts = []
        opacity_parts = []
        depth_parts = []
        total_rays = local_origins.shape[1]

        for start in range(0, total_rays, self.args.ray_chunk):
            end = min(start + self.args.ray_chunk, total_rays)
            rgb, opacity, depth = self.model.render_rays(
                local_origins[:, start:end],
                local_directions[:, start:end],
                local_config,
                randomized=False,
            )
            rgb_parts.append(rgb)
            opacity_parts.append(opacity)
            depth_parts.append(depth * instance.scale)

        return (
            torch.cat(rgb_parts, dim=1)[0],
            torch.cat(opacity_parts, dim=1)[0],
            torch.cat(depth_parts, dim=1)[0],
        )

    def render_array(self):
        world_origins, world_directions = make_rays(
            self.azimuth,
            self.elevation,
            self.config,
            self.device,
        )

        colors = []
        opacities = []
        depths = []

        for instance in self.instances:
            rgb, opacity, depth = self.render_instance(
                world_origins,
                world_directions,
                instance,
            )
            colors.append(rgb)
            opacities.append(opacity)
            depths.append(
                torch.where(
                    opacity > 1e-4,
                    depth,
                    torch.full_like(depth, float("inf")),
                )
            )

        colors = torch.stack(colors, dim=0)
        opacities = torch.stack(opacities, dim=0).clamp(0.0, 1.0)
        depths = torch.stack(depths, dim=0)

        order = torch.argsort(depths, dim=0)
        gather_rgb = order[..., None].expand(-1, -1, 3)
        sorted_colors = torch.gather(colors, 0, gather_rgb)
        sorted_opacities = torch.gather(opacities, 0, order)

        ray_count = sorted_opacities.shape[1]
        output = torch.zeros((ray_count, 3), device=self.device)
        transmittance = torch.ones(ray_count, device=self.device)

        for index in range(len(self.instances)):
            alpha = sorted_opacities[index]
            output = output + transmittance[:, None] * alpha[:, None] * sorted_colors[index]
            transmittance = transmittance * (1.0 - alpha)

        return output.reshape(self.config.height, self.config.width, 3)

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
                rgb = self.render_array()

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
                f"Trees: {len(self.instances)}   "
                f"Azimuth: {self.azimuth:7.2f}   "
                f"Elevation: {self.elevation:6.2f}   "
                f"FPS: {self.live_fps:6.2f}   "
                f"Render: {self.config.width}²   "
                f"Samples: {self.config.samples_per_ray}   "
                f"{state}"
            )
        )
        self.root.title(
            f"AIModel Tree Array - {len(self.instances)} trees - {self.live_fps:.1f} FPS"
        )
        self.root.after(1, self.update)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=2.6)
    parser.add_argument("--camera-radius", type=float, default=10.0)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=18.0)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--render-resolution", type=int, default=128)
    parser.add_argument("--samples-per-ray", type=int, default=20)
    parser.add_argument("--ray-chunk", type=int, default=16384)
    parser.add_argument("--speed", type=float, default=18.0)
    parser.add_argument("--drag-sensitivity", type=float, default=0.5)
    args = parser.parse_args()

    if args.grid_size < 1:
        raise ValueError("--grid-size must be at least 1")
    if args.spacing <= 0:
        raise ValueError("--spacing must be positive")

    app = TreeArrayPreview(args)
    app.run()


if __name__ == "__main__":
    main()
