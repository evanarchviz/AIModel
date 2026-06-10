import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CameraConfig:
    width: int = 128
    height: int = 128
    fov_degrees: float = 35.0
    radius: float = 3.0
    near: float = 1.5
    far: float = 4.5
    samples_per_ray: int = 48


def camera_origin(azimuth_degrees, elevation_degrees, radius, device):
    azimuth = torch.as_tensor(
        azimuth_degrees,
        dtype=torch.float32,
        device=device,
    ) * math.pi / 180.0
    elevation = torch.as_tensor(
        elevation_degrees,
        dtype=torch.float32,
        device=device,
    ) * math.pi / 180.0

    x = radius * torch.cos(elevation) * torch.sin(azimuth)
    y = radius * torch.sin(elevation)
    z = radius * torch.cos(elevation) * torch.cos(azimuth)
    return torch.stack((x, y, z), dim=-1)


def make_camera_basis(origin):
    forward = F.normalize(-origin, dim=-1)
    world_up = torch.tensor(
        [0.0, 1.0, 0.0],
        dtype=origin.dtype,
        device=origin.device,
    ).expand_as(forward)

    right = F.normalize(torch.cross(forward, world_up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1)
    return right, up, forward


def make_rays(
    azimuth_degrees,
    elevation_degrees,
    config,
    device,
    pixel_indices=None,
):
    origin = camera_origin(
        azimuth_degrees,
        elevation_degrees,
        config.radius,
        device,
    )
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)

    right, up, forward = make_camera_basis(origin)

    if pixel_indices is None:
        ys, xs = torch.meshgrid(
            torch.arange(config.height, device=device),
            torch.arange(config.width, device=device),
            indexing="ij",
        )
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
    else:
        pixel_indices = pixel_indices.to(device)
        ys = torch.div(pixel_indices, config.width, rounding_mode="floor")
        xs = pixel_indices % config.width

    aspect = config.width / config.height
    tan_half_fov = math.tan(math.radians(config.fov_degrees) * 0.5)

    px = (
        ((xs.float() + 0.5) / config.width) * 2.0 - 1.0
    ) * tan_half_fov * aspect
    py = -(
        ((ys.float() + 0.5) / config.height) * 2.0 - 1.0
    ) * tan_half_fov

    directions = (
        forward[:, None, :]
        + right[:, None, :] * px[None, :, None]
        + up[:, None, :] * py[None, :, None]
    )
    directions = F.normalize(directions, dim=-1)
    origins = origin[:, None, :].expand_as(directions)
    return origins, directions


class DenseTreeField(nn.Module):
    def __init__(self, resolution=64):
        super().__init__()
        self.resolution = resolution
        self.density_grid = nn.Parameter(
            torch.full((1, 1, resolution, resolution, resolution), -6.0)
        )
        self.color_grid = nn.Parameter(
            torch.zeros((1, 3, resolution, resolution, resolution))
        )

    def sample_field(self, points):
        original_shape = points.shape[:-1]
        inside = (points.abs() <= 1.0).all(dim=-1)
        grid = points.reshape(1, 1, 1, -1, 3)

        density_logits = F.grid_sample(
            self.density_grid,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(*original_shape)

        color_logits = F.grid_sample(
            self.color_grid,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(3, *original_shape).movedim(0, -1)

        density = F.softplus(density_logits) * inside.to(density_logits.dtype)
        color = torch.sigmoid(color_logits) * inside[..., None].to(color_logits.dtype)
        return density, color

    def render_rays(self, ray_origins, ray_directions, config, randomized=False):
        batch, ray_count, _ = ray_origins.shape
        t_values = torch.linspace(
            config.near,
            config.far,
            config.samples_per_ray,
            device=ray_origins.device,
            dtype=ray_origins.dtype,
        )

        if randomized and self.training:
            interval = (config.far - config.near) / max(config.samples_per_ray - 1, 1)
            t_values = t_values + (torch.rand_like(t_values) - 0.5) * interval
            t_values = t_values.clamp(config.near, config.far)
            t_values, _ = torch.sort(t_values)

        points = (
            ray_origins[:, :, None, :]
            + ray_directions[:, :, None, :] * t_values[None, None, :, None]
        )

        density, color = self.sample_field(points)

        deltas = t_values[1:] - t_values[:-1]
        last_delta = deltas[-1:].clone()
        deltas = torch.cat((deltas, last_delta), dim=0)
        deltas = deltas[None, None, :].expand(batch, ray_count, -1)

        alpha = 1.0 - torch.exp(-density * deltas)
        transmittance = torch.cumprod(
            torch.cat(
                (
                    torch.ones_like(alpha[..., :1]),
                    1.0 - alpha + 1e-10,
                ),
                dim=-1,
            ),
            dim=-1,
        )[..., :-1]
        weights = alpha * transmittance

        rgb = torch.sum(weights[..., None] * color, dim=-2)
        opacity = torch.sum(weights, dim=-1)
        depth = torch.sum(weights * t_values[None, None, :], dim=-1)
        return rgb, opacity, depth

    def render_image(self, azimuth, elevation, config, device, ray_chunk=2048):
        origins, directions = make_rays(
            azimuth,
            elevation,
            config,
            device,
        )

        outputs = []
        opacities = []
        depths = []
        total_rays = origins.shape[1]

        for start in range(0, total_rays, ray_chunk):
            end = min(start + ray_chunk, total_rays)
            rgb, opacity, depth = self.render_rays(
                origins[:, start:end],
                directions[:, start:end],
                config,
                randomized=False,
            )
            outputs.append(rgb)
            opacities.append(opacity)
            depths.append(depth)

        rgb = torch.cat(outputs, dim=1)
        opacity = torch.cat(opacities, dim=1)
        depth = torch.cat(depths, dim=1)

        rgb = rgb.reshape(config.height, config.width, 3)
        opacity = opacity.reshape(config.height, config.width)
        depth = depth.reshape(config.height, config.width)
        return rgb, opacity, depth
