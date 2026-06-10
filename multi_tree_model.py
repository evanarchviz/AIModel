import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiTreeConfig:
    image_resolution: int = 128
    memory_dim: int = 256
    plane_channels: int = 16
    plane_resolution: int = 32
    samples_per_ray: int = 32
    fov_degrees: float = 35.0
    camera_radius: float = 3.0
    near: float = 1.5
    far: float = 4.5


def camera_code(azimuth_degrees, elevation_degrees, device):
    azimuth = torch.as_tensor(azimuth_degrees, dtype=torch.float32, device=device)
    elevation = torch.as_tensor(elevation_degrees, dtype=torch.float32, device=device)
    azimuth = torch.deg2rad(azimuth)
    elevation = torch.deg2rad(elevation)
    return torch.stack(
        (
            torch.sin(azimuth),
            torch.cos(azimuth),
            torch.sin(elevation),
            torch.cos(elevation),
        ),
        dim=-1,
    )


def camera_origin(azimuth_degrees, elevation_degrees, radius, device):
    azimuth = torch.deg2rad(
        torch.as_tensor(azimuth_degrees, dtype=torch.float32, device=device)
    )
    elevation = torch.deg2rad(
        torch.as_tensor(elevation_degrees, dtype=torch.float32, device=device)
    )
    x = radius * torch.cos(elevation) * torch.sin(azimuth)
    y = radius * torch.sin(elevation)
    z = radius * torch.cos(elevation) * torch.cos(azimuth)
    return torch.stack((x, y, z), dim=-1)


def make_rays(azimuth, elevation, config, device, pixel_indices):
    origin = camera_origin(azimuth, elevation, config.camera_radius, device)
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)

    forward = F.normalize(-origin, dim=-1)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(forward)
    right = F.normalize(torch.cross(forward, world_up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1)

    pixel_indices = pixel_indices.to(device)
    ys = torch.div(pixel_indices, config.image_resolution, rounding_mode="floor")
    xs = pixel_indices % config.image_resolution

    tan_half = math.tan(math.radians(config.fov_degrees) * 0.5)
    px = (((xs.float() + 0.5) / config.image_resolution) * 2.0 - 1.0) * tan_half
    py = -(((ys.float() + 0.5) / config.image_resolution) * 2.0 - 1.0) * tan_half

    directions = (
        forward[:, None, :]
        + right[:, None, :] * px[None, :, None]
        + up[:, None, :] * py[None, :, None]
    )
    directions = F.normalize(directions, dim=-1)
    origins = origin[:, None, :].expand_as(directions)
    return origins, directions


class ViewEncoder(nn.Module):
    def __init__(self, memory_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 192, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Sequential(
            nn.Linear(192 + 4, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(self, image, pose_code):
        features = self.cnn(image).flatten(1)
        return self.project(torch.cat((features, pose_code), dim=-1))


class RecurrentTreeMemory(nn.Module):
    def __init__(self, memory_dim):
        super().__init__()
        self.cell = nn.GRUCell(memory_dim, memory_dim)

    def forward(self, observation, memory):
        return self.cell(observation, memory)


class TriPlaneGenerator(nn.Module):
    def __init__(self, memory_dim, plane_channels, plane_resolution):
        super().__init__()
        self.plane_channels = plane_channels
        self.plane_resolution = plane_resolution
        output_size = 3 * plane_channels * plane_resolution * plane_resolution
        self.net = nn.Sequential(
            nn.Linear(memory_dim, 512),
            nn.SiLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, memory):
        planes = self.net(memory)
        return planes.view(
            memory.shape[0],
            3,
            self.plane_channels,
            self.plane_resolution,
            self.plane_resolution,
        )


class FieldDecoder(nn.Module):
    def __init__(self, plane_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(plane_channels * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 5),
        )

    def forward(self, features):
        raw = self.net(features)
        density = F.softplus(raw[..., 0] - 1.0)
        color = torch.sigmoid(raw[..., 1:4])
        uncertainty = F.softplus(raw[..., 4])
        return density, color, uncertainty


class MultiTreeModel(nn.Module):
    def __init__(self, config: MultiTreeConfig):
        super().__init__()
        self.config = config
        self.encoder = ViewEncoder(config.memory_dim)
        self.memory_updater = RecurrentTreeMemory(config.memory_dim)
        self.plane_generator = TriPlaneGenerator(
            config.memory_dim,
            config.plane_channels,
            config.plane_resolution,
        )
        self.field_decoder = FieldDecoder(config.plane_channels)

    def initial_memory(self, batch_size, device):
        return torch.zeros(batch_size, self.config.memory_dim, device=device)

    def observe(self, image, azimuth, elevation, memory):
        pose = camera_code(azimuth, elevation, image.device)
        observation = self.encoder(image, pose)
        return self.memory_updater(observation, memory)

    def _sample_planes(self, planes, points):
        batch, rays, samples, _ = points.shape
        coords = (
            points[..., [0, 1]],
            points[..., [0, 2]],
            points[..., [1, 2]],
        )
        sampled = []
        for plane_index, coordinate in enumerate(coords):
            grid = coordinate.reshape(batch, rays * samples, 1, 2)
            value = F.grid_sample(
                planes[:, plane_index],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            value = value.squeeze(-1).transpose(1, 2)
            sampled.append(value.reshape(batch, rays, samples, -1))
        return torch.cat(sampled, dim=-1)

    def render_rays(self, memory, origins, directions):
        planes = self.plane_generator(memory)
        t_values = torch.linspace(
            self.config.near,
            self.config.far,
            self.config.samples_per_ray,
            device=origins.device,
            dtype=origins.dtype,
        )
        points = origins[:, :, None, :] + directions[:, :, None, :] * t_values[None, None, :, None]
        inside = (points.abs() <= 1.0).all(dim=-1)
        features = self._sample_planes(planes, points)
        density, color, uncertainty = self.field_decoder(features)
        density = density * inside.to(density.dtype)
        color = color * inside[..., None].to(color.dtype)

        deltas = t_values[1:] - t_values[:-1]
        deltas = torch.cat((deltas, deltas[-1:]), dim=0)
        alpha = 1.0 - torch.exp(-density * deltas[None, None, :])
        transmittance = torch.cumprod(
            torch.cat((torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10), dim=-1),
            dim=-1,
        )[..., :-1]
        weights = alpha * transmittance

        rgb = torch.sum(weights[..., None] * color, dim=-2)
        opacity = torch.sum(weights, dim=-1)
        ray_uncertainty = torch.sum(weights * uncertainty, dim=-1)
        return rgb, opacity, ray_uncertainty
