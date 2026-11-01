import math
from typing import Iterable, Literal, Sequence, cast

import torch
from torch import Tensor, nn

from algo.light import LightParameters


class TrainableLight(nn.Module):
    """Optimize intensity while keeping the other light fields fixed."""

    kind: Literal["Point", "Spot"]
    position: nn.Parameter
    color_linear: nn.Parameter
    intensity: nn.Parameter
    range: nn.Parameter
    attenuation_smoothness: Tensor
    direction: nn.Parameter | None
    inner_half_angle: Tensor | None
    outer_half_angle: Tensor | None
    initial_direction: Tensor | None
    
    def __init__(
        self,
        light: LightParameters,
        intensity_scale: float,
    ):
        super().__init__()
        self.kind = light.kind

        if light.kind not in ("Point", "Spot"):
            raise ValueError(
                "Only Point and Spot lights are supported"
            )

        if light.intensity.ndim != 0:
            raise ValueError("Light intensity must be a scalar")

        if (
            not torch.isfinite(light.intensity)
            or light.intensity < 0
        ):
            raise ValueError(
                "Light intensity must be finite and nonnegative"
            )

        for name in (
            "attenuation_smoothness",
            "inner_half_angle",
            "outer_half_angle",
        ):
            value = getattr(light, name)
            # These properties are fixed not for optimizer
            self.register_buffer(
                name,
                None if value is None else value.detach().clone(),
            )

        self.intensity = nn.Parameter(
            light.intensity.detach().clone() * intensity_scale
        )
        self.position = nn.Parameter(
            light.position.detach().clone()
        )

        self.range = nn.Parameter(
            light.range.detach().clone()
        )

        self.color_linear = nn.Parameter(
            light.color_linear.detach().clone()
        )

        if self.kind == "Spot":
            if light.direction is None:
                raise ValueError("Spot lights require a direction")

            direction = light.direction.detach().clone()
            length = torch.linalg.vector_norm(direction)

            if not torch.isfinite(direction).all() or length <= 1e-8:
                raise ValueError(
                    "Spot direction must be finite and nonzero"
                )

            direction = direction / length
            self.direction = nn.Parameter(direction.clone())

            self.register_buffer(
                "initial_direction",
                direction.clone(),
            )
        else:
            self.register_parameter("direction", None)
            self.register_buffer("initial_direction", None)
    
    def forward(self) -> LightParameters:
        return LightParameters(
            kind=self.kind,
            position=self.position,
            color_linear=self.color_linear,
            intensity=self.intensity,
            range=self.range,
            attenuation_smoothness=self.attenuation_smoothness,
            direction=(
                None if self.direction is None else
                torch.nn.functional.normalize(
                    self.direction,
                    dim=0,
                    eps=1e-8,
                )
            ),
            inner_half_angle=self.inner_half_angle,
            outer_half_angle=self.outer_half_angle,
        )


class TrainableLights(nn.Module):
    """Keep light order and expose only intensities to the optimizer."""

    def __init__(
        self,
        lights: Sequence[LightParameters],
        intensity_scale: float = 0.5,
    ):
        super().__init__()

        if len(lights) == 0:
            raise ValueError("At least one light is required")

        if (
            not math.isfinite(intensity_scale)
            or intensity_scale < 0
        ):
            raise ValueError(
                "intensity_scale must be finite and nonnegative"
            )

        self.lights = nn.ModuleList([
            TrainableLight(light, intensity_scale)
            for light in lights
        ])

    def forward(self) -> list[LightParameters]:
        return [
            light()
            for light in cast(Iterable[TrainableLight], self.lights)
        ]

    @torch.no_grad()
    def project_constraints(self) -> None:
        for light in cast(Iterable[TrainableLight], self.lights):
            light.intensity.clamp_(min=0.0)
            light.range.clamp_(min=1e-4)
            light.color_linear.clamp_(min=0.0)

            direction = light.direction

            if direction is None:
                continue

            initial_direction = light.initial_direction

            if initial_direction is None:
                raise RuntimeError(
                    "Trainable direction requires an initial direction"
                )

            length = torch.linalg.vector_norm(direction)
            unit = direction / length.clamp_min(1e-8)

            direction.copy_(
                torch.where(
                    length > 1e-8,
                    unit,
                    initial_direction,
                )
            )
