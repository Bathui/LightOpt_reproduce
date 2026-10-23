"""LightOpt light forward model: interfaces and implementation TODOs.

Suggested order: compute_light_geometry -> distance_attenuation ->
evaluate_point_light -> angular_attenuation -> evaluate_spot_light -> evaluate_irradiance.

Conventions:
- N is the number of sample records. Use consistent world units for positions,
  ranges, and attenuation smoothness (which has units of distance squared).
- All tensors share a device and dtype. Scalars are zero-dimensional; vectors are [3].
- color_linear is linear-space base RGB. Cookies, shadows, color-temperature
  conversion, BRDFs, and material colors are excluded. Handle color temperature
  explicitly rather than ignoring the exported useColorTemperature field.
- Cone angles are half angles in radians: halve Unity full angles before conversion.
- Preserve the PyTorch graph in forward functions: do not use NumPy, math, .item(),
  .detach(), or rewrap existing tensors. Loader conversion and validation are separate.
- Do not clamp output to [0,1] or apply area_weights here.
- Two-sided foliage samples are already paired. Clamp negative cosines to zero;
  do not use abs or duplicate samples again.

Loading helpers parse JSON into fixed parameters. Forward functions do not read
NPZ files, create trainable parameters, save targets, or update lights."""
from dataclasses import dataclass
from typing import Literal, Sequence

from torch import Tensor
import json

import torch
import math
@dataclass
class LightParameters:
    """Internal parameters of one light; the caller converts JSON fields and validates inputs.
    
    Requires intensity >= 0, range > 0, and attenuation_smoothness > 0.
    attenuation_smoothness is a in paper Eq. (4), measured in distance squared.
    It is unrelated to material Smoothness and must be configured explicitly because
    it is not included in the current Unity export.
    Spot lights require direction and half angles satisfying 0 <= inner < outer < pi.
    direction points outward from the light, opposite to sample-to-light omega."""

    kind: Literal["Point", "Spot"]
    position: Tensor                      # [3]
    color_linear: Tensor                  # [3]
    intensity: Tensor                     # scalar
    range: Tensor                         # scalar
    attenuation_smoothness: Tensor        # scalar
    direction: Tensor | None = None       # [3], Spot only
    inner_half_angle: Tensor | None = None # scalar, radians
    outer_half_angle: Tensor | None = None # scalar, radians

# load light information from parsed lights to initialize LightParameters
def single_light_info(
    light_record: dict,
    attenuation_smoothness: float = 1.0,
    device: str = "cpu",
) -> LightParameters:
    """Convert one exported light record to tensor parameters and validate its inputs."""
    if not isinstance(light_record, dict):
        raise ValueError("Light record must be a dictionary")

    kind = light_record["type"]
    if kind not in ("Point", "Spot"):
        raise ValueError(f"Not supported type of Light: {kind}")

    if light_record.get("hasCookie", False):
        raise ValueError("Please turn off Cookie")
    if light_record.get("useColorTemperature", False):
        raise ValueError("Please disable color temperature or export its effective color")

    def to_tensor(value):
        return torch.tensor(value, dtype=torch.float32, device=device)

    # Validate Python scalars before creating tensors to avoid repeated GPU synchronization.
    intensity_value = float(light_record["intensity"])
    range_value = float(light_record["range"])
    smoothness_value = float(attenuation_smoothness)

    if not all(math.isfinite(v) for v in (intensity_value, range_value, smoothness_value)):
        raise ValueError("Intensity, range and attenuation smoothness must be finite")
    if intensity_value < 0 or range_value <= 0 or smoothness_value <= 0:
        raise ValueError("Require intensity >= 0, range > 0 and attenuation smoothness > 0")

    intensity = to_tensor(intensity_value)
    light_range = to_tensor(range_value)
    smoothness = to_tensor(smoothness_value)
    if not all(torch.isfinite(v).all() for v in (intensity, light_range, smoothness)):
        raise ValueError("Scalar parameters must be representable as finite float32 values")
    if light_range <= 0 or smoothness <= 0:
        raise ValueError("Range and attenuation smoothness must remain positive in float32")

    direction = None
    inner_angle = None
    outer_angle = None

    if kind == "Spot":
        d = light_record["direction"]
        direction = to_tensor([d["x"], d["y"], d["z"]])
        if not torch.isfinite(direction).all():
            raise ValueError("Direction must be finite")
        length = torch.linalg.vector_norm(direction)
        if not torch.isfinite(length) or length <= 1e-8:
            raise ValueError("Direction must have a finite nonzero length")
        direction = direction / length

        # Convert Unity full angles to radian half angles; validate finiteness and ordering.
        inner = math.radians(float(light_record["innerSpotAngleDegrees"]) / 2)
        outer = math.radians(float(light_record["outerSpotAngleDegrees"]) / 2)
        if not (math.isfinite(inner) and math.isfinite(outer)
                and 0 <= inner < outer < math.pi):
            raise ValueError("Spot half angles must satisfy 0 <= inner < outer < pi")
        inner_angle = to_tensor(inner)
        outer_angle = to_tensor(outer)
        if not inner_angle < outer_angle:
            raise ValueError("Spot inner/outer angles collapse to the same float32 value")

    position_record = light_record["position"]
    color_record = light_record["colorLinear"]
    position = to_tensor([position_record[k] for k in ("x", "y", "z")])
    color = to_tensor([color_record[k] for k in ("r", "g", "b")])
    if not torch.isfinite(position).all():
        raise ValueError("Position must be finite")
    if not torch.isfinite(color).all() or (color < 0).any():
        raise ValueError("Color must be finite and nonnegative")

    return LightParameters(
        kind=kind,
        position=position,
        color_linear=color,
        intensity=intensity,
        range=light_range,
        attenuation_smoothness=smoothness,
        direction=direction,
        inner_half_angle=inner_angle,
        outer_half_angle=outer_angle,
    )


def load_lights(
    path: str,
    attenuation_smoothness: float = 1.0,
    device: str = "cpu",
) -> list[LightParameters]:
    """Read lights.json from a sample directory; preserve record order and return [] for empty input."""
    with open(path, "r", encoding="utf-8-sig") as file:
        light_records = json.load(file)
    if not isinstance(light_records, list):
        raise ValueError('The outermost layer of "lights.json" must be a list')

    lights = []
    for index, light_record in enumerate(light_records):
        if not isinstance(light_record, dict):
            raise ValueError(f"Light record {index + 1} must be a dictionary")
        try:
            light = single_light_info(
                light_record,
                attenuation_smoothness=attenuation_smoothness,
                device=device,
            )
        except (KeyError, TypeError, ValueError) as error:
            name = light_record.get("path", light_record.get("id", "unnamed"))
            raise ValueError(f"Light {index + 1} read failed: {name}; {error}") from error
        lights.append(light)
    return lights


def compute_light_geometry(
    positions: Tensor,
    normals: Tensor,
    light_position: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    # Return (omega[N,3], distance_squared[N], cosine[N]).
   
    vectors = light_position - positions #[N, 3]
    distance_squared = torch.sum(vectors * vectors, dim=-1)  # [N]

    # Clamp before sqrt to keep normalization gradients finite at zero distance.
    safe_distance = torch.sqrt(torch.clamp_min(distance_squared, eps))  # [N]

    omega = vectors / safe_distance.unsqueeze(-1) # unit vector

    cosine = torch.sum(normals * omega, dim=-1)
    cosine = torch.clamp_min(cosine, 0.0)

    return omega, distance_squared, cosine

def distance_attenuation(
    distance_squared: Tensor,
    light_range: Tensor,
    attenuation_smoothness: Tensor,
) -> Tensor:
    """Paper Eq. (4): map squared distances [N] to distance attenuation [N].
    a / (a + d^2) * max(0, 1 - d^2 / r^2)^2
    Requires d^2 >= 0, r > 0, and a > 0. Falloff is zero at and beyond the range."""

    range_factor = torch.clamp_min(1.0 - distance_squared / light_range.square(), 0.0).square()

    a = attenuation_smoothness
    distance_factor = a / (a + distance_squared)

    return distance_factor * range_factor

def angular_attenuation(
    omega: Tensor,
    light_direction: Tensor,
    inner_half_angle: Tensor,
    outer_half_angle: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Spot branch of paper Eq. (5): map omega[N,3] to angular attenuation [N].
    
    clamp((dot(-omega, direction) - cos(outer)) /
          (cos(inner) - cos(outer)), 0, 1)^2
    """
    direction_squared_len = torch.sum(light_direction.square())
    direction = light_direction / torch.sqrt(torch.clamp_min(direction_squared_len, eps))

    angle_cos = torch.sum(-omega * direction, dim=-1)

    inner_cosine = torch.cos(inner_half_angle)
    outer_cosine = torch.cos(outer_half_angle)

    angular_factor = torch.clamp((angle_cos - outer_cosine) / (inner_cosine - outer_cosine), 0.0, 1.0).square()

    return angular_factor

def evaluate_point_light(
    positions: Tensor,
    normals: Tensor,
    light: LightParameters,
) -> Tensor:
    """Return RGB irradiance [N,3] from one point light at N samples.
    Eq. (3) Vi * ai * ci * I * Ai * (n * wi)
    Also serves as the spot-light base term; direction and cone angles are unused."""
    
    _, distance_squared, cosine = compute_light_geometry(
        positions,
        normals,
        light.position,
    )

    attenuation = distance_attenuation(distance_squared,
                                       light.range,
                                       light.attenuation_smoothness
    )

    scale_light = attenuation *  cosine * light.intensity
    irradiance = scale_light.unsqueeze(-1) * light.color_linear

    return irradiance


def evaluate_spot_light(
    positions: Tensor,
    normals: Tensor,
    light: LightParameters
) -> Tensor:
    """Return RGB irradiance [N,3] from one spot light at N samples."""
    # The only difference between spot light and point light is their angular factor term.
    if (
        light.direction is None
        or light.inner_half_angle is None
        or light.outer_half_angle is None
    ):
        raise ValueError(
            "Spot lights require a direction and inner/outer half angles"
        )

    # Point light calculation is part of the spotlight calculation
    base_irradiance = evaluate_point_light(positions, normals, light)

    omega, _, _ = compute_light_geometry(positions, normals, light.position, )

    angular_factor = angular_attenuation(
        omega,
        light.direction, 
        light.inner_half_angle, 
        light.outer_half_angle, )

    spotLight_irradiance = angular_factor.unsqueeze(-1) * base_irradiance
    return spotLight_irradiance


def evaluate_irradiance(
    positions: Tensor,
    normals: Tensor,
    lights: Sequence[LightParameters],
) -> Tensor:
    """Sum all light contributions and return total RGB irradiance [N,3].
    
    Supports both fixed-target generation and optimization. Do not disable gradients
    internally. The caller uses torch.no_grad() when generating a fixed target."""
    irradiance = torch.zeros_like(positions) # [N, 3]

    for light in lights:
        if light.kind == "Point":
            irradiance = irradiance + evaluate_point_light(
                positions,
                normals,
                light
            )
        elif light.kind == "Spot":
            irradiance = irradiance + evaluate_spot_light(
                positions,
                normals,
                light
            )
        else:
            raise ValueError(
                f"Unsupported light type: {light.kind}"
            )

    return irradiance