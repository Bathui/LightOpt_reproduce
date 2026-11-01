from torch import Tensor
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from algo.light import LightParameters, evaluate_per_light

from config.loss_config import LossConfig

# helpler functions
def rgb_luminance(rgb: Tensor) -> Tensor:
    """Convert linear RGB [..., 3] to scalar luminance [...]."""
    coefficients = rgb.new_tensor([0.2126, 0.7152, 0.0722])
    return (rgb * coefficients).sum(dim=-1)

def point_geometry(
    positions: Tensor,
    lights: Sequence[LightParameters],
):
    if any(light.kind != "Point" for light in lights):
        raise ValueError("This prototype accepts Point lights only")

    light_positions = torch.stack([
       light.position for light in lights
    ]) # [M, 3]

    # [M, 1, 3] - [1, N, 3]
    samp2light = light_positions[:, None, :] - positions[None, :, :]
    dist = torch.linalg.vector_norm(
        samp2light,
        dim = -1
    )

    ranges = torch.stack(
        [light.range for light in lights]
    )[:, None]  # [M, 1]

    return samp2light, dist, ranges

def spot_geometry(
    positions: Tensor,
    lights: Sequence[LightParameters],
    eps: float = 1e-8,
):
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if len(lights) == 0:
        raise ValueError("spot_geometry requires at least one light")

    if any(
        light.kind != "Spot"
        or light.direction is None
        or light.outer_half_angle is None
        for light in lights
    ):
        raise ValueError(
            "Require Spot lights with direction and outer angle"
        )

    light_positions = torch.stack([
           light.position for light in lights
        ]) # [M, 3]
    
    # [M, 1, 3] - [1, N, 3]
    samp2light = light_positions[:, None, :] - positions[None, :, :]

    directions: list[Tensor] = []
    for light in lights:
        direction = light.direction
        if direction is None:
            raise ValueError(
                "Spot light must have a direction"
            )
        directions.append(direction)

    light_directions = F.normalize(
        torch.stack(directions, dim=0),
        dim=-1,
        eps=eps,
    )[:, None, :]  # [M,1,3]

    ranges = torch.stack(
        [light.range for light in lights]
    )[:, None]  # [M,1]

    outer_angles: list[Tensor] = []

    for light in lights:
        angle = light.outer_half_angle

        if angle is None:
            raise ValueError("Spot lights require outer_half_angle")

        outer_angles.append(angle)

    outer = torch.stack(outer_angles)[:, None]  # [M,1]

    samp2light = (
        light_positions[:, None, :]
        - positions[None, :, :]
    )  # [M,N,3]

    distances = torch.linalg.vector_norm(samp2light, dim=-1,)  # [M,N]
    

    ray = (
        -samp2light
        / distances.clamp_min(eps)[..., None]
    )  # [M,N,3]

    cosine = (ray * light_directions).sum(dim=-1)  # [M,N]

    sine = torch.linalg.vector_norm(
        torch.linalg.cross(ray, light_directions, dim=-1),
        dim=-1,
    )  # [M,N]

    cosine = torch.where(
        distances > eps,
        cosine,
        torch.ones_like(cosine),
    )

    angles = torch.atan2(sine, cosine)

    return samp2light, distances, ranges, angles, outer

# Eq.13: this makes sure the softmin is differentiable
def weighted_softmin(
        a: Tensor,
        b: Tensor,
        beta: float = 1.0
)->Tensor:
    values = torch.stack(
        (a, b),
        dim=-1,
    )  # [M,N,2]
    weights = torch.softmax(
        -beta * values,
        dim=-1,
    )  # [M,N,2]

    return (weights * values).sum(dim=-1)  # [M,N]

# loss function
def reconstruction_loss (
        E_pred: Tensor,
        E_target: Tensor,
        area_weights: Tensor
)->Tensor:
    #  Eq.6 in the paper
    #  Δs · Σ_{x∈S_all} (Ê(x) - E(x))²
    if E_pred.ndim != 2 or E_pred.shape[1] != 3:
        raise ValueError("E_pred must have shape [N, 3]")

    if E_target.shape != E_pred.shape:
        raise ValueError("E_target must have the same shape as E_pred")

    if area_weights.shape != (E_pred.shape[0],):
        raise ValueError("area_weights must have shape [N]")
    
    difference = E_pred - E_target          #[N, 3]

    # we need to sum the square error over all rgb channels
    error_per_sample = difference.square().sum(dim=-1) #[N]
    weighted_error = area_weights * error_per_sample  #[N]

    total_error = weighted_error.sum() # scalar: add up all sampled loss
    return total_error


# loss for point lights

def point_attraction_loss(
        positions: Tensor,
        normals: Tensor,
        lights: Sequence[LightParameters],
        E_pred: Tensor,
        E_target: Tensor,
        under_eps: float = 0.1,
        dark_eps: float = 1e-8,
        volume: float = 15.0, # from paper's experiment
         beta_p: float = 1.0
):
    # Eq.17 Σ_{x∈S_underlit}Σ_{l_i∈L_attraction} d_i(x)
    if len(lights) == 0:
        return E_pred.sum() * 0.0

    samp2light, distances, ranges = point_geometry(positions, lights)

    target_luminance = rgb_luminance(E_target) # [N]
    pred_luminance = rgb_luminance(E_pred) # [N]

    # underlit region is the region that should not be any dark regions initially
    underlit = (target_luminance > dark_eps) & (pred_luminance <= under_eps * target_luminance) # [N]

    # For each underlit sample, light must lie on the front-facing side
    front_facing = (samp2light * normals[None, :, :]).sum(dim=-1) > 0  # [M, N]

    # Refer to Eq.16, but doubt the paper might have a mistake or unclear illustration in this part
    search_radius = (ranges.pow(3) + 3.0 * volume / (4.0 * math.pi)
           ).pow (1.0 / 3.0)

    candidates = (
        underlit[None, :]
        & front_facing
        & (distances > ranges)
        & (distances < search_radius)
    )  # [M, N]

    geo_penalty = F.softplus(distances - ranges, beta=beta_p) # [M, N]

    attraction_Loss = (geo_penalty * candidates.to(geo_penalty.dtype)).sum()
    
    return attraction_Loss 

# Loss for Spot lights  
def spot_attraction_loss(
        positions: Tensor,
        normals: Tensor,
        lights: Sequence[LightParameters],
        E_pred: Tensor,
        E_target: Tensor,
        under_eps: float = 0.1,
        dark_eps: float = 1e-8,
        volume: float = 15.0, # from paper's experiment
        beta_m: float = 1.0,
        eps: float = 1e-8,
        *,
        sign_correction: bool = True,
):
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")
    if (
        normals.shape != positions.shape
        or E_pred.shape != positions.shape
        or E_target.shape != positions.shape
    ):
        raise ValueError("normals, E_pred and E_target must match positions [N,3]")
    if (
        not 0 <= under_eps <= 1
        or dark_eps < 0
        or volume < 0
        or beta_m <= 0
        or eps <= 0
    ):
        raise ValueError("Invalid attraction hyperparameters")
    if len(lights) == 0:
        return E_pred.sum() * 0.0

    samp2light, distances, ranges, angles, outer = spot_geometry(positions, lights, eps)

    target_luminance = rgb_luminance(E_target) # [N]
    pred_luminance = rgb_luminance(E_pred) # [N]

    # underlit region is the region that should not be any dark regions initially
    underlit = (target_luminance > dark_eps) & (pred_luminance <= under_eps * target_luminance) # [N]

    # For each underlit sample, light must lie on the front-facing side
    front_facing = (samp2light * normals[None, :, :]).sum(dim=-1) > 0  # [M, N]

    # Eq. (16) as printed: eta is the center-distance threshold in Eq. (15).
    eta = (
        ranges.pow(3) + 3.0 * volume / (4.0 * math.pi)
    ).pow(1.0 / 3.0) - ranges

    candidates = (
        underlit[None, :]
        & front_facing
        & (distances < eta)
    )  # [M, N]

    # Signed interior distances from paper Eq. (11).
    radial_inside = ranges - distances  # [M,N]

    cone_inside = (
        distances * torch.sin(outer - angles)
    )  # [M,N]

    # Preserve the paper's signed interior geometry for baseline comparisons.
    signed_inside = weighted_softmin(
        radial_inside,
        cone_inside,
        beta=beta_m,
    )  # [M,N]

    # Negate Attraction only; Overlap retains its original sign.
    geo_penalty = -signed_inside if sign_correction else signed_inside

    attraction_loss = (
        geo_penalty * candidates.to(geo_penalty.dtype)
    ).sum()

    return attraction_loss

# Mixed Integrated functions

def build_overlap_terms(
    positions: Tensor,
    lights: Sequence[LightParameters],
    cfg: LossConfig,
) -> tuple[Tensor, Tensor]:
    """Return candidate masks and geometric penalties, both [M,N]."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")

    masks = []
    penalties = []

    for light in lights:
        if light.kind == "Point":
            _, d, r = point_geometry(positions, [light])

            # Get candidates lie on doubled ranges
            mask = d < 2.0 * r # [M, N] bool tensor

            penalty = F.softplus(
                r - d,
                beta=cfg.beta_p,
            ) # [M, N]

        elif light.kind == "Spot":
            _, d, r, angle, outer = spot_geometry(
                positions,
                [light],
                cfg.eps,
            )

            mask = (
                (d > cfg.eps)
                & (d < 2.0 * r)
                & (angle < outer)
            )

            # Eq.11: softmin(radial_distance, angular_distance)
            penalty = weighted_softmin(
                r - d,
                d * torch.sin(outer - angle),
                beta=cfg.beta_m,
            )

        else:
            raise ValueError(
                f"Unsupported light type: {light.kind}"
            )

        # Remove the singleton light dimension: [1,N] -> [N].
        masks.append(mask[0])
        penalties.append(penalty[0])

    if len(lights) == 0:
        shape = (0, positions.shape[0])

        return (
            torch.zeros(
                shape,
                device=positions.device,
                dtype=torch.bool,
            ),
            positions.new_zeros(shape),
        )

    return (
        torch.stack(masks, dim=0),
        torch.stack(penalties, dim=0),
    )


def overlap_loss(
    positions: Tensor,
    lights: Sequence[LightParameters],
    E_per_light: Tensor,
    cfg: LossConfig,
) -> Tensor:
    """Compute joint coverage and contribution weights across Point/Spot lights."""
    # Eq. 7: sum overlap-scaled, contribution-weighted geometric penalties.
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")
    if E_per_light.shape != (len(lights), positions.shape[0], 3):
        raise ValueError("E_per_light must have shape [M,N,3]")

    candidates, geo_penalty = build_overlap_terms(positions, lights, cfg)
    if len(lights) == 0:
        return E_per_light.sum() * 0.0

    count = candidates.sum(dim=0)

    # Eq.8 alpha(n) = (max(0, n-nmax))^2
    alpha = (
        count.to(positions.dtype) - cfg.n_max
    ).clamp_min(0).square()  # [N]

    # Smaller contribution receives more penalties
    light_contribution = rgb_luminance(E_per_light) #[M, N]
    logits = (-cfg.beta_w * light_contribution).masked_fill(
        ~candidates,
        -torch.inf # lights that are not candidates is e^-inf
    )
    has_candidates = candidates.any(dim=0, keepdim=True)

    # to avoid nan, we fill 0 to -inf first
    logits = torch.where(has_candidates, logits, torch.zeros_like(logits))

    # we use candidates to clear the position without any candidates to 0
    weights = torch.softmax(logits, dim=0) * candidates # [M, N]
    weighted_penalty = (weights * geo_penalty).sum(dim=0) # [N]

    return (alpha * weighted_penalty).sum()


def attraction_loss(
    positions: Tensor,
    normals: Tensor,
    lights: Sequence[LightParameters],
    E_pred: Tensor,
    E_target: Tensor,
    cfg: LossConfig,
) -> Tensor:
    """Reuse current type-specific models with the same total prediction."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")
    if any(
        light.kind not in ("Point", "Spot")
        for light in lights
    ):
        raise ValueError(
            "Only Point and Spot lights are supported"
        )

    if any(
        value.shape != positions.shape
        for value in (normals, E_pred, E_target)
    ):
        raise ValueError(
            "normals, E_pred and E_target must match positions [N,3]"
        )

    points = [
        light for light in lights
        if light.kind == "Point"
    ]

    spots = [
        light for light in lights
        if light.kind == "Spot"
    ]

    point_term = point_attraction_loss(
        positions,
        normals,
        points,
        E_pred,
        E_target,
        under_eps=cfg.under_eps,
        dark_eps=cfg.dark_eps,
        volume=cfg.volume,
        beta_p=cfg.beta_p,
    )

    spot_term = spot_attraction_loss(
        positions,
        normals,
        spots,
        E_pred,
        E_target,
        under_eps=cfg.under_eps,
        dark_eps=cfg.dark_eps,
        volume=cfg.volume,
        beta_m=cfg.beta_m,
        eps=cfg.eps,
        sign_correction=cfg.spot_sign_correction,
    )

    return point_term + spot_term


def compute_losses(
    positions: Tensor,
    normals: Tensor,
    lights: Sequence[LightParameters],
    E_target: Tensor,
    area_weights: Tensor,
    cfg: LossConfig,
) -> dict[str, Tensor]:
    # Starting point of the loss calculation
    
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")
    if normals.shape != positions.shape or E_target.shape != positions.shape:
        raise ValueError("normals and E_target must match positions [N,3]")
    if area_weights.shape != (positions.shape[0],):
        raise ValueError("area_weights must have shape [N]")

    E_per_light = evaluate_per_light(positions, normals, lights)
    E_pred = E_per_light.sum(dim=0)

    reconstruction = reconstruction_loss(E_pred, E_target, area_weights)
    overlap = overlap_loss(positions, lights, E_per_light, cfg)
    attraction = attraction_loss(positions, normals, lights, E_pred, E_target, cfg)
    total = (
        reconstruction
        + cfg.overlap_weight * overlap
        + cfg.attraction_weight * attraction
    )

    return {
        "total": total,
        "reconstruction": reconstruction,
        "overlap": overlap,
        "attraction": attraction,
    }

