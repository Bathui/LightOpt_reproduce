import math
from dataclasses import fields, replace
from typing import Sequence

import torch
from torch import Tensor
from sklearn.cluster import DBSCAN

from algo.light import (
    LightParameters,
    evaluate_irradiance,
    evaluate_per_light,
)
from algo.loss import rgb_luminance
from config.light_set_config import LightSetConfig
from config.loss_config import LossConfig



def snapshot_lights(
    lights: Sequence[LightParameters],
) -> list[LightParameters]:
    """Detach discrete-stage data from the previous optimization graph."""
    new_lights = []

    for light in lights:
        new_values = {}

        for field in fields(light):
            value = getattr(light, field.name)

            if isinstance(value, Tensor):
                new_values[field.name] = value.detach().clone()

        new_light = replace(light, **new_values)
        new_lights.append(new_light)

    return new_lights


@torch.no_grad()
def remove_lights(positions, normals, lights, cfg):
    """Paper Eq. (18), scalarized with the project's RGB luminance."""
    per_light = evaluate_per_light(
        positions,
        normals,
        lights,
    )  # [M,N,3]

    importance = rgb_luminance(per_light).sum(dim=1)  # [M]

    kept = (
        (importance >= cfg.importance_threshold)
        .nonzero()
        .flatten()
        .tolist()
    )
    removed = (
        (importance < cfg.importance_threshold)
        .nonzero()
        .flatten()
        .tolist()
    )

    return snapshot_lights([lights[i] for i in kept]), {
        "importance": importance.cpu().tolist(),
        "kept_indices": kept,
        "removed_indices": removed,
    }


# The function helps us find the candidate as the new added light
@torch.no_grad()
def _fit_cluster(
    positions,
    normals,
    target,
    predicted,
    area,
    cfg,
):
    center = positions.mean(dim=0)  # [3]
    mean_normal = normals.mean(dim=0)
    normal_length = mean_normal.norm()

    if normal_length <= 1e-8:
        # Opposing normals need a separate clustering policy.
        return None

    mean_normal = mean_normal / normal_length
    direction = -mean_normal

    mean_rgb = target.mean(dim=0)
    color = mean_rgb / mean_rgb.max().clamp_min(1e-8)

    # Keep negative channels to penalize overshoot.
    residual = target - predicted  # [K,3]

    heights = torch.logspace(
        math.log10(cfg.height_min),
        math.log10(cfg.height_max),
        cfg.height_steps,
        device=positions.device,
        dtype=positions.dtype,
    )

    best = None
    best_error = float("inf")

    for height in heights:
        location = center + height * mean_normal

        offsets = positions - location
        distances = offsets.norm(dim=-1)
        rays = offsets / distances.clamp_min(1e-8)[:, None]

        outer = torch.acos(
            (rays * direction).sum(-1).clamp(-1, 1)
        ).max()
        outer = outer + math.radians(cfg.angle_margin_degrees) # add extra 2 degrees to make sure at least
        # if the outer angle is too big as a point light, it will turn into a spot light
        point = (
            2 * outer > math.radians(cfg.point_angle_degrees)
        )

        light = LightParameters(
            kind="Point" if point else "Spot",
            position=location,
            color_linear=color,
            intensity=positions.new_tensor(1.0),
            range=(
                distances.max().clamp_min(1e-4)
                * cfg.range_margin
            ),
            attenuation_smoothness=positions.new_tensor(
                cfg.attenuation_smoothness
            ),
            direction=None if point else direction,
            inner_half_angle=(
                None if point else outer * cfg.inner_angle_ratio
            ),
            outer_half_angle=None if point else outer,
        )

        unit = evaluate_irradiance(
            positions,
            normals,
            [light],
        )  # [K,3]

        denominator = ( area[:, None] * unit.square()).sum()

        if denominator <= 0:
            continue

        # This is the best intensity we can get by having the derivative= 0
        intensity = (
            (area[:, None] * unit * residual).sum()
            / denominator
        ).clamp_min(0)

        error = (
            area[:, None]
            * (intensity * unit - residual).square()
        ).sum().item()

        if intensity > 0 and error < best_error:
            best = replace(light, intensity=intensity)
            best_error = error

    return best


@torch.no_grad()
def add_lights(
    positions: Tensor,
    normals: Tensor,
    lights: Sequence[LightParameters],
    E_target: Tensor,
    area_weights: Tensor,
    cfg: LightSetConfig,
    loss_cfg: LossConfig,
):
    """Cluster underlit samples and initialize at most one light per cluster."""
    current = snapshot_lights(lights)
    predicted = evaluate_irradiance(
        positions,
        normals,
        current,
    )
    target_y = rgb_luminance(E_target)

    def underlit():
        return (
            (target_y > loss_cfg.dark_eps)
            & (
                rgb_luminance(predicted)
                <= loss_cfg.under_eps * target_y
            )
        )

    indices = underlit().nonzero().flatten()

    report = {
        "underlit_count": indices.numel(),
        "noise_count": 0,
        "added_indices": [],
        "skipped": [],
        "cluster_count": 0,
        "accepted": [],
    }

    if indices.numel() == 0:
        return current, report

    labels = DBSCAN(
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        metric="euclidean",
    ).fit_predict(
        positions[indices].cpu().numpy()
    )

    labels = torch.as_tensor(
        labels,
        device=positions.device,
    )
    report["noise_count"] = (labels < 0).sum().item()

    clusters = []
    for label in labels.unique().tolist():
        if label >= 0:
            cluster = indices[labels == label]
            clusters.append(cluster)


    clusters.sort(
        key=lambda idx: target_y[idx].sum().item(),
        reverse=True,
    )
    report["cluster_count"] = len(clusters)

    for cluster_number, cluster in enumerate(clusters):
        if len(report["added_indices"]) >= cfg.max_additions:
            report["unprocessed_clusters"] = len(clusters) - cluster_number
            break

        cluster = cluster[underlit()[cluster]]

        if cluster.numel() == 0:
            report["skipped"].append({
                "cluster": cluster_number, "reason": "already_recovered",
            })
            continue

        importance = target_y[cluster].sum().item()
        if importance <= cfg.importance_threshold:
            report["skipped"].append({
                "cluster": cluster_number, "reason": "low_importance",
                "importance": importance,
            })
            continue

        light = _fit_cluster(
            positions[cluster],
            normals[cluster],
            E_target[cluster],
            predicted[cluster],
            area_weights[cluster],
            cfg,
        )

        reason = None

        if light is None:
            reason = "no_valid_initialization"
        else:
            contribution = evaluate_irradiance(
                positions,
                normals,
                [light],
            )

            before = (
                area_weights[:, None] * (predicted - E_target).square()
            ).sum()
            
            after = (
                area_weights[:, None] * (
                    predicted + contribution - E_target
                ).square()
            ).sum()

            if not torch.isfinite(after):
                reason = "nonfinite_reconstruction"
            elif (
                cfg.require_reconstruction_improvement
                and after >= before
            ):
                reason = "no_global_reconstruction_improvement"

        if reason is not None:
            report["skipped"].append({
                "cluster": cluster_number,
                "reason": reason,
            })
            continue

        report["added_indices"].append(len(current))
        report["accepted"].append({
            "cluster": cluster_number, "light_index": len(current),
            "sample_count": cluster.numel(), "importance": importance,
            "kind": light.kind, "intensity": light.intensity.item(),
            "range": light.range.item(),
            "reconstruction_before": before.item(),
            "reconstruction_after": after.item(),
        })
        current.append(light)
        predicted = predicted + contribution

    return snapshot_lights(current), report


