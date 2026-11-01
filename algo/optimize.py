from typing import Iterable, Sequence, cast

import torch
from torch import Tensor

from algo.light import LightParameters
from algo.loss import compute_losses
from algo.trainable_lights import TrainableLight, TrainableLights
from config.loss_config import LossConfig
from config.optimization_config import OptimizationConfig

from dataclasses import replace

from algo.light_set import (
    add_lights,
    remove_lights,
    snapshot_lights,
)
from config.light_set_config import LightSetConfig

def optimize_intensities(
    positions: Tensor,
    normals: Tensor,
    lights: Sequence[LightParameters],
    E_target: Tensor,
    area_weights: Tensor,
    cfg: OptimizationConfig,
    loss_cfg: LossConfig | None = None,
) -> tuple[TrainableLights, list[dict[str, float | int]]]:
    """Run fixed-count optimization of intensity, position, and range.

    All inputs must share a floating dtype/device. The source lights and
    fixed data are not modified. Each history row describes parameters
    after exactly 'step' updates; the last row matches the returned model.
    """
    positions = positions.detach()
    normals = normals.detach()
    E_target = E_target.detach()
    area_weights = area_weights.detach()

    model = TrainableLights(
        lights,
        cfg.initial_intensity_scale,
    )

    trainable = list(
        cast(Iterable[TrainableLight], model.lights)
    )

    # three independent learning rates
    groups = [
        {
            "params": [light.intensity for light in trainable],
            "lr": cfg.intensity_learning_rate,
        },
        {
            "params": [light.position for light in trainable],
            "lr": cfg.position_learning_rate,
        },
        {
            "params": [light.range for light in trainable],
            "lr": cfg.range_learning_rate,
        },
        {
            "params": [light.color_linear for light in trainable],
            "lr": cfg.color_learning_rate,
        },
    ]

    directions = []

    for light in trainable:
        if light.direction is not None:
            directions.append(light.direction)

    if directions:
        groups.append({
            "params": directions,
            "lr": cfg.direction_learning_rate,
        })

    optimizer = torch.optim.Adam(groups)

    if loss_cfg is None:
        loss_cfg = LossConfig(
            overlap_weight=0.0,
            attraction_weight=0.0,
        )

    history: list[dict[str, float | int]] = []

    for step in range(cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        # Rebuild light views for every new forward graph.
        losses = compute_losses(
            positions,
            normals,
            model(),
            E_target,
            area_weights,
            loss_cfg,
        )

        total = losses["total"]

        if not torch.isfinite(total):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )
        # record the loss to history in every step 
        record: dict[str, float | int] = {"step": step}
        record.update({
            name: value.detach().item()
            for name, value in losses.items()
        })
        history.append(record)

        if step % cfg.log_every == 0 or step == cfg.steps:
            print(
                f"step={step:04d} "
                f"total={record['total']:.8g} "
                f"reconstruction={record['reconstruction']:.8g} "
                f"overlap={record['overlap']:.8g} "
                f"attraction={record['attraction']:.8g}"
            )

        if step == cfg.steps:
            break

        total.backward()

        for parameter in model.parameters():
            if (
                parameter.grad is None
                or not torch.isfinite(parameter.grad).all()
            ):
                raise RuntimeError(
                    f"Missing or non-finite gradient at step {step}"
                )

        optimizer.step() #update
        model.project_constraints()

    return model, history


def optimize_remove_add(
    positions: Tensor,
    normals: Tensor,
    lights: Sequence[LightParameters],
    E_target: Tensor,
    area_weights: Tensor,
    cfg: OptimizationConfig,
    set_cfg: LightSetConfig,
    loss_cfg: LossConfig | None = None,
    rounds: int = 3,
) -> tuple[list[LightParameters], list[dict]]:
    """Alternate fixed-count optimization with removal/addition, without merging."""
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")

    if loss_cfg is None:
        loss_cfg = LossConfig(
            overlap_weight=0.0,
            attraction_weight=0.0,
        )

    positions = positions.detach()
    normals = normals.detach()
    E_target = E_target.detach()
    area_weights = area_weights.detach()

    current = [
        replace(
            light,
            intensity=(
                light.intensity
                * cfg.initial_intensity_scale
            ),
        )
        for light in snapshot_lights(lights)
    ]

    # Never reapply the initialization scale to surviving or newly added lights.
    block_cfg = replace(
        cfg,
        initial_intensity_scale=1.0,
    )

    @torch.no_grad()
    def loss_snapshot(current):
        values = compute_losses(
            positions, normals, current, E_target, area_weights, loss_cfg,
        )
        if not all(torch.isfinite(value) for value in values.values()):
            raise RuntimeError("Non-finite loss at a discrete stage")
        return {key: value.item() for key, value in values.items()}

    history = []

    def refine(current):
        if not current:
            return [], []

        # Rebuilding the model also rebuilds Adam for the new parameter set.
        model, rows = optimize_intensities(
            positions,
            normals,
            current,
            E_target,
            area_weights,
            block_cfg,
            loss_cfg,
        )
        return snapshot_lights(model()), rows

    changed = False

    for iteration in range(rounds):
        print(f"Round {iteration + 1}/{rounds}: optimize {len(current)} lights")
        current, rows = refine(current)
        before_remove = rows[-1] if rows else loss_snapshot(current)

        current, removal = remove_lights(
            positions,
            normals,
            current,
            set_cfg,
        )
        after_remove = loss_snapshot(current)
        current, addition = add_lights(
            positions,
            normals,
            current,
            E_target,
            area_weights,
            set_cfg,
            loss_cfg,
        )
        after_add = loss_snapshot(current)

        changed = bool(
            removal["removed_indices"]
            or addition["added_indices"]
        )

        history.append({
            "stage": "round",
            "round": iteration,
            "optimization": rows,
            "removal": removal,
            "addition": addition,
            "light_count": len(current),
            "losses_before_remove": {
                key: before_remove[key]
                for key in ("total", "reconstruction", "overlap", "attraction")
            },
            "losses_after_remove": after_remove,
            "losses_after_add": after_add,
        })
        print(
            f"Round {iteration + 1}: removed={len(removal['removed_indices'])} "
            f"added={len(addition['added_indices'])} lights={len(current)} "
            f"R(opt/remove/add)={before_remove['reconstruction']:.8g}/"
            f"{after_remove['reconstruction']:.8g}/{after_add['reconstruction']:.8g}"
        )

    if changed and current:
        print(f"Final refinement: {len(current)} lights")
        current, rows = refine(current)
        history.append({
            "stage": "final_refine",
            "optimization": rows,
            "light_count": len(current),
            "losses": {key: rows[-1][key] for key in
                       ("total", "reconstruction", "overlap", "attraction")},
        })

    return current, history
