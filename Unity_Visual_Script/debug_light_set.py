"""Debug variable-count optimization and save complete independent light states."""
import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from algo.light import LightParameters, evaluate_irradiance, evaluate_per_light, load_lights
from algo.light_set import snapshot_lights
from algo.loss import compute_losses, rgb_luminance
from algo.optimize import optimize_remove_add
from config.light_set_config import LightSetConfig
from config.loss_config import LossConfig
from config.optimization_config import OptimizationConfig
from scripts.export_unity_preview import sha256

FORMAT = "lightopt.light_set_debug.v1"


def encode_lights(lights):
    """Preserve every field without pairing the final list with source lights."""
    output = []
    for light in lights:
        record = {}
        for field in fields(light):
            value = getattr(light, field.name)
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError("Non-finite light parameter: " + field.name)
                value = value.detach().cpu().tolist()
            record[field.name] = value
        if light.kind not in ("Point", "Spot") or light.intensity < 0 or light.range <= 0:
            raise ValueError("Invalid optimized light")
        output.append(record)
    return output


def decode_lights(records, template):
    """Read the complete tensor representation used by this debug format."""
    lights = [LightParameters(**{
        key: value if key == "kind" or value is None else template.new_tensor(value)
        for key, value in record.items()
    }) for record in records]
    for light in lights:
        if light.position.shape != (3,) or light.color_linear.shape != (3,):
            raise ValueError("Require light position/color [3]")
        if light.intensity.ndim or light.range.ndim or light.attenuation_smoothness.ndim:
            raise ValueError("Require scalar intensity/range/attenuation")
        if (light.color_linear < 0).any() or light.attenuation_smoothness <= 0:
            raise ValueError("Invalid color or attenuation")
        if light.kind == "Spot" and (
            light.direction is None or light.direction.shape != (3,)
            or light.direction.norm() <= 1e-8
            or light.inner_half_angle is None or light.outer_half_angle is None
            or not 0 <= light.inner_half_angle < light.outer_half_angle < np.pi / 2
        ):
            raise ValueError("Invalid Spot geometry")
    encode_lights(lights)
    return lights


@torch.no_grad()
def summarize(p, n, w, target, lights, loss_cfg):
    losses = compute_losses(p, n, lights, target, w, loss_cfg)
    if not all(torch.isfinite(value) for value in losses.values()):
        raise ValueError("Non-finite diagnostic loss")
    per = evaluate_per_light(p, n, lights)
    target_y = rgb_luminance(target)
    under = ((target_y > loss_cfg.dark_eps)
             & (rgb_luminance(per.sum(0)) <= loss_cfg.under_eps * target_y))
    return dict(light_count=len(lights),
                losses={key: value.item() for key, value in losses.items()},
                underlit_samples=under.sum().item(),
                actual_excess_samples=((per.sum(-1) > 0).sum(0) > loss_cfg.n_max).sum().item())


def load_start(path, folder, records, original, template):
    """Resume a completed debug result or the fixed-count Unity experiment."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    if Path(saved["sampleFolder"]).resolve() != folder:
        raise ValueError("Start result uses another sample folder")
    for name in ("samples.npz", "lights.json"):
        source = folder / name
        if saved["inputSha256"].get(str(source)) != sha256(source):
            raise ValueError("Start result input hash mismatch: " + name)
    if saved.get("format") in (FORMAT, "lightopt.scene_light_set.v1"):
        return decode_lights(saved["states"]["after"]["lights"], template)
    if saved.get("schemaVersion") != 2:
        raise ValueError("Unsupported start result")
    entries = saved["experiments"][0]["lights"]
    if len(entries) != len(original):
        raise ValueError("Fixed-count start result has a different light count")
    current = []
    for source, entry, light in zip(records, entries, original):
        if source["id"] != entry["id"] or source["type"] != entry["type"]:
            raise ValueError("Start result light identity mismatch")
        current.append(replace(light,
            position=template.new_tensor([entry["afterPosition"][key] for key in "xyz"]),
            intensity=template.new_tensor(entry["afterIntensity"]),
            range=template.new_tensor(entry["afterRange"])))
    return decode_lights(encode_lights(current), template)


def plot_report(bundle, path):
    """Plot real continuous steps and discrete transitions on the same timeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    offset = 0
    for stage in bundle["history"]:
        rows = stage.get("optimization", [])
        if rows:
            axes[0].plot([offset + row["step"] for row in rows],
                         [row["reconstruction"] for row in rows], color="#2777a5")
            offset += rows[-1]["step"]
        if stage["stage"] == "round":
            values = [stage[key]["reconstruction"] for key in
                      ("losses_before_remove", "losses_after_remove", "losses_after_add")]
            axes[0].plot([offset] * 3, values, "o-", color="#c65e31", markersize=4)
    stages = [bundle["states"]["before"]["summary"]["light_count"]]
    names = ["Before"]
    for stage in bundle["history"]:
        stages.append(stage["light_count"])
        names.append("Init prune" if stage["stage"] == "initial_removal" else
                     f"Round {stage['round']+1}" if stage["stage"] == "round" else "Final refine")
    axes[0].set(xlabel="Adam updates (all blocks)", ylabel="Reconstruction error",
                title="Continuous optimization and remove/add transitions")
    axes[1].bar(names, stages, color="#2777a5")
    axes[1].set(ylabel="Light count", title="Actual light-set size")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(sample_folder, *, cfg=None, set_cfg=None, loss_cfg=None, rounds=3,
        device="auto", start_from=None, output=None, make_plot=True):
    """Run the real algorithm against a fixed target; leave original experiments intact."""
    folder = Path(sample_folder).resolve()
    if not folder.is_relative_to(ROOT / "data"):
        raise ValueError("Select a sample folder inside data")
    start_from = Path(start_from).resolve() if start_from else None
    cfg = cfg or OptimizationConfig(steps=100, initial_intensity_scale=1. if start_from else .5)
    set_cfg = set_cfg or LightSetConfig(importance_threshold=.1)
    loss_cfg = loss_cfg or LossConfig(overlap_weight=0., attraction_weight=0.)
    output = Path(output or ROOT / "data/light_set_debug" / (folder.name + ".json")).resolve()
    inputs = [folder/"samples.npz", folder/"lights.json"]
    inputs += [ROOT/name for name in (
        "algo/light.py", "algo/loss.py", "algo/light_set.py", "algo/optimize.py",
        "algo/trainable_lights.py", "config/light_set_config.py", "config/loss_config.py",
        "config/optimization_config.py", "scripts/export_unity_preview.py")]
    inputs += [Path(__file__).resolve()]
    if start_from:
        inputs.append(start_from)
    if (not output.is_relative_to(ROOT/"data") or output.suffix.lower() != ".json"
            or output in inputs):
        raise ValueError("Output must be a separate JSON inside data")
    if output.exists() and json.loads(output.read_text(encoding="utf-8")).get("format") != FORMAT:
        raise FileExistsError("Output is not a light-set debug result; choose another file")
    hashes = {str(path): sha256(path) for path in inputs}
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device is unavailable")
    with np.load(folder/"samples.npz", allow_pickle=False) as data:
        p, n, w = [np.array(data[key], copy=True) for key in ("positions", "normals", "area_weights")]
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or n.shape != p.shape or w.shape != (len(p),):
        raise ValueError("Require nonempty positions/normals [N,3], weights [N]")
    if (not all(np.isfinite(value).all() for value in (p, n, w))
            or (w < 0).any() or w.sum() <= 0
            or not np.allclose(np.linalg.norm(n, axis=1), 1., atol=1e-5, rtol=1e-5)):
        raise ValueError("Require finite samples, unit normals, and valid area weights")
    p, n, w = [torch.as_tensor(value, device=device, dtype=torch.float32) for value in (p, n, w)]
    if not all(torch.isfinite(value).all() for value in (p, n, w)):
        raise ValueError("Input overflows float32")
    records = json.loads((folder/"lights.json").read_text(encoding="utf-8-sig"))
    original = load_lights(str(folder/"lights.json"), device=device,
                           attenuation_smoothness=set_cfg.attenuation_smoothness)
    with torch.no_grad():
        target = evaluate_irradiance(p, n, original)
    current = load_start(start_from, folder, records, original, p) if start_from else snapshot_lights(original)
    before = [replace(light, intensity=light.intensity * cfg.initial_intensity_scale) for light in current]
    states = {"reference": {"lights": encode_lights(original),
                             "summary": summarize(p, n, w, target, original, loss_cfg)},
              "before": {"lights": encode_lights(before),
                          "summary": summarize(p, n, w, target, before, loss_cfg)}}
    print(f"Samples={len(p)} lights={len(current)} device={device}; rounds={rounds} steps/block={cfg.steps}")
    print(f"Weights: R=1 O={loss_cfg.overlap_weight} A={loss_cfg.attraction_weight}")
    started = time.perf_counter()
    optimized, history = optimize_remove_add(p, n, current, target, w, cfg, set_cfg, loss_cfg, rounds)
    states["after"] = {"lights": encode_lights(optimized),
                       "summary": summarize(p, n, w, target, optimized, loss_cfg)}
    elapsed = time.perf_counter() - started
    for path in inputs:
        if sha256(path) != hashes[str(path)]:
            raise ValueError("Input/source changed while running: " + str(path))
    bundle = dict(format=FORMAT, schemaVersion=1, sampleFolder=str(folder), sampleCount=len(p),
                  targetMode="original_lights_recomputed", device=device,
                  startFrom=str(start_from) if start_from else None,
                  optimizationConfig=asdict(cfg), lightSetConfig=asdict(set_cfg), lossConfig=asdict(loss_cfg),
                  rounds=rounds, elapsedSeconds=elapsed, inputSha256=hashes,
                  states=states, history=history, inputsUnchanged=True,
                  note="Independent light lists; Python irradiance diagnostics, not Unity preview data.")
    text = json.dumps(bundle, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if make_plot:
        plot_report(bundle, output.with_suffix(".png"))
    pending = output.with_suffix(".json.tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(output)
    print("Before:", states["before"]["summary"])
    print("After:", states["after"]["summary"])
    print("Saved:", output)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_folder", nargs="?")
    parser.add_argument("--steps", type=int, default=100, help="Adam updates per continuous block")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--initial-scale", type=float)
    parser.add_argument("--intensity-learning-rate", type=float, default=.05)
    parser.add_argument("--position-learning-rate", type=float, default=.001)
    parser.add_argument("--range-learning-rate", type=float, default=.001)
    parser.add_argument("--overlap-weight", type=float, default=0.)
    parser.add_argument("--attraction-weight", type=float, default=0.)
    parser.add_argument("--importance-threshold", type=float, default=.1)
    parser.add_argument("--dbscan-eps", type=float, default=.5)
    parser.add_argument("--dbscan-min-samples", type=int, default=8)
    parser.add_argument("--height-min", type=float, default=.1)
    parser.add_argument("--height-max", type=float, default=5.)
    parser.add_argument("--height-steps", type=int, default=24)
    parser.add_argument("--max-additions", type=int, default=8)
    parser.add_argument("--attenuation-smoothness", type=float, default=1.)
    parser.add_argument("--start-from", help="Resume the AFTER state of a saved result")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    folder = args.sample_folder or input("采样目录（例如 data/sponza_samples_001）: ").strip().strip('"')
    scale = args.initial_scale if args.initial_scale is not None else (1. if args.start_from else .5)
    cfg = OptimizationConfig(steps=args.steps, initial_intensity_scale=scale,
        intensity_learning_rate=args.intensity_learning_rate, position_learning_rate=args.position_learning_rate,
        range_learning_rate=args.range_learning_rate, log_every=max(1, args.steps//5))
    set_cfg = LightSetConfig(importance_threshold=args.importance_threshold,
        dbscan_eps=args.dbscan_eps, dbscan_min_samples=args.dbscan_min_samples,
        height_min=args.height_min, height_max=args.height_max, height_steps=args.height_steps,
        max_additions=args.max_additions, attenuation_smoothness=args.attenuation_smoothness)
    run(folder, cfg=cfg, set_cfg=set_cfg,
        loss_cfg=LossConfig(overlap_weight=args.overlap_weight, attraction_weight=args.attraction_weight),
        rounds=args.rounds, device=args.device, start_from=args.start_from, output=args.output,
        make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
