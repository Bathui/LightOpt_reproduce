"""Optimize a sampled scene with optional light-set adjustment and versioned results."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from algo.light import evaluate_irradiance, load_lights
from algo.optimize import optimize_intensities, optimize_remove_add
from config.light_set_config import LightSetConfig
from config.optimization_config import OptimizationConfig
from config.loss_config import LossConfig
from scripts.export_unity_preview import make_experiment, sha256

PROJECT = ROOT / "editor_scene/LightOpt_Scene"
LIGHT_SET_FORMAT = "lightopt.scene_light_set.v1"


def light_count_report(original_count, final_count, stages=None):
    """Count discrete operations, including initial pruning in legacy results."""
    enabled = stages is not None
    removed = added = 0
    for stage in stages or []:
        removed += len(stage.get("removed_indices", []))
        removed += len(stage.get("removal", {}).get("removed_indices", []))
        added += len(stage.get("addition", {}).get("added_indices", []))
    if original_count - removed + added != final_count:
        raise RuntimeError("Light-count ledger does not match the final light set")
    return dict(adjustmentEnabled=enabled, original=original_count, final=final_count,
                removed=removed, added=added, netReduction=original_count-final_count)


def print_light_count_report(report):
    """Print a final summary for both fixed-count and variable-count runs."""
    print("\n===== Light Count Report =====")
    print("Light-set adjustment:", "ENABLED" if report["adjustmentEnabled"]
          else "DISABLED (fixed-count mode; no removal or addition)")
    print(f"Original light count: {report['original']}")
    print(f"Final optimized light count: {report['final']}")
    print(f"Total removed: {report['removed']} / Total added: {report['added']}")
    print(f"Net light reduction: {report['netReduction']}")


def run_light_set(p, n, w, lights, target, cfg, loss_cfg, set_cfg, rounds,
                  folder, scene, guid, device, output, hashes, make_plot):
    """Save variable-count states independently; do not pair them with source lights."""
    from Unity_Visual_Script.debug_light_set import encode_lights, plot_report, summarize

    before = [replace(light, intensity=light.intensity * cfg.initial_intensity_scale)
              for light in lights]
    states = {
        "reference": {"lights": encode_lights(lights),
                      "summary": summarize(p, n, w, target, lights, loss_cfg)},
        "before": {"lights": encode_lights(before),
                   "summary": summarize(p, n, w, target, before, loss_cfg)},
    }
    optimized, stages = optimize_remove_add(
        p, n, lights, target, w, cfg, set_cfg, loss_cfg=loss_cfg, rounds=rounds,
    )
    states["after"] = {"lights": encode_lights(optimized),
                       "summary": summarize(p, n, w, target, optimized, loss_cfg)}
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError("Input or source changed during optimization: " + path)
    bundle = dict(
        schemaVersion=3, format=LIGHT_SET_FORMAT, sampleFolder=str(folder), sampleCount=len(p),
        sourceScene=str(scene), sourceSceneGuid=guid, sourceSceneSha256=hashes[str(scene)],
        device=device, attenuationSmoothness=set_cfg.attenuation_smoothness,
        targetMode="original_lights_recomputed", optimizedParameters=[
            "intensity",
            "position",
            "range",
            "color_linear",
            "direction",
        ],
        optimizationConfig=asdict(cfg), lightSetConfig=asdict(set_cfg), lossConfig=asdict(loss_cfg),
        lossWeights=dict(reconstruction=1., overlap=loss_cfg.overlap_weight,
                        attraction=loss_cfg.attraction_weight),
        rounds=rounds, states=states, history=stages, inputSha256=hashes,
        inputsUnchanged=True,
        lightCountReport=light_count_report(len(lights), len(optimized), stages),
        note="Variable-count results for the updated Unity schema-3 preview.",
    )
    text = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if make_plot:
        plot_report(bundle, output.with_suffix(".png"))
    pending = output.with_suffix(".json.tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(output)
    print(f"Saved: {output}\nLights: {len(before)} -> {len(optimized)}")
    print("Reconstruction:", states["before"]["summary"]["losses"]["reconstruction"],
          "->", states["after"]["summary"]["losses"]["reconstruction"])
    print_light_count_report(bundle["lightCountReport"])
    return bundle


def find_scene(records, scene=None):
    """Resolve the original scene by exported light identity, never by filename."""
    guids = set()
    for record in records:
        match = re.fullmatch(r"GlobalObjectId_V1-2-([0-9a-f]{32})-\d+-\d+", record["id"])
        if not match:
            raise ValueError("Lights need Unity GlobalObjectId identities")
        guids.add(match.group(1))
    if len(guids) != 1:
        raise ValueError("Lights must belong to exactly one Unity scene")
    guid = guids.pop()
    candidates = [Path(scene).resolve()] if scene else [Path(str(meta)[:-5]) for meta in (PROJECT / "Assets").rglob("*.unity.meta")]
    matches = []
    for candidate in candidates:
        meta = Path(str(candidate) + ".meta")
        if candidate.is_file() and meta.is_file() and re.search(r"^guid: " + guid + r"\s*$", meta.read_text(), re.MULTILINE):
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("Cannot uniquely resolve original scene; specify --scene with the matching .unity file")
    return matches[0], guid


def run(sample_folder, *, steps=300, intensity_learning_rate=.05, initial_scale=.5,
        position_learning_rate=.001, range_learning_rate=.001,
        overlap_weight=.1, attraction_weight=.2,
        device="auto", attenuation_smoothness=1., scene=None, output=None,
        adjust_lights=False, rounds=3, light_set_cfg=None, make_plot=True):
    """Optimize intensity, position, and range against a fixed reference target."""
    folder = Path(sample_folder).resolve()
    if not folder.is_relative_to(ROOT / "data"):
        raise ValueError("Select a sample folder inside this project's data directory")
    default_name = "experiment_light_set.json" if adjust_lights else "experiment.json"
    output = Path(output or ROOT / "data" / default_name).resolve()
    if not output.is_relative_to(ROOT / "data") or output.suffix.lower() != ".json":
        raise ValueError("Output must be a JSON file inside data")
    cfg = OptimizationConfig(steps=steps, intensity_learning_rate=intensity_learning_rate,
                             position_learning_rate=position_learning_rate,
                             range_learning_rate=range_learning_rate,
                             initial_intensity_scale=initial_scale, log_every=max(1, steps // 10))
    loss_cfg = LossConfig(overlap_weight=overlap_weight, attraction_weight=attraction_weight)
    if adjust_lights:
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        light_set_cfg = light_set_cfg or LightSetConfig(
            importance_threshold=.1, attenuation_smoothness=attenuation_smoothness,
        )
        if light_set_cfg.attenuation_smoothness != attenuation_smoothness:
            raise ValueError("Existing and added lights must use the same attenuation smoothness")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select cpu")
    records = json.loads((folder / "lights.json").read_text(encoding="utf-8-sig"))
    scene, guid = find_scene(records, scene)
    inputs = [folder / "samples.npz", folder / "lights.json", scene,
              ROOT / "algo/light.py", ROOT / "algo/loss.py", ROOT / "algo/optimize.py",
              ROOT / "algo/trainable_lights.py", ROOT / "config/optimization_config.py",
              ROOT / "config/loss_config.py", ROOT / "scripts/export_unity_preview.py",
              Path(__file__).resolve()]
    if adjust_lights:
        inputs += [ROOT / "algo/light_set.py", ROOT / "config/light_set_config.py",
                   ROOT / "Unity_Visual_Script/debug_light_set.py"]
    if output in inputs:
        raise ValueError("Output must not overwrite an input file")
    hashes = {str(path): sha256(path) for path in inputs}
    # Validate identity and preview compatibility before starting an expensive run.
    reference = [record["intensity"] for record in records]
    result = dict(reference_intensities=reference, final_intensities=reference,
                  initial_intensity_scale=initial_scale, steps=steps,
                  learning_rate=intensity_learning_rate, initial_loss=0., final_loss=0.)
    experiment = make_experiment(result, records, folder.name, "scene", guid, hashes[str(scene)])
    if any(record.get("mode") != "Realtime" or not record.get("enabled") or not record.get("active") for record in records):
        raise ValueError("Only active, enabled Realtime lights are supported")
    with np.load(folder / "samples.npz", allow_pickle=False) as data:
        p, n, w = (np.array(data[key], copy=True) for key in ("positions", "normals", "area_weights"))
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or n.shape != p.shape or w.shape != (len(p),):
        raise ValueError("Require nonempty positions/normals [N,3] and area_weights [N]")
    if not all(np.isfinite(value).all() for value in (p, n, w)) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Samples must be finite and area weights nonnegative with positive sum")
    if not np.allclose(np.linalg.norm(n, axis=1), 1., atol=1e-5, rtol=1e-5):
        raise ValueError("Sample normals must be unit length")
    p, n, w = (torch.as_tensor(value, dtype=torch.float32, device=device) for value in (p, n, w))
    if not all(torch.isfinite(value).all() for value in (p, n, w)):
        raise ValueError("Samples overflow float32")
    lights = load_lights(str(folder / "lights.json"), attenuation_smoothness=attenuation_smoothness, device=device)
    # The fixed reference uses original intensities; only initialization is scaled.
    with torch.no_grad():
        target = evaluate_irradiance(p, n, lights)
    if not torch.isfinite(target).all():
        raise ValueError("Reference target is non-finite")
    print(f"Scene: {scene.name}; samples: {len(p)}; lights: {len(lights)}; device: {device}")
    print(f"Learning rates: intensity={intensity_learning_rate}, position={position_learning_rate}, range={range_learning_rate}")
    print(f"Loss weights: reconstruction=1, overlap={overlap_weight}, attraction={attraction_weight}")
    if adjust_lights:
        print(f"Light-set adjustment enabled: rounds={rounds}; steps per block={steps}")
        return run_light_set(p, n, w, lights, target, cfg, loss_cfg, light_set_cfg, rounds,
                             folder, scene, guid, device, output, hashes, make_plot)
    model, history = optimize_intensities(p, n, lights, target, w, cfg, loss_cfg=loss_cfg)
    optimized = model()
    if any(not torch.isfinite(value).all() for light in optimized for value in (light.position, light.range, light.intensity)):
        raise ValueError("Optimized light parameters are non-finite")
    if any(light.range <= 0 or light.intensity < 0 for light in optimized):
        raise ValueError("Optimized light parameters violate constraints")
    result.update(final_intensities=[light.intensity.detach().item() for light in optimized],
                  initial_loss=history[0]["reconstruction"], final_loss=history[-1]["reconstruction"])
    experiment = make_experiment(result, records, folder.name, "scene", guid, hashes[str(scene)])
    for entry, original, final in zip(experiment["lights"], lights, optimized):
        def vector(tensor):
            return dict(zip(("x", "y", "z"), tensor.detach().cpu().tolist()))
        entry.update(
            referencePosition=vector(original.position),
            beforePosition=vector(original.position),
            afterPosition=vector(final.position),
            referenceRange=original.range.detach().item(),
            beforeRange=original.range.detach().item(),
            afterRange=final.range.detach().item(),
        )
    experiment.update(initialTotalLoss=history[0]["total"], finalTotalLoss=history[-1]["total"])
    for path in inputs:
        if sha256(path) != hashes[str(path)]:
            raise ValueError("Input or source changed during optimization: " + str(path))
    # Version 2 prevents the intensity-only Unity reader from silently dropping geometry updates.
    bundle = dict(schemaVersion=2, experiments=[experiment], sampleFolder=str(folder),
                  device=device, attenuationSmoothness=attenuation_smoothness,
                  targetMode="original_lights_recomputed", optimizedParameters=[
                    "intensity",
                    "position",
                    "range",
                    "color_linear",
                    "direction",
                ],
                  optimizationConfig=asdict(cfg), lossConfig=asdict(loss_cfg),
                  lossWeights=dict(reconstruction=1., overlap=overlap_weight, attraction=attraction_weight),
                  inputSha256=hashes, history=history,
                  lightCountReport=light_count_report(len(lights), len(optimized)))
    text = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a completed run; preserve the previous experiment if Adam fails.
    pending = output.with_suffix(".json.tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(output)
    print(f"Saved: {output}\nReconstruction: {result['initial_loss']:.9g} -> {result['final_loss']:.9g}")
    print_light_count_report(bundle["lightCountReport"])
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_folder", nargs="?", help="Sample folder inside data")
    parser.add_argument("--scene", help="Original Unity scene; otherwise resolved by GUID")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--intensity-learning-rate", "--learning-rate", dest="intensity_learning_rate", type=float)
    parser.add_argument("--position-learning-rate", type=float)
    parser.add_argument("--range-learning-rate", type=float)
    parser.add_argument("--overlap-weight", type=float)
    parser.add_argument("--attraction-weight", type=float)
    parser.add_argument("--initial-scale", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--attenuation-smoothness", type=float, default=1.)
    parser.add_argument("--output", help="Default: experiment_light_set.json with adjustment, otherwise experiment.json")
    parser.add_argument("--adjust-lights", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable removal/addition between continuous optimization blocks")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--importance-threshold", type=float)
    parser.add_argument("--dbscan-eps", type=float)
    parser.add_argument("--dbscan-min-samples", type=int)
    parser.add_argument("--max-additions", type=int)
    parser.add_argument("--height-min", type=float, default=.1)
    parser.add_argument("--height-max", type=float, default=5.)
    parser.add_argument("--height-steps", type=int, default=24)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    interactive = args.sample_folder is None
    folder = args.sample_folder or input("采样目录（例如 data/garden_samples_004）: ").strip().strip('"')
    def value(current, prompt, default, convert):
        if current is not None:
            return current
        entered = input(f"{prompt} [{default}]: ").strip() if interactive else ""
        return convert(entered) if entered else default
    adjust = args.adjust_lights
    if adjust is None:
        adjust = bool(value(None, "启用增删灯（1=是，0=否）", 1 if interactive else 0, int))
    rounds = value(args.rounds, "增删灯外循环轮数", 3, int) if adjust else 3
    set_cfg = LightSetConfig(
        importance_threshold=value(args.importance_threshold, "删灯/增灯重要性阈值", .1, float),
        dbscan_eps=value(args.dbscan_eps, "DBSCAN 邻域距离", .5, float),
        dbscan_min_samples=value(args.dbscan_min_samples, "DBSCAN 最少样本数", 8, int),
        max_additions=value(args.max_additions, "每轮最多增灯数", 8, int),
        height_min=args.height_min, height_max=args.height_max, height_steps=args.height_steps,
        attenuation_smoothness=args.attenuation_smoothness,
    ) if adjust else None
    run(folder, steps=value(args.steps, "每阶段 Adam 步数" if adjust else "Adam 步数", 300, int),
        intensity_learning_rate=value(args.intensity_learning_rate, "Intensity 学习率", .05, float),
        position_learning_rate=value(args.position_learning_rate, "Position 学习率", .001, float),
        range_learning_rate=value(args.range_learning_rate, "Range 学习率", .001, float),
        overlap_weight=value(args.overlap_weight, "Overlap 权重", .1, float),
        attraction_weight=value(args.attraction_weight, "Attraction 权重", .2, float),
        initial_scale=value(args.initial_scale, "初始强度 / 原始强度", .5, float),
        device=args.device, attenuation_smoothness=args.attenuation_smoothness,
        scene=args.scene, output=args.output, adjust_lights=adjust, rounds=rounds,
        light_set_cfg=set_cfg, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
