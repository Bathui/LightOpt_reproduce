"""Area-based mesh sampling followed by a conservative doubled-light-range mask.

No rendering, occlusion test, light fitting, or illumination-dependent filtering.
The Poisson option uses Open3D sample elimination with a target count, not an
exact fixed-radius sampler. Keep that distinction in experimental reports.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def xyz(value):
    return np.array([value[k] for k in ("x", "y", "z")], dtype=np.float64)


def load_scene(folder):
    folder = Path(folder)
    if (folder / "INCOMPLETE.txt").exists():
        raise ValueError("Incomplete Unity export; export again to an empty folder.")
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8-sig"))
    if scene.get("schemaVersion") not in (1, 2):
        raise ValueError("Unsupported scene schema")
    vertices = np.fromfile(folder / "vertices.f32", dtype="<f4").reshape(-1, 6).astype(np.float64)
    triangles = np.fromfile(folder / "triangles.i32", dtype="<i4").reshape(-1, 3)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError("Empty/nonfinite geometry")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("Invalid triangle indices")
    norm = np.linalg.norm(vertices[:, 3:], axis=1)
    if np.any(norm < 1e-8):
        raise ValueError("Zero vertex normal")
    vertices[:, 3:] /= norm[:, None]
    return scene, vertices, triangles


def sample_uniform(vertices, triangles, count, rng):
    """Triangle selection by world-space area; square-root barycentric sampling."""
    corners = vertices[triangles, :3]
    areas = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                        corners[:, 2] - corners[:, 0]), axis=1)
    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 0:
        raise ValueError("No nondegenerate surface area")
    face_ids = rng.choice(len(triangles), count, p=areas / total_area)
    root = np.sqrt(rng.random(count))
    v = rng.random(count)
    bary = np.column_stack((1 - root, root * (1 - v), root * v))
    points = np.einsum("ni,nij->nj", bary, vertices[triangles[face_ids], :3])
    normals = np.einsum("ni,nij->nj", bary, vertices[triangles[face_ids], 3:])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Interpolated normals cancel; check exported geometry")
    normals /= lengths[:, None]
    return points, normals, total_area


def sample_poisson(vertices, triangles, count, seed):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Poisson sampling needs open3d. Install it, or use --method uniform for a smoke test.") from exc
    o3d.utility.random.seed(seed)
    mesh = o3d.geometry.TriangleMesh()
    # convert numpy to open3d point cloud
    mesh.vertices = o3d.utility.Vector3dVector(vertices[:, :3])
    mesh.vertex_normals = o3d.utility.Vector3dVector(vertices[:, 3:])
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    area = mesh.get_surface_area()
    if not math.isfinite(area) or area <= 0:
        raise ValueError("No nondegenerate surface area")
    cloud = mesh.sample_points_poisson_disk(number_of_points=count, init_factor=5,
                                           use_triangle_normal=False)
    # convert back into numpy ds
    points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Invalid interpolated normals")
    return points, normals / lengths[:, None], float(area)


def range_masks(points, lights):
    """Conservative spheres for both point/spot lights; no shadow/normal/cone test.

    In particular, never filter samples using E_target > 0: dark samples matter.
    """
    retained, original_range = np.zeros(len(points), bool), np.zeros(len(points), bool)
    for light in lights:
        distance2 = np.sum((points - xyz(light["position"])) ** 2, axis=1)
        radius2 = float(light["range"]) ** 2
        retained |= distance2 <= 4 * radius2
        original_range |= distance2 <= radius2
    return retained, original_range


def write_ply(path, points, normals, in_original_range, categories=None):
    with Path(path).open("w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\nelement vertex " + str(len(points)) + "\n")
        for name in ("x", "y", "z", "nx", "ny", "nz"):
            file.write("property float " + name + "\n")
        for name in ("red", "green", "blue"):
            file.write("property uchar " + name + "\n")
        file.write("end_header\n")
        for i, (p, n, original) in enumerate(zip(points, normals, in_original_range)):
            color = (255, 160, 50) if original else (60, 140, 255)
            if categories is not None:
                color = [(180, 180, 180), (160, 90, 40), (70, 200, 80)][int(categories[i])]
            file.write(" ".join(f"{x:.8g}" for x in np.r_[p, n]) +
                       " " + " ".join(map(str, color)) + "\n")


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh empty output directory")
    scene, vertices, triangles = load_scene(args.input)
    print("Input:", Path(args.input).resolve(), flush=True)
    print("Export schema:", scene["schemaVersion"], flush=True)
    if scene["schemaVersion"] == 1:
        print("CHECK: Legacy v1 export has no UV/alpha masks. Excluded trees cannot be recovered by resampling; use the new Unity v2 export.", flush=True)
    inventory = {}
    for light in scene["lights"]:
        key = light["type"] + "/" + light["mode"]
        inventory[key] = inventory.get(key, 0) + 1
    print("Light inventory:", inventory)
    lights = [light for light in scene["lights"] if light["type"] in ("Point", "Spot")
              and light["enabled"] and light["active"] and float(light["intensity"]) > 0
              and float(light["range"]) > 0 and light["mode"] in args.light_modes]
    if not lights:
        raise ValueError("No eligible lights. Inspect scene.json and explicitly choose --light-modes Realtime Mixed if appropriate.")
    # Fail rather than silently ignore light linking (needed later in the irradiance evaluator).
    warnings = []
    if any(light.get("hasCookie") for light in lights):
        warnings.append("Some lights use cookies; the paper's basic model does not include cookies.")
    if any(light.get("shadows") != "None" for light in lights):
        warnings.append("Some lights cast shadows; v1 sampling does not evaluate visibility.")
    if any(light.get("useColorTemperature") for light in lights):
        warnings.append("Some lights use color temperature; effective color needs calibration before irradiance evaluation.")
    if any(light.get("cullingMask", -1) != -1 for light in lights):
        warnings.append("Some lights have restricted culling masks; implement light/receiver linking before optimization.")
    if any(light["mode"] != "Realtime" for light in lights):
        warnings.append("Mixed/Baked selected: baked contributions will need separate handling in Unity comparisons.")
    alpha_stats = None
    sample_data = None
    if scene.get("schemaVersion") == 2:
        from alpha_sampling import sample_textured
        sample_data, alpha_stats = sample_textured(args.input, scene, vertices, triangles,
                                                   args.count, args.method, args.seed)
        points, normals = sample_data["positions"], sample_data["normals"]
        area = alpha_stats["estimated_one_sided_visible_area"]
        warnings.append("Alpha-visible area is an independent Monte Carlo estimate; see its standard error in the report.")
        warnings.append("Double-sided samples are paired after Poisson elimination; each side carries a full area weight.")
        warnings.append("Static bind-pose vegetation only: set wind amplitude to zero in the comparison scene.")
        if args.method == "poisson":
            warnings.append("Poisson elimination uses alpha-valid candidates; not an exact fixed-radius sampler.")
    elif args.method == "uniform":
        points, normals, area = sample_uniform(vertices, triangles, args.count, np.random.default_rng(args.seed))
    else:
        points, normals, area = sample_poisson(vertices, triangles, args.count, args.seed)
        warnings.append("Open3D uses target-count sample elimination; no exact r0 is enforced. Area weights are A/N approximations.")
    keep, in_original = range_masks(points, lights)
    if not keep.any():
        raise ValueError("No samples inside doubled ranges; check scale/coordinates/count.")
    # Do NOT renormalize weights after masking: retain original surface quadrature weights.
    weights = (sample_data["area_weights"][keep] if sample_data is not None else
               np.full(int(keep.sum()), area / len(points), dtype=np.float64))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose a fresh empty output directory")
    saved = dict(positions=points[keep].astype(np.float32), normals=normals[keep].astype(np.float32),
                 area_weights=weights, inside_original_range=in_original[keep])
    if sample_data is not None:
        for key in ("surface_ids", "triangle_ids", "categories", "back_side"):
            saved[key] = sample_data[key][keep]
        (output / "surfaces.json").write_text(json.dumps(dict(surfaces=scene["surfaces"],
            materials=scene["materials"]), indent=2, ensure_ascii=False), encoding="utf-8")
        write_ply(output / "samples_categories.ply", points[keep], normals[keep], in_original[keep], saved["categories"])
    np.savez_compressed(output / "samples.npz", **saved)
    write_ply(output / "samples.ply", points[keep], normals[keep], in_original[keep])
    (output / "lights.json").write_text(json.dumps(lights, indent=2, ensure_ascii=False), encoding="utf-8")
    report = dict(input=str(Path(args.input).resolve()), scene=scene["scene"], method=args.method,
                  seed=args.seed, requested_count=args.count, retained_count=int(keep.sum()),
                  outside_original_spheres_count=int((keep & ~in_original).sum()),
                  total_exported_area=area, area_per_sample=float(weights[0]),
                  equivalent_area_radius=math.sqrt(float(weights[0]) / math.pi),
                  filter="union of doubled point/spot range spheres; conservative for spotlights",
                  normal_convention="interpolated world-space vertex normals, no normal maps",
                  lights=len(lights), light_inventory=inventory, warnings=warnings,
                  exporter_notes=scene.get("notes", []))
    if alpha_stats is not None:
        report["texture_sampling"] = alpha_stats
        report["category_counts"] = {name: int((saved["categories"] == i).sum())
                                     for i, name in enumerate(("other", "bark", "foliage"))}
        report["retained_back_side_count"] = int(saved["back_side"].sum())
        ids, counts = np.unique(saved["surface_ids"], return_counts=True)
        report["surface_counts"] = {scene["surfaces"][int(i)]["path"]: int(n) for i, n in zip(ids, counts)}
    (output / "sampling_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {keep.sum():,}/{len(points):,} samples; {len(lights)} lights; area/sample={float(weights[0]):.6g}")
    for warning in warnings:
        print("CHECK:", warning)
    print("Output:", output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Unity export folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("poisson", "uniform"), default="poisson")
    parser.add_argument("--count", type=int, default=100000, help="Sample count BEFORE range filtering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--light-modes", nargs="+", choices=("Realtime", "Mixed", "Baked"), default=["Realtime"])
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    try:
        run(args)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
