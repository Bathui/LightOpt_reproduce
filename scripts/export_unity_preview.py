"""Export the three already-verified intensity experiments for a Unity preview."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "editor_scene/LightOpt_Scene"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Intensity must be finite and nonnegative")
    return float(value)


def vector(x, y, z):
    return dict(x=x, y=y, z=z)


def gamma_color(linear):
    convert = lambda x: 12.92*x if x <= .0031308 else 1.055*x**(1/2.4)-.055
    return dict(zip(("r", "g", "b", "a"), [*(convert(x) for x in linear), 1.]))


def synthetic_lights():
    return [
        dict(id="synthetic_point", path="Point", type="Point", position=vector(0.,1.,0.), direction=vector(0.,-1.,0.),
             colorRaw=gamma_color((1.,.5,.2)), range=2., intensity=2., innerSpotAngleDegrees=0., outerSpotAngleDegrees=30.),
        dict(id="synthetic_spot", path="Spot", type="Spot", position=vector(3.,1.,0.), direction=vector(0.,-1.,0.),
             colorRaw=gamma_color((.2,.5,1.)), range=2., intensity=3., innerSpotAngleDegrees=math.degrees(.4), outerSpotAngleDegrees=math.degrees(1.)),
    ]


def make_experiment(result, records, name, family, scene_guid="", scene_hash=""):
    reference = result["reference_intensities"]
    after = result["final_intensities"]
    if len(reference) != len(records) or len(after) != len(records):
        raise ValueError("Experiment light count does not match its source")
    scale = nonnegative(result["initial_intensity_scale"])
    entries = []
    ids, paths = set(), set()
    for record, original, optimized in zip(records, reference, after):
        original, optimized = nonnegative(original), nonnegative(optimized)
        if not math.isclose(original, record["intensity"], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Reference intensity no longer matches the source light")
        if record["id"] in ids or record["path"] in paths:
            raise ValueError("Duplicate light identity")
        ids.add(record["id"])
        paths.add(record["path"])
        if record["type"] not in ("Point", "Spot"):
            raise ValueError("Only Point/Spot experiments are supported")
        if family != "synthetic":
            match = re.fullmatch(r"GlobalObjectId_V1-2-([0-9a-f]{32})-\d+-\d+", record["id"])
            if not match or match.group(1) != scene_guid:
                raise ValueError("Light GlobalObjectId belongs to a different scene")
            if record.get("mode") != "Realtime" or not record.get("enabled") or not record.get("active") or record.get("hasCookie") or record.get("useColorTemperature"):
                raise ValueError("Require active Realtime lights without cookies or temperature")
        entry = {key: copy.deepcopy(record[key]) for key in ("id", "path", "type", "position", "direction", "colorRaw", "range", "innerSpotAngleDegrees", "outerSpotAngleDegrees")}
        entry.update(referenceIntensity=original, beforeIntensity=nonnegative(original*scale), afterIntensity=optimized)
        entries.append(entry)
    return dict(name=name, family=family, sourceSceneGuid=scene_guid, sourceSceneSha256=scene_hash,
                steps=result["steps"], learningRate=result["learning_rate"],
                initialLoss=result["initial_loss"], finalLoss=result["final_loss"], lights=entries)


def create_bundle(report, records, scene_guid, scene_hash):
    def find(name):
        return next((case for case in report["garden"] if case["name"] == name), None)
    default = find("garden_default_cuda") or find("garden_default_cpu")
    tuned = find("garden_lr02_cuda") or find("garden_lr02_cpu")
    if default is None or tuned is None:
        raise ValueError("The report must contain default and tuned Garden experiments")
    return dict(schemaVersion=1, experiments=[
        make_experiment(report["synthetic"], synthetic_lights(), "1. Point + Spot 小场景", "synthetic"),
        make_experiment(default, records, "2. Garden：默认配置 / 300 步", "garden", scene_guid, scene_hash),
        make_experiment(tuned, records, "3. Garden：调参配置 / 600 步", "garden", scene_guid, scene_hash),
    ])


def export(report_path, lights_path, scene_path, output):
    report_path, lights_path, scene_path, output = map(lambda p: Path(p).resolve(), (report_path, lights_path, scene_path, output))
    if output.exists():
        raise FileExistsError("Preview data already exists; choose a new output file")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for section in ("source_sha256", "input_sha256"):
        for relative, expected in report[section].items():
            path = (ROOT / Path(relative.replace("\\", "/"))).resolve()
            if not path.is_relative_to(ROOT) or sha256(path) != expected:
                raise ValueError("Saved experiment input/source changed: " + relative)
    expected_lights_hash = next((value for key,value in report["input_sha256"].items() if key.replace("\\", "/").endswith("garden_samples_004/lights.json")), None)
    if sha256(lights_path) != expected_lights_hash:
        raise ValueError("Light file does not match the tested Garden experiment")
    records = json.loads(lights_path.read_text(encoding="utf-8-sig"))
    match = re.search(r"^guid: ([0-9a-f]{32})$", Path(str(scene_path)+".meta").read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise ValueError("Cannot read source scene GUID")
    bundle = create_bundle(report, records, match.group(1), sha256(scene_path))
    bundle.update(sourceReportSha256=sha256(report_path), sourceLightsSha256=sha256(lights_path),
                  note="Saved Python intensity experiments. Unity uses the same parameters for qualitative A/B viewing; this does not establish shader calibration.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT/"tmp/validation/intensity_optimization/report.json"))
    parser.add_argument("--lights", default=str(ROOT/"data/garden_samples_004/lights.json"))
    parser.add_argument("--scene", default=str(PROJECT/"Assets/Scenes/Garden_LightOpt.unity"))
    parser.add_argument("--output", default=str(PROJECT/"Assets/LightOptPreviewData/experiments.json"))
    args = parser.parse_args()
    bundle = export(args.report, args.lights, args.scene, args.output)
    print("Exported", len(bundle["experiments"]), "saved experiments to", args.output)
