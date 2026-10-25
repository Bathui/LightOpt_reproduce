import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from algo.light import load_lights, evaluate_irradiance

"""
This file is the entrance of the light calculation from the sample points
"""
def file_sha256(path: Path) -> str:
    """Identify the exact input bytes and the sample ordering they contain."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_target(
    sample_folder,
    output_folder,
    attenuation_smoothness: float = 1.0,
    device: str = "cpu",
) -> None:

    sample_folder = Path(sample_folder).resolve()
    output_folder = Path(output_folder).resolve()

    samples_path = sample_folder / "samples.npz"
    lights_path = sample_folder / "lights.json"

    if output_folder.exists():
        raise FileExistsError(
            f"Choose a new output directory: {output_folder}"
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    # Fingerprint inputs before reading; paths alone do not identify a cache.
    model_path = Path(__file__).with_name("light.py")
    samples_hash = file_sha256(samples_path)
    lights_hash = file_sha256(lights_path)
    model_hash = file_sha256(model_path)

    with np.load(samples_path, allow_pickle=False) as samples:
        positions = samples["positions"]
        normals = samples["normals"]

    # not accept the dimensions that does not meet the requirements
    if (positions.ndim != 2 
        or positions.shape[1] != 3
        or normals.shape != positions.shape
        or len(positions) == 0
    ):
        raise ValueError("Positions and Normals have shape [N, 3], and N must be positive")

    # check if some of the data in positions and normals are inf or nan
    if not(
        np.isfinite(positions).all() and
        np.isfinite(normals).all()
    ):
        raise ValueError("Sampled data has inf or nan")

    normal_lens = np.linalg.norm(normals, axis=-1)

    if not np.allclose(normal_lens, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Sample normal must be normalized")

    # convert to tensor used in torch
    positions = torch.as_tensor(positions, 
                                dtype=torch.float32, 
                                device=device)

    normals = torch.as_tensor(normals, 
                                dtype=torch.float32, 
                                device=device)

    lights = load_lights(str(lights_path),
                         attenuation_smoothness=attenuation_smoothness,
                         device=device,
                         )

    if not lights:
        raise ValueError("No lights available for target generation")

    # we are calculating the target value so we do not need grad
    with torch.no_grad():
        target = evaluate_irradiance(
            positions,
            normals,
            lights
        )

    # validate sampled irradiance before saving
    if target.shape != positions.shape:
        raise ValueError("target irradiance must have shape[N,3]")

    if not torch.isfinite(target).all():
        raise ValueError("Target irradiance contains NaN or Inf")

    if (target < 0).any():
        raise ValueError("Target irradiance contains negative values")

    # Do not publish a target if its inputs changed during computation.
    if (file_sha256(samples_path) != samples_hash
        or file_sha256(lights_path) != lights_hash
        or file_sha256(model_path) != model_hash
    ):
        raise RuntimeError("Input files or light model changed during target generation")

    target_array = target.cpu().numpy()

    metadata = {
        "sample_file": str(samples_path),
        "light_file": str(lights_path),
        "sample_sha256": samples_hash,
        "light_sha256": lights_hash,
        "model_source_sha256": model_hash,
        "model": "LightOpt equations 3-5",
        "sample_count": len(positions),
        "light_count": len(lights),
        "target_shape": list(target_array.shape),
        "dtype": str(target_array.dtype),
        "device": device,
        "attenuation_smoothness": attenuation_smoothness,
        "visibility": 1.0,
        "cookies": "unsupported; rejected by loader",
        "color_temperature": "unsupported; rejected by loader"
    }

    # create output folder
    output_folder.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        output_folder / "target_irradiance.npz",
        irradiance=target_array
    )

    # write metadata as a json
    metadata["complete"] = True
    (output_folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Saved target: {output_folder}")
    print(f"Samples: {len(positions)}, lights: {len(lights)}")
    print(f"Target shape: {target_array.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and cache fixed target irradiance."
    )
    parser.add_argument("sample_folder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoothness", type=float, default=1.0)

    args = parser.parse_args()

    build_target(
        sample_folder=args.sample_folder,
        output_folder=args.output,
        attenuation_smoothness=args.smoothness,
        device=args.device,
    )
