"""Smoke-check LightOpt sampling, clustering, plotting, and CUDA autograd."""
import io
import json
import platform
import sys
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy
from sklearn.cluster import DBSCAN
import torch
import trimesh
import yaml
from torch.utils.tensorboard import SummaryWriter


def main():
    assert torch.cuda.is_available(), "CUDA is unavailable"
    device = torch.device("cuda")
    # Check an actual GPU kernel and backward pass, rather than device enumeration.
    x = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
    x.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    matrix = torch.eye(64, device=device)
    torch.testing.assert_close(matrix @ matrix, matrix)
    torch.cuda.synchronize()

    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(42)
    cloud = mesh.sample_points_poisson_disk(100)
    assert len(cloud.points) == 100
    assert np.isfinite(np.asarray(cloud.points)).all()

    labels = DBSCAN(eps=0.2, min_samples=2).fit_predict(
        np.array([[0., 0., 0.], [0.1, 0., 0.], [2., 0., 0.], [2.1, 0., 0.]]))
    assert len(set(labels)) == 2 and -1 not in labels
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    assert buffer.getbuffer().nbytes > 0
    assert yaml.safe_load("ready: true")["ready"] is True
    assert trimesh.creation.box().area > 0

    packages = ["torch", "numpy", "open3d", "scipy", "scikit-learn", "matplotlib",
                "trimesh", "tqdm", "PyYAML", "tensorboard", "pytest", "ipykernel"]
    report = {
        "python": platform.python_version(), "executable": sys.executable,
        "packages": {name: version(name) for name in packages},
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "checks": ["CUDA tensor operations", "CUDA backward", "CUDA matrix multiply",
                   "Open3D Poisson sampling", "DBSCAN", "Matplotlib PNG", "mesh IO dependencies"],
        "status": "passed",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
