# LightOpt

A lighting optimization experiment built with Unity and PyTorch. Export scene geometry and lights, optimize light parameters at sampled surface points, and compare the reference, initial, and optimized lighting in Unity.

## Demo

<video src="demo.mp4" controls playsinline preload="metadata" width="960">
</video>


## Quick Start

Run the commands below in PowerShell from the project root. These examples use Courtyard; substitute the scene and data directories to use Garden or Sponza. Use new export and sample directories for each run to avoid mixing datasets.

### 1. Set Up the Environment

```powershell
Set-Location 'E:\computer science\Paper\Graphics\Code\LightOpt'
conda env create -f config/environment.yml
```

Create the environment once. Subsequent commands use it through `conda run -n lightopt`. The dependencies include PyTorch with CUDA 12.8 support; GPU optimization requires a compatible NVIDIA GPU and driver.

Open `editor_scene/LightOpt_Scene` in Unity Hub using the project's recorded Unity version, **6000.6.7f1**. Wait for asset import and script compilation to finish.

### 2. Export the Scene from Unity

1. Open `Assets/Scenes/Courtyard_LightOpt.unity` and stop Play Mode.
2. Freeze time-of-day changes, lighting controllers, and object animations. For Courtyard, run `Tools > LightOpt > Disable Courtyard Day Night Completely`, then save the scene.
3. Select `Tools > LightOpt > Export active scene`.
4. Choose a new, empty folder inside the project's `data` directory, such as `data/garden_export_004`.

A successful export contains `scene.json`, `vertices.f32`, `triangles.i32`, and `uvs.f32`. If `INCOMPLETE.txt` appears, resolve the export error in the Unity Console and export again to a new, empty directory.

### 3. Generate Samples

```powershell
conda run --no-capture-output -n lightopt python sampling/sample_scene.py data/courtyard_export_006 --output data/courtyard_samples_003 --method poisson --count 100000 --seed 42
```

`--count` specifies the number of candidate samples before filtering. The terminal reports the number actually retained. The output directory contains `samples.npz`, `lights.json`, and `sampling_report.json`. Only Realtime lights are selected by default.

### 4. Optimize and Generate experiment.json

Enable light removal and addition with the current joint loss configuration:

```powershell
python Unity_Visual_Script/optimize_scene.py data/courtyard_samples_003 --device cuda --adjust-lights --rounds 3 --steps 300 --intensity-learning-rate 0.05 --position-learning-rate 0.001 --range-learning-rate 0.001 --initial-scale 0.5 --overlap-weight 0.1 --attraction-weight 0.2 --output data/experiment.json
```

All original lights enter the first round; **there is no initial pruning**. Each round runs continuous optimization, removes lights, and then adds lights. If the last round changes the light set, an additional continuous optimization stage follows.

| Argument | Meaning |
| --- | --- |
| `--adjust-lights` | Enable removal and addition between rounds; omitting it in a noninteractive command keeps the light count fixed |
| `--rounds 3` | Run three outer rounds |
| `--steps 300` | Perform 300 Adam updates per continuous optimization stage |
| `--initial-scale 0.5` | Initialize intensities at half their original values; the reference target still uses the original lights |
| `--overlap-weight` / `--attraction-weight` | Weights of the two regularization terms; the reconstruction weight is 1 |
| `--output` | Output path; must be a `.json` file inside `data` |
| `--device cpu` | Use the CPU when CUDA is unavailable; execution will be slower |

The command generates `data/experiment.json` and the loss plot `data/experiment.png`, and prints a light count report. Reusing an output path overwrites the existing result. Choose a different name to preserve earlier experiments, such as `--output data/experiment_courtyard_002.json`.

To optimize parameters while keeping the light count fixed:

```powershell
python Unity_Visual_Script/optimize_scene.py data/courtyard_samples_003 --device cuda --no-adjust-lights --steps 300 --output data/experiment_fixed.json
```

To optimize reconstruction alone, with both regularization terms disabled:

```powershell
conda run --no-capture-output -n lightopt python Unity_Visual_Script/optimize_scene.py data/garden_samples_004 --device cuda --adjust-lights --rounds 3 --steps 300 --overlap-weight 0 --attraction-weight 0 --output data/experiment.json
```

### 5. Preview the Results in Unity

1. Stop Play Mode and open the source scene used for export.
2. Select `Tools > LightOpt > Optimization preview`.
3. Click `Load Experiment` and select the generated `data/experiment.json`.
4. Click `Generate / Open Preview` to create a comparison scene copy.
5. Use `Reference`, `Before`, and `After` to compare the original lighting, the optimization's initial state, and the final result.
6. Click `Enter Play Mode / Free Fly` to navigate the scene. You can also switch lighting states through the runtime interface.



