# LightOpt 复现

工程根目录：`E:\computer science\Paper\Graphics\Code\LightOpt`。

## 目录

- `editor_scene/LightOpt_Scene/`：已有 Unity 工程。
- `unity/Editor/LightOptSceneExporter.cs`：导出器源码；已安装在 Unity 工程 `Assets/Editor/`。
- `sampling/`：表面采样脚本及其验证。
- `scripts/check_environment.py`：CUDA、采样与分析依赖检查。
- `data/garden_export_001/`：为首次 Unity 导出准备的空目录。
- `config/`：依赖清单 `requirements.txt` 和 conda 配置 `environment.yml`。

## 立即开始

1. Unity 打开 `Garden_LightOpt`，停止 Play Mode。
2. 点击 `Tools > LightOpt > Export active scene`。
3. 选择 `E:\computer science\Paper\Graphics\Code\LightOpt\data\garden_export_001`。
4. 等待 Console 提示导出完成，再运行采样：

```powershell
Set-Location -LiteralPath 'E:\computer science\Paper\Graphics\Code\LightOpt'
conda activate lightopt
python sampling/sample_scene.py data/garden_export_001 --output data/garden_samples_001 --count 100000
```

若 activate 尚未初始化，改用：

```powershell
conda run -n lightopt python sampling/sample_scene.py data/garden_export_001 --output data/garden_samples_001 --count 100000
```

已有 `lightopt` 环境可直接使用。当前只准备数据与采样，不含完整灯光优化器。

## 环境与验证

在工程根目录重建环境：

```powershell
conda env create -f config/environment.yml
```

或在已激活的 Python 3.11 环境中安装依赖：

```powershell
python -m pip install -r config/requirements.txt
```

验证：

```powershell
conda run -n lightopt python scripts/check_environment.py
conda run -n lightopt python -m unittest discover -s sampling -v
```

环境检查只在终端显示结果，不生成报告文件。VS Code/Notebook 内核选择 `Python (lightopt)`。

## 数据与采样约定

- Unity 输出 `scene.json`（网格信息、灯光与排除项）、`vertices.f32`（小端 float32，世界坐标 x,y,z,nx,ny,nz）、`triangles.i32`（小端 int32，全局三角形索引）。
- 仅导出活动场景中启用的静态 MeshRenderer；LODGroup 取最高细节级别。schema v2 支持不透明和已适配的 Alpha Clip 材质，仍排除透明混合和未适配的镂空 Shader。暂不处理 Terrain、蒙皮、粒子、shader 顶点变形和 normal map，也不剔除模型内部表面。法线经过逆转置变换，坐标保持 Unity 世界坐标。
- 默认选启用的 Realtime Point/Spot；检查 Cookie、阴影、颜色温度和灯光接收关系，后续辐照度计算需处理或明确简化。Mixed/Baked 不能直接当成实时灯参与图像对比。
- 先按世界表面积采样，再用原始灯光的两倍 Range 球体并集筛选；聚光灯用保守球形范围。保留暗区，不按目标辐照度是否大于零筛选。
- `--count` 是范围筛选前的点数。固定 seed 默认为 42。每点保留原始面积权重 A/N，筛选后不重新归一化。
- 默认 Open3D Poisson sample elimination 按目标点数生成，不严格保证论文半径 r0；A/N 是等面积近似。`--method uniform` 可用于均匀采样检查。
- 输出 `samples.npz`（位置、法线、面积权重、原始范围标记）、`samples.ply`、`lights.json`、`sampling_report.json`。PLY 橙色仅表示位于原始 Range 球内，蓝色为扩展区域，不代表已计算真实照明或遮挡。
- 数据验证检查点位、法线、尺度和暗区覆盖。真实 Garden 导出与采样结果需单独检查；环境及合成数据测试不替代场景验收。


## 树干、树叶采样（schema v2）

新版导出器已同步到 Unity 工程的 `Assets/Editor/LightOptSceneExporter.cs`。
切回 Unity 等待编译，停止 Play Mode，使用 `Tools > LightOpt > Export active scene`，
选择一个新的空目录，例如 `data/garden_export_002`。旧导出没有叶片 UV 和透明遮罩，不能直接补采树叶。

```powershell
conda run -n lightopt python sampling/sample_scene.py data/garden_export_002 --output data/garden_samples_002 --count 100000
```

### 导出与透明度约定

- schema v1 仍可读；schema v2 保留原来的六浮点顶点布局，并新增 `uvs.f32`（每个顶点两个 float32 UV0）。
- `scene.json` 新增材质和子网格表，记录每段三角形的材质、Alpha Clip 阈值、双面设置、UV 变换与纹理寻址方式。
- `alpha/*.a8` 保存 Unity GPU 回读的导入后纹理 Alpha：uint8，底行在前，x 最快。不会修改纹理的 Read/Write 或导入设置。
- Garden `VegetationSSS_Graph` 使用专门适配：已核对 Graph 的 BaseMap.A、固定 0.5 阈值、UV0（忽略未启用的 ST）、RenderFace Both。不能依赖材质里遗留的 `_AlphaClip=0` 或 `_Cull=2`。
- 支持 URP Lit/Simple Lit、Standard、Legacy Cutout 的标准 Alpha Clip 配置。未识别的自定义镂空 Shader 会写入排除记录；MaterialPropertyBlock 会明确报错，避免漏掉覆盖的纹理或阈值。
- 当前遮罩用 mip 0，Point 或 Bilinear（Trilinear 按 mip 0 Bilinear）；不模拟视角相关 mip、LOD 淡化、叶片透射/SSS、Shader 顶点变形。
- **导出是静态未变形网格。比较 Unity 画面前，将 Garden 植被材质的 Wind/Amplitude（`_Amplitude`）设为 0；单纯停止 Play Mode 不保证 Shader 风停止。** 导出器不会自动修改原始场景或材质。

### 采样、权重和双面

- 先按世界表面积生成候选点，插值 UV，按 Alpha 阈值拒绝透明区域；Poisson elimination 只从通过遮罩的候选点选择，不能在完整叶片矩形上重新生成点。
- 用独立的 100,000 点 Monte Carlo pilot 估计 Alpha 可见单面面积及标准误差。权重为可见单面面积 / 基础采样点数；不能沿用整张面片面积。
- `--count` 是范围过滤前的**基础表面位置数**。Poisson 之后为双面材质复制同位置、反法线样本，各侧携带完整单面权重。最终数量可能大于 `--count`；法线相反的同位置点是有意保留的。
- 两侧样本一起参与后续半球辐照度计算，不能在优化器中再取 `abs(n dot omega)` 或再次翻倍权重。
- 随后仍按两倍灯光 Range 球并集过滤，范围外的树木即便已导出也不会出现在最终采样中。

### 输出与查看

- `samples.ply`：维持原颜色，橙色=原始灯光 Range 球内，蓝色=扩展区域，不等于真实受光/阴影。
- `samples_categories.ply`：灰色=普通几何，棕色=Garden 植被 Bark/Knot 材质，绿色=其他 Garden 植被材质（叶片/花等）；这是材质分类，不是完整植物语义识别。
- `samples.npz`：新增 `surface_ids`、`triangle_ids`、`categories`、`back_side`，便于定位树冠样本、检查双面法线。
- `surfaces.json`：采样输出对应的子网格/材质索引。
- `sampling_report.json`：新增类别数量、逐子网格数量、透明候选剔除数、面积估计及标准误差、反面样本数。
- 可先用 `--count 10000 --method uniform` 验证新导出，再运行默认 Poisson 版本。旧结果不覆盖。
