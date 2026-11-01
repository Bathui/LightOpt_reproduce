# 单场景联合优化与 Unity 对比数据

## Unity 增删灯结果界面（2026-10-30）

停止Play，等待编译后，打开`Tools > LightOpt > Optimization preview`，点击`Load Experiment`加载`optimize_scene.py --adjust-lights`生成的schema3文件，例如`data/scene_light_set/garden_smoke.json`或`data/scene_light_set/sponza_smoke.json`，再点击`Generate / Open Preview`。

面板显示Before/After灯数、累计Removed/Added和每轮记录。Before/Reference使用源比较灯，After使用最终参数的独立预览对象；不把不同长度数组的同一索引当作同一盏灯。预览对象总数可以大于任何单个状态的灯数，Light Count按当前状态统计。Play HUD显示累计增删摘要，按1/2/3切换同时更新灯的enabled、位置、Range和强度。旧预览需重新点击Enter Play Mode / Free Fly更新序列化运行数据。

当前已通过Runtime/Editor实际编译及两份真实JSON的计数账目审计；未实际进入GUI/Play检查布局和画面。以下早期记录中“Unity不支持schema3”的限制已被本次读取器更新替代。原场景保持不变。

## 在原 optimize_scene.py 中启用增删灯（2026-10-29）

同一个入口现已支持固定灯数与增删灯两种模式：

```powershell
conda run --no-capture-output -n lightopt python Unity_Visual_Script/optimize_scene.py data/garden_samples_004 --adjust-lights --rounds 3 --steps 100 --overlap-weight 0 --attraction-weight 0
```

不带目录交互启动时，会先询问是否启用增删灯，默认1；然后可输入外循环轮数、重要性阈值、DBSCAN邻域/最少样本、每轮最多新增灯数，以及已有的学习率、损失权重和初始强度比例。命令行带目录时需显式加`--adjust-lights`开启；`--no-adjust-lights`或不加开关保留原固定灯数模式。原权重默认值仍为1/.1/.2；上面的命令明确用R-only验证增删流程。

增删模式`--steps`是每个连续优化阶段的步数，最后一次集合变化后还会补一个阶段。默认输出`data/experiment_light_set.json`和同名PNG；`--output`可选择其他结果路径，`--no-plot`可只存JSON。输出schemaVersion=3/format=lightopt.scene_light_set.v1，含原场景身份、完整配置和源码哈希、独立reference/before/after灯列表与每阶段损失/删增日志，不再使用固定长度zip。`debug_light_set.py --start-from`也可读取其after状态续跑。

固定模式继续输出schemaVersion=2到`data/experiment.json`，兼容现有Unity预览。Unity当前只读取schema 1/2，不能加载增删模式的schema 3；本次没有修改Unity读取器。增删模式的保存与重算已通过测试，不代表Unity可变灯集合显示已完成。

本轮36项相关测试全部通过（新增入口10项、原入口11项、调试/续跑15项），0跳过。真实CUDA短程：Garden 13->7灯，R33.729694->27.603821；Sponza初始强度.01，21->25灯，R1206.800293->945.338013。结果为`data/scene_light_set/`，报告`tmp/validation/scene_light_set/report.json`。短程检查仅验收执行和导出，不是收敛或论文质量验收。

## 可变灯数的 Python 调试入口（2026-10-29）

在项目根目录运行：

```powershell
conda run --no-capture-output -n lightopt python Unity_Visual_Script/debug_light_set.py data/sponza_samples_001 --device cuda
```

也可以不带采样目录启动，再输入目录。默认每个连续阶段100步、外循环3轮、初始强度0.5、仅重建损失；重要性阈值0.1、DBSCAN邻域距离0.5米、最少8点、每轮最多增8灯。它们是调试起点，不是论文默认值。`--steps`表示每个连续阶段的步数；若最后增删过灯，还会增加一次连续优化。

此入口调用用户已接入的 `optimize_remove_add()`，保留全部原始灯进入第一轮，逐轮连续优化/删灯/增灯，允许空灯集合重新增灯。目标始终由原始灯光计算，新增灯进入下一轮真实Adam参数集合。每个连续阶段重新创建Adam并重置动量，初始强度比例只应用一次。

默认输出 `data/light_set_debug/<采样目录名>.json` 和同名PNG。JSON包含独立的reference/before/after完整灯列表、每轮优化历史、删灯索引与重要性、聚类噪声/跳过原因、新灯参数摘要、删灯/增灯前后四项损失、最终欠照明/实际重叠数以及输入源码哈希。灯数不同也完整保存，不按原灯列表zip裁剪。PNG上蓝线是连续优化，橙色点线是删灯/增灯瞬间的重建误差。

JSON格式为 `lightopt.light_set_debug.v1`，用于Python检查和续跑；当前Unity预览仍读取原先的一一对应实验格式，不能加载此文件。原始scene、采样、lights.json和已有experiment*.json不会被此入口改写，输出路径指向原实验文件时会拒绝覆盖。

从旧Sponza实验的after状态继续调试：

```powershell
conda run --no-capture-output -n lightopt python Unity_Visual_Script/debug_light_set.py data/sponza_samples_001 --device cuda --start-from data/experiment_sponza.json --steps 20 --rounds 3 --output data/light_set_debug/sponza_resume.json
```

`--start-from`也支持本入口生成的结果，读取其after状态；默认初始比例为1，避免再次减半。原始目标仍来自同一采样目录中的原灯，续跑前核对采样/灯光哈希。

显式开启当前联合损失：

```powershell
conda run --no-capture-output -n lightopt python Unity_Visual_Script/debug_light_set.py data/sponza_samples_001 --device cuda --steps 100 --rounds 3 --overlap-weight 0.1 --attraction-weight 0.2 --output data/light_set_debug/sponza_joint.json
```

增删流程不会自动修复当前负Overlap/Spot Attraction候选问题。真实联合运行已经发现后续连续优化会抵消增灯的重建收益，不能以total变小或没有报错作为质量验收。Merge仍未实现，本入口不是完整Algorithm 1。

专项检查：

```powershell
conda run --no-capture-output -n lightopt python -m unittest discover -s tests -p 'test_light_set*.py' -v
```

本轮已完成12项核心测试、15项调试/保存/续跑测试，以及11项现有固定灯数预览入口回归；均通过，无跳过。四组真实CUDA运行及独立JSON回读重算的结果见 `tmp/validation/light_set_debug/report.json`。未执行Unity动态灯集合显示验证。

## 当前运行方式（2026-10-28）

在项目根目录运行：

```powershell
conda run --no-capture-output -n lightopt python Unity_Visual_Script/optimize_scene.py
```

交互输入采样目录、Adam 步数、Intensity/Position/Range 三个学习率、Overlap/Attraction 权重和初始强度比例。输入目录需在 data 内，并包含 samples.npz、lights.json。

本轮实际运行命令：

```powershell
conda run --no-capture-output -n lightopt python Unity_Visual_Script/optimize_scene.py data/garden_samples_004 --steps 600 --intensity-learning-rate 0.2 --position-learning-rate 0.001 --range-learning-rate 0.001 --overlap-weight 0.1 --attraction-weight 0.2
```

默认参数为 300 步，三个学习率 .05/.001/.001，损失权重 1/.1/.2，初始强度比例 .5。它们是当前实验配置，不是已经验证的论文最优参数。可以用 --output data/experiment_other.json 保留其他实验。

## 数据与优化

调用现有 optimize_intensities，当前训练 intensity、position、range，方向/角度/颜色固定。原始灯光计算一次固定目标，采样和目标不随参数更新。Overlap/Attraction 使用现有实现及候选集合；本工具没有修改论文公式或符号实验选择。

data/experiment.json 保存本次真实 Adam 输出、逐步四项损失、完整配置、输入/源码哈希。每盏灯包含 reference/before/after 的 Intensity、Position、Range。initialLoss/finalLoss 仍表示 reconstruction；initialTotalLoss/finalTotalLoss 表示加权总损失。

本版输出 schemaVersion=2。Assets/Editor/LightOptOptimizationPreview.cs 已同步读取三组强度、世界坐标位置和 Range；兼容旧版单场景 schemaVersion=1（位置和 Range 固定）。初次生成校验原始参数，后续 Resolve 仅约束身份和固定属性。Transform 与 Light 均记录 Undo/Prefab 覆盖，支持反复切换。

操作入口为 Tools > LightOpt > Optimization preview，选择原始场景、加载 data/experiment.json、生成副本，再切换三个状态。若面板仍显示旧提示，等待编译完成并关闭后重新打开面板。无需重跑 Python。C# 本轮已编译通过；尚未实际点击 GUI 验证切换或画面。

## 本轮实际结果

- 11 项 Python 专项测试通过，覆盖真实 Adam、三组学习率和权重传入、完整几何结果与模型逐值匹配、总损失组合、Overlap/Attraction 非零小案例、异常输入和输出保护。它们验证接口接入，不等于所有 Attraction 梯度方向都已验证。
- Garden CUDA，86114 样本/13 灯/600 步：重建损失 33.7296944 -> 0.0552676171。8 盏灯移动/改变 Range，最大位移约 .79567，最大 Range 变化约 .55969（项目世界单位）。
- Overlap 601 条记录均为 0；Attraction 462 条记录非零，最终为 0。
- 同初值、学习率和步数的重建-only 对照：最终重建损失同为 .0552676171，7 盏灯移动/改变 Range，最大位移约 .08523。不能声称正则项改善了本次重建误差。
- 报告 tmp/validation/joint_scene_optimization.json；前一版 experiment 已备份到 tmp/review/joint-scene-optimization/experiment.before.json。旧算法源码、原始灯光/采样文件和用户材质修改未由本工具改写。

当前 project_constraints 只投影 intensity>=0、range>=1e-4；尚没有场景位置边界、Range 上限或动态重采样，不据此宣称完整模块 4 已完成。Python 使用 V=1 的解析辐照度，仍不能替代 Unity 图像校准。


## 全局相机视角

新建测试副本时默认按场景中启用网格的世界包围盒，生成斜上方全局视角，并按当前相机宽高比留出边距。已有副本点击“全局视角（俯视整个场景）”即可重新取景，无需重跑 Python 或重新生成副本。

若全场景包含较远的背景、导致庭院显得太小，可在 Scene 窗口导航到想要的角度，再点击“使用当前 Scene 视图角度”。两个按钮都会保存副本相机并更新锁定视角，之后切换优化前后保持该视角。

本轮 Unity 工程依赖下 C# 编译通过，尚未实际点击 GUI 确认全景效果。


## Play Mode 自由漫游

等待 Unity 编译后，编辑模式下打开对应测试副本，在面板先选择“优化前 / 优化后 / 原始参考”，再点击“进入 Play 自由漫游”。已有副本会安装独立的根节点导航相机并保存，随后进入 Play；新副本自动安装自由相机，也可直接按 Unity Play。

先点击 Game 窗口，然后按住鼠标右键：移动鼠标转向，W/S 前后，A/D 左右，Q/E 世界坐标下降/上升，Shift 加速，滚轮调整速度，R 回到本次 Play 的起点。松开右键或按 Esc 释放鼠标。

这是无重力、无碰撞的飞行相机，可穿过屋顶/墙体检查庭院内部。停止 Play 后运行中的移动不保存；如需设置下一次起点，可在编辑模式调整 Scene 视角，再点击“使用当前 Scene 视图角度”。运行中编辑器面板按钮保持禁用；使用 Game 内热键 1/2/3 或屏幕按钮切换，无需停止 Play。

只在测试副本内停用原 FirstPersonController、StarterAssetsInputs、PlayerInput、CinemachineBrain 与其他相机/AudioListener；独立相机复制原 URP 渲染设置，直接输出到 Game。原场景、优化数据不改。新增运行时脚本位于 Assets/LightOpt/Runtime/LightOptFlyCamera.cs，使用本项目已启用的 Input System。

本轮 Runtime 与 Editor 两个程序集均用 Unity 自带编译器及工程依赖编译通过。尚未实际进入 Play 或验证键鼠操作、渲染效果；不将编译通过表述为运行验证。


## 漫游中热切换灯光

已有副本先退出 Play，等待编译，再从对比面板点击一次“进入 Play 自由漫游”，安装并保存最新运行时切换组件。新生成副本自动包含该组件。

Game 窗口内按 1（优化前）、2（优化后）、3（原始参考），主键盘与数字小键盘均支持。可以按住右键漫游时直接按数字；也可松开右键或按 Esc 释放鼠标，点击左上角对应按钮。当前状态显示在屏幕上。

切换同时应用每灯的 intensity、世界坐标 position 和 range，不调用相机锁定/恢复，不修改当前漫游位置、朝向、速度。使用进入 Play 前加载的三组结果，不在运行中重新优化或读取更新后的 JSON。停止 Play 后运行时切换不会写回原始场景或 experiment.json。

实现位于 Assets/LightOpt/Runtime/LightOptRuntimeComparison.cs；编辑器将已匹配的灯引用与三状态参数序列化到副本导航相机。Runtime 与 Editor 程序集编译通过，尚未实际进入 Play 验证键鼠与画面。


## 热切换无变化的检查

Garden 的 DynamicLightController 即使被禁用，也可能被 DynamicLightsManager 的 SceneCamera Reset 路径关闭灯并覆盖参数。工具现会从测试副本的对比灯中移除该控制器，并在导航相机渲染前恢复所选三状态。原场景和 prefab 资产不修改。

更新后先停止 Play，再从面板点击“进入 Play 自由漫游”。观察 Game 左上角：当前 Garden 数据参与对比 13 灯，启用与参数匹配应为 13；按 1/2/3，强度总和应约为 114 / 207.01 / 228。若数值变化但画面不明显，应在庭院/走廊受光表面近距离比较，并区分固定烘焙、自发光、其他未参与灯光贡献。本轮确认代码缺陷并编译通过，未实际复现或验证 Play 画面。


## English UI and loss display

The preview window and runtime controls now use English. Open Tools > LightOpt > Optimization preview, click Load Experiment, select the experiment JSON, then Generate / Open Preview. Camera actions are Overview Camera (Entire Scene), Use Current Scene View, and Record Current Camera View. Enter Play Mode / Free Fly installs or updates runtime comparison data for an existing preview.

The editor panel shows Reconstruction, Overlap, Attraction, and Total in Before/After columns, with saved loss weights. These are the first and last history records from Python, not live Unity image metrics. Runtime keys 1/2 select the corresponding saved losses. Reference (3) displays N/A because its losses were not recorded. Negative values are preserved; missing metrics are not shown as zero. No new optimization run is needed.

After updating scripts, stop Play, reload the experiment JSON, then use Enter Play Mode / Free Fly once to update metrics on an existing preview. Both assemblies compiled successfully; the actual UI layout and Play interaction have not been verified in this turn.
