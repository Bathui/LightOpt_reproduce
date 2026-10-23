# LightOpt 项目记忆

最后更新：2026-10-23（America/Los_Angeles）。状态描述是最后一次检查的记录，后续任务须核对当前文件再继续。

## 目标和范围

用户自行复现 **LightOpt: Lights Optimization for Real-time Rendering**，重点理解并实现 core algorithm，避免重建 PT/RT 或完整游戏引擎。采用 Unity 提供场景、几何和灯光，Python/PyTorch 实现解析可微灯光模型和优化，再回 Unity 验证。

论文：`E:\computer science\Paper\Graphics\lightopt-paper.pdf`。
作者项目页：https://mirage-c.github.io/LightOpt/ 。
实际工程：`E:\computer science\Paper\Graphics\Code\LightOpt`。
Unity 工程：`editor_scene/LightOpt_Scene/`。
场景：官方 Universal 3D Sample 的 Garden，实验副本 `Assets/Scenes/Garden_LightOpt.unity`。
旧 C 盘聊天工作目录不存在，不要在那个位置建立另一份实现。

## 固定参考路线：6 个主体模块 + 1 个可选扩展

| 模块 | 主要工作 | 论文位置 | 验收重点 |
| --- | --- | --- | --- |
| 1. 场景数据与采样 | 读取固定几何与原始灯光；表面 Poisson disk 采样；构造 S_all；缓存原始目标辐照度 | §3.2，并依赖 §3.1 前向模型 | 数据有效、保留暗区、目标缓存及输入关联可复核 |
| 2. 可微灯光模型 | 点光源/聚光灯的距离衰减、角度衰减和法线余弦；每灯与总辐照度 | §3.1，式 (3)-(5) | 数值、广播、边界、自动梯度正确 |
| 3. 损失函数 | 辐照度重建误差、Overlap Loss、Attraction Loss；对应样本/候选灯集合 | §3.3-3.4 | 每项作用明确，数值与梯度方向正确 |
| 4. 连续优化与调度 | 固定灯数优化参数；参数约束、收敛判断；交替调用连续优化和离散操作 | Algorithm 1 | 误差可下降，状态与灯集合更新一致 |
| 5. 灯集合调整 | 删除低贡献灯、合并冗余灯、在欠照明区域新增灯 | §3.5-3.7 | Remove/Merge/Add 可分别检查，再接入外循环 |
| 6. 验证与可视化 | 目标/优化辐照度、误差、覆盖数、灯数；引擎图像和 GPU 时间对比 | §4 | 多视角、消融、真实光照阶段性能 |
| 7. ToD（可选，后做） | 汇总时间关键帧、参数插值、按时间积分权重联合优化 | §3.8 | 静态流程稳定后再扩展 |

端到端流程：

```text
固定 Unity 实验场景
  -> 导出几何/灯光/评测相机
  -> 表面采样与两倍 Range 邻域筛选
  -> 原始灯光前向计算并缓存 E_target
  -> 连续参数优化（重建 + overlap + attraction）
  -> 删除 / 合并 / 补灯
  -> 重复优化并导出新灯集合
  -> Unity 多视角画面与光照阶段 GPU 时间验证
  -> 可选 ToD
```

注意模块边界：模块 1 的目标缓存需要先实现模块 2 的最小前向计算。当前推进这两部分是依赖关系，不是另起一套模块编号。

## 光照和数学约定

- 优化固定表面样本的直接辐照度；优化阶段可见性 V=1，不计算阴影遮挡。
- 主要灯型为 Point/Spot；Directional 等作为固定背景，不纳入当前变量。
- Mixed/Baked 是烘焙模式，Directional 是类型，不是同一维度的分类。Mixed 灯能否纳入需检查其直接光与烘焙贡献，不能直接改模式并保留旧烘焙而忽略重复照明。
- 原始目标可以由相同解析模型计算，也可从 Unity 专门导出。相同模型的自洽测试不等于与 Unity 完全一致；精确引擎比较需对齐当前 URP 的公式、单位、颜色处理。
- Unity 返回辐照度数值不能自动恢复其与灯参数的计算图；PyTorch 前向函数用于求当前灯配置的结果及参数梯度。
- `omega = normalize(light_position - sample_position)`；灯 `direction` 指向灯向外照射方向，Spot 轴向项使用 `dot(-omega, direction)`。
- 基础颜色用线性 RGB；Cookie、色温不能静默忽略。当前加载器对二者明确报错。
- Unity 导出的完整角度先除以 2，再转弧度；内部参数存半角。
- 距离衰减：`a/(a+d2) * max(0,1-d2/range**2)**2`。`a` 单位为距离平方，与材质 Smoothness 无关；当前导出没有 a，需显式配置。起步 a=1 是实现选择，不是论文默认值。
- 表面余弦：`max(dot(normal, omega), 0)`。不取 abs，不再次复制双面叶片。
- 前向辐照度不乘 area_weights、不乘 albedo/BRDF、不裁剪到 [0,1]；积分和重建损失才使用面积权重。
- 前向函数保留计算图，不使用 NumPy、`.item()`、`.detach()` 或重新包装已有 Tensor。JSON 初始化可使用 Python/math；目标缓存由调用方在 `torch.no_grad()` 下生成。
- CPU 小例子调试，GPU 批量前向/优化；全部参与运算的 Tensor 必须位于相同 device/dtype。
- Attraction Loss 的论文式 (10)/(17) 符号方向及式 (15)/(16) 搜索半径曾发现值得核对的不一致。做模块 3 时先用单灯/单点测试梯度方向，不要无解释照抄或自行宣称修正是作者原意。

## 已有实现与数据

- `unity/Editor/LightOptSceneExporter.cs`：导出器源码；安装副本为 Unity 工程 `Assets/Editor/LightOptSceneExporter.cs`。改导出器时保持两份同步。
- `sampling/sample_scene.py`：采样入口；`sampling/alpha_sampling.py`：UV/Alpha 有效候选与双面处理。
- `algo/light.py`：LightParameters、single_light_info、load_lights、几何/距离/角度衰减、Point/Spot/总辐照度。
- `tests/test_light_validation.py`：实际 PyTorch 前向、加载器、CPU/CUDA、梯度回归检查。
- `data/garden_export_001` 是旧 schema 1，无树；不能通过重新采样恢复已排除的树。
- `data/garden_export_002` 是 schema 2，已导出 Cherry_tree_01 的四个子网格、UV 和 Alpha。
- `data/garden_samples_002` 与 `garden_samples_003` 最后核实都来自 export_002。重跑前必须看 sampling_report.json 的 input，不能仅凭输出目录编号判断版本。
- 最新核实的采样记录数：86,114；分类 other=76,742、bark=2,462、foliage=6,910；中央 Cherry_tree_01 四个子网格合计 3,078 条记录（含双面配对）。
- 采样目录的 lights.json 是筛选后的启用/激活/正强度/正 Range 的 Realtime Point/Spot，当前 13 盏（6 Point + 7 Spot），不是活动场景全部灯。更完整灯清单在导出 scene.json 的 lights。

### 植被采样和 PLY

- Garden 植被使用 VegetationSSS_Graph，已核对 Shader Graph：BaseMap.A -> Alpha，固定 cutoff=.5，RenderFace Both，UV0 不启用纹理 ST。
- 不依赖材质中遗留的 `_AlphaClip=0`、`_Cull=2` 判断此 Graph 的真实行为。
- 根据插值 UV 查询导入后的 mip 0 Alpha（GPU 回读 a8，底行在前），拒绝透明空白，再在有效候选上做 Poisson elimination。
- 当前是目标点数控制的 Poisson sample elimination，不严格保证论文固定 r0；透明可见面积通过独立 100,000 点 Monte Carlo pilot 估计，报告含标准误差。
- Poisson 之后为双面材质配对相反法线，各侧携带完整单面面积权重；同位置反法线点是有意保留。
- `--count` 是范围过滤前的基础位置数，双面配对后最终记录数可能更多。
- 用原始 Point/Spot 两倍 Range 球并集筛选，Spot 使用保守球；不按 E_target>0 删除暗区，不在范围过滤后重新归一化面积权重。
- `samples.ply`：橙色=原始灯 Range 球内，蓝色=扩展球内但原始球外；不等于真实亮区/阴影，不做锥角或可见性判断。
- `samples_categories.ply`：同一批点，灰色普通几何、棕色 Garden Bark/Knot 材质、绿色其他 Garden 植被材质。它只是检查用配色；算法读取 samples.npz。
- samples.npz 含 positions、normals、area_weights、inside_original_range、surface_ids、triangle_ids、categories、back_side；目前没有 E_target。
- 比较 Unity 画面前需冻结风/Shader 顶点变形（例如植被材质 `_Amplitude=0`）；导出的是静态未变形网格，停止 Play Mode 不保证风停止。导出器不会自动修改原材质或场景。

## 最后一次验证状态：模块 1 未完成，light.py 回归测试全部通过

2026-10-23：用户授权修复后，已修复两处核心错误，并实际运行原有 22 项 PyTorch 测试：**22 通过、0 失败、0 错误、0 跳过**。未修改测试断言来迎合实现。

已修复：
1. `compute_light_geometry` 保存真实距离平方，只对归一化分母的平方距离 clamp 后开方；修复非单位距离衰减、Range 边界和重合位置 NaN 梯度。
2. `evaluate_spot_light` 使用 `angular_factor.unsqueeze(-1) * base_irradiance`，逐采样点缩放 RGB。

本次检查包括加载器正常/异常输入、顺序与参数/device 传递、方向归一化和半角、距离与角度手算、Range 边界、N=2/N=3 的广播、混合灯累加、空集合、非边界 gradcheck、重合位置有限梯度、CPU/CUDA 点光源一致性与 CUDA backward。

- 非单位距离 d=2、r=4、a=1、I=1、红色通道 c=1：修复后为正确的 .1125。
- 实际硬件：RTX 5090 Laptop GPU；torch 2.11.0+cu128，CUDA 可用。旧的 `_ctypes` 拦截已不影响最后一次测试，不要未经重试声称仍然被拦截。
- 原始数据未改动：86,114 条采样记录的基础检查通过；真实 lights.json 仍有 6 盏 Cookie、12 盏色温，因此真实加载仍会按当前约定拒绝输入。这不是本次修复范围。
- 六盏 Point 的 Range=.13 米，没有范围内样本。不能据此认定它们在真实画面中不重要，也不要随意扩大 Range 来制造贡献。
- 尚未生成原始目标辐照度缓存，目标统计/照明可视化未完成，所以模块 1 仍未验收。

证据：`tmp/validation/light_validation_report.json` 保存被测源码 SHA256 和结果；`tmp/review/light-forward-fixes.before.py` 为真实修改前快照，`tmp/review/light-forward-fixes.diff` 为准确差异。`tmp/validation/audit_light_data.py` 已改为动态运行测试，不再写死旧的失败统计。

## 下一步顺序

1. 明确 Cookie/色温策略，并同步 Unity 实验配置、导出数据和目标定义；不要仅改 JSON 标记来伪装已处理。
2. 编写目标生成入口：读取 samples.npz / lights.json，调用总辐照度，在 no_grad 下缓存 RGB E_target[N,3]。
3. 保存输入文件 hash、模型版本、a、颜色/可见性策略；统计每灯非零样本数、面积积分贡献、最大值，检查有限性/非负值并做照明可视化。保留零目标样本。
4. 完成上述闭环才验收模块 1，然后进入模块 3 的损失构造和模块 4/5 外循环。

## 常用命令

在实际工程根目录运行：

```powershell
conda run -n lightopt python -m unittest discover -s tests -p test_light_validation.py -v
conda run -n lightopt python -m unittest discover -s sampling -v
conda run -n lightopt python sampling/sample_scene.py data/garden_export_002 --output data/NEW_EMPTY_SAMPLE_FOLDER --count 100000
```

采样和 Unity 导出使用新的空输出目录，不覆盖既有数据。不要凭旧 README 中的 001 起步命令误用 schema 1。

## 协作与记忆维护

- 用户希望自己手写并理解算法，常请求参考代码、伪代码和检查。保留这种学习方式，不要把参考代码请求自动变成全量实现。
- 用户明确要求所有代码注释和 docstring 使用英文，交流使用中文。
- 用户关心改动对比。最后检查工程没有 Git，旧聊天路径已失效；缺对比时输出有依据的 diff，不声称它是 Git 历史。此前重建对比已标注来源。
- 后续任务先读本文件、再读当前实现和测试结果；维护“已完成/未完成/最后验证日期”，避免把历史建议当成已实现状态。
- 本文件是项目文件形式的持久记忆，由根目录 AGENTS.md 指向。新任务应在实际 E 盘工程目录中启动，或先显式读取这两个文件。
