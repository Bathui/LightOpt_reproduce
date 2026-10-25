# LightOpt 项目记忆

最后更新：2026-10-24（America/Los_Angeles）。状态描述是最后一次检查的记录，后续任务须核对当前文件再继续。

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

## 2026-10-23 历史验证：light.py 回归测试全部通过（最新状态见后续日期记录）

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

## 2026-10-24：build_target.py 调试与修复

- 用户手写了 `algo/build_target.py`，授权检查并修复。已保留其结构，修复命令行入口错误（移到模块顶层，补 sample_folder 参数）、输入有限性检查（or 改 and）、移除无用 sympy 导入、补 CUDA 不可用提示。
- 补充样本/灯光/模型源码 SHA256、模型名称，并在计算后重新核对，防止输入变化后保存不匹配目标。metadata.complete 仍在目标成功写入后设置。
- 新增 `tests/test_build_target.py`：18 项，修复前 10 通过、6 断言失败、2 错误；修复后 18 全通过。
- 联合原有 22 项灯光测试，实际执行 **40 项，40 通过，0 失败，0 错误，0 跳过**，包括实际命令行子进程、手算 .28125、暗区保留、NPZ/JSON 回读、输入/目标异常、缓存指纹/覆盖保护、no_grad、CPU/CUDA 数值一致性。错误分支有明确的 mock 注入，不将其表述为真实场景数值验收。
- 真实 `data/garden_samples_002` CUDA 运行已尝试：第一盏灯的色温被 load_lights 明确拒绝；未创建 `data/garden_target_validation_001`，未改场景或原始数据。因此目标生成工具已可用，但真实 Garden 目标缓存还没生成。
- `tmp/review/build-target-fixes.before.py` 是真实修改前快照；`tmp/review/build-target-fixes.diff` 是精确差异；`tmp/validation/build_target_validation_report.json` 记录本次结果与源码哈希。
- 模块 1 仍未完成：剩真实 Cookie/色温策略、Garden 目标生成、目标统计和照明可视化。

## 2026-10-24：批量关闭实验灯光色温和 Cookie 的编辑器菜单

- 已新增 `unity/Editor/LightOptLightTools.cs`，安装副本位于 Unity 工程 `Assets/Editor/LightOptLightTools.cs`；两份一致，并通过当前 Unity API 编译检查。
- 菜单：`Tools > LightOpt > Disable temperature and cookies`。
- 仅处理活动场景中启用/激活、Intensity>0、Range>0 的 Realtime Point/Spot，与默认采样选择一致；不改变灯光模式、启用状态、基础颜色、强度或其他场景。
- 将 useColorTemperature=false、cookie=null，记录 prefab 场景实例覆盖；支持一次 Ctrl+Z 撤销。仅标记场景待保存，不自动保存或修改 prefab 资产。
- 只是菜单已安装，尚未替用户执行此场景操作。用户需要停止 Play Mode、选择 Garden_LightOpt 为活动场景、点击菜单、保存后重新导出与采样。现有 lights.json 不会自动更新。
- 用户截图实际搜索字样为 `tLight`，正确类型筛选应为 `t:Light`；前者会匹配名字中的 Light，包含模型等对象。

## 2026-10-24：真实目标缓存验证和照明可视化已生成

- 用户已另行生成 `data/garden_export_004`、`data/garden_samples_004` 和 `data/garden_target_004`；本次核对缓存 metadata 指向 samples_004，样本/灯光/模型源码 SHA256 一致。
- 当前 samples_004/lights.json 的 13 盏灯已不再触发 Cookie/色温拒绝，能够进行实际前向计算。历史 samples_002 仍保持旧状态，不要混淆数据版本。
- 新增 `algo/validate_target.py`：缓存与输入哈希/完成标记/形状/dtype/有限值/非负值/法线/面积权重检查，显示专用 Reinhard+sRGB RGB 点云、亮度热图 PLY、亮度直方图、俯视/侧视投影、JSON 统计。可用 --recompute 重新计算并统计各灯的非零贡献样本和面积加权贡献。
- 实际运行命令：`conda run -n lightopt python -m algo.validate_target data/garden_target_004 --output data/garden_validation_004 --recompute --device cuda`。
- 输出 `target_rgb.ply`、`target_luminance.ply`、`diagnostics.png`、`validation_report.json`，全部在 garden_validation_004。已查看 PNG，图表和标签正常。
- 实际缓存含 86,114 条记录：82,753 条全零（96.10%），3,361 条非零；GPU 重算与 CPU 生成缓存通过 rtol=1e-4 / atol=1e-6 比较，最大绝对差 1.9073486328125e-6。
- 灯 1-6（小 Range Point）在现有样本上贡献为零。高零值比例是需检查的实验现象，不自动判定缓存错误；本轮仅有 13 盏实时灯，扩展球样本、Spot 锥外样本和双面背面等均可能为零。不代表全部 Unity 照明，不能把结果当作完整引擎画面。
- 原始 NPZ/JSON 只读，PLY 保留全部记录和法线；sample_index 对应原数组行。RGB/热图是显示映射，未覆盖 raw irradiance；投影最多绘制 50,000 点并优先保留亮点，不是遮挡正确的相机渲染。
- 默认 display scale 为正亮度的 p99；对比多个缓存时使用相同 --display-scale，避免分别自动曝光造成误判。双面点同坐标但法线/辐照度可能不同，不能只靠 PLY 深度显示判定某一侧受光。
- 新增 `tests/test_validate_target.py`，11 项测试覆盖实际重算、文件只读、哈希不符、错误目标、未完成缓存、数值篡改、全黑、旧模型、输出保护、显示映射。完整 tests 套件实际 **51 项全部通过，0 跳过**。
- 模块 1 的采样、目标缓存、数值统计和可视化产物现在已有完整闭环，数值/文件验证通过；尚不能宣称已验证与 Unity Shader/最终画面一致。接下来由用户查看 PLY/图像确认覆盖及参考照明是否满足实验意图，然后进入损失实现。
- 本次另确认工程已有 `.git`（用户新增），以后使用 Git 状态/差异；不要继续沿用历史“工程无 Git”的现状描述。保留用户已有修改，不自行提交。

## 2026-10-24：逐灯接口专项测试

- 用户已自行实现 `algo/light.py` 的 `evaluate_per_light()`，输出为 `[M,N,3]`。本轮只完善用户指定的 `tests/test.py`，未修改灯光模型或损失代码。
- `tests/test.py` 可从工程根目录直接运行：`conda run --no-capture-output -n lightopt python tests/test.py`。数值案例打印 expected/actual；使用独立手算结果，而非只检查逐灯求和自洽。
- 实际运行 15 项测试，15 通过、0 失败、0 错误、0 跳过；覆盖 Point 距离/法线/Range/背面/零强度、Spot 轴向/半影/锥外/反向、混合灯顺序与 RGB、逐灯切片及总和、空灯集合/空样本、非法灯型、强度手算梯度、灯间梯度隔离、Point/Spot 位置/强度/Range 有限差分 gradcheck，以及 CUDA 数值与 backward。
- 当前 `positions.new_zeros(0, positions.shape[0], 3)` 写法在实际环境的空灯集合测试中通过；不要仅凭静态判断声称该分支出错。
- 这些是合成样本上的解析模型验证，不代表真实 Garden 每灯贡献或 Unity Shader 对齐已验收。本轮未重跑全项目测试套件。
- 原测试脚本快照：`tmp/review/per-light-tests.before.py`；精确前后差异：`tmp/review/per-light-tests.diff`。tests 目录仍按用户既有 `.gitignore` 排除，不改变忽略规则。

## 下一步顺序

1. 查看 garden_validation_004 的 RGB/亮度 PLY 和 diagnostics.png，确认当前 13 盏实时灯对应的参考照明是用户希望优化的范围。96.10% 零值应结合逐灯覆盖检查；如调整灯集合，需重新导出/采样/生成目标/验证。
2. 模块 1 数据闭环及数值检查已完成，Unity 最终画面/Shader 对齐仍是独立工作，不能用自洽重算代替。
3. 继续原定模块 3 的重建、Overlap、Attraction 损失；Attraction 先核对论文公式与单点梯度方向。
4. 后续进入模块 4/5 的连续与离散优化循环，以及模块 6 多视角/性能评测，不扩展 ToD 直到静态流程稳定。

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
