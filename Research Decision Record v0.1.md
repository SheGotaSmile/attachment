# Research Decision Record v0.1

- 日期：2026-09-17
- 状态：研究范围冻结，待执行验证
- 基础仓库：InstinctLab，目标提交 ba28d3d
- 机器人：Unitree G1 Popsicle 资产
- 当前记录性质：研究决策，不是实验结果，不是已确认论文贡献

## 0. 冻结规则

在实验 E2 完成前，暂不新增以下模块：

- 深度相机或 LiDAR 输入；
- VAE、teacher-student 或世界模型；
- 新的 gate loss；
- gate smoothness 或 action smoothness 专用正则；
- 风险 critic、CVaR 或恢复专家；
- AMP、动作模仿或人类步态先验；
- 新的专家分工标签；
- 面向不同模型单独调节 reward、curriculum 或 domain randomization。

E2 的唯一目的，是判断在公平匹配条件下 MoE 是否值得继续投入。

本记录中的“继续研究”不等同于“方法有效”，只表示实验结果足以支持下一阶段机制研究。

## 1. 暂定论文题目

**Parameter-Matched Comparison of Single Policies and Mixture-of-Experts for Humanoid Locomotion on Spatially Varying Slopes**

中文工作题目：

**参数公平匹配下单一策略与多专家策略在空间变化坡度人形行走中的比较研究**

题目不预设 MoE 优于单一策略，也不声称完成视觉控制、真实部署或极限坡度行走。

## 2. 核心研究问题

在 Unitree G1 人形机器人上，当以下条件完全一致时：

- actor 观测；
- reward；
- critic；
- PPO 配置；
- 环境 transition 数量；
- 并行环境数；
- 动力学和观测随机化；
- 随机种子；
- actor 参数量和推理计算预算；

一个四专家 dense MoE actor 是否相较于参数量和计算量匹配的单一策略，在空间连续变化的有符号坡度上具有可重复的实际优势？

重点观察优势是否集中在：

- 坡度变化区域；
- 上坡到下坡或下坡到上坡的转换区域；
- 高坡度变化率区域；
- 低摩擦或模型扰动区域；

而不是只表现为更大网络带来的普遍容量提升。

## 3. 零假设 H0

在公平匹配条件下，四专家 MoE 与参数匹配的单一策略之间不存在具有实际意义的性能差异。

具体地说，MoE 在未见连续坡度测试集上的主要失败率指标不满足预设的最小实际改进阈值；即使存在差异，也可以由以下因素解释：

- 参数量；
- 推理计算量；
- critic 结构；
- 训练样本数量；
- 随机种子；
- 速度降低；
- reward 或环境随机化差异。

H0 不要求两个模型数值完全相同，而是拒绝把小于预设实际阈值的差异解释为 MoE 的有效优势。

## 4. 备择假设 H1

在相同观测、reward、critic、PPO、环境 transition 数量、随机化、并行环境数和随机种子的条件下，四专家 MoE 在未见连续坡度过渡测试集上相较于参数匹配单一策略具有可重复的实际优势。

H1 需要同时满足：

1. 主要失败率指标达到预设的相对改进阈值；
2. 优势在多个随机种子上方向一致；
3. 优势主要出现在坡度过渡或高变化率区域，而不是只出现在平地；
4. 实际前进速度没有因明显减速而换取成功率；
5. 优势不只来自 MoE critic；
6. 专家路由干预能够改变相应坡度区域的性能。

H1 是待验证假设，不是当前结论。

## 5. 任务定义

### 5.1 机器人

- 机器人：Unitree G1。
- 当前仓库实际资产：G1 Popsicle 版本。
- 当前审计得到的关节数：29 个 revolute joints。
- 动作：29 维关节位置目标。
- 动作语义：

  `q_target = q_default + scale * action`

- 控制周期：当前 Flat 配置为 0.02 s，即 50 Hz。
- 低层 actuator、PD 和物理执行由 Isaac Lab/外部仿真栈承担。
- 本研究第一阶段不修改底层 actuator，也不研究 torque policy。

上述资产、动作和控制周期需在实际环境创建后再次确认。

### 5.2 坡度定义

令 `s` 为沿机器人前进方向的水平路径坐标，`h(s)` 为地形高度。

有符号坡度角定义为：

`theta(s) = atan(dh(s) / ds)`

坡度空间变化率定义为：

`kappa(s) = d theta(s) / ds`

实现时应使用弧度计算，报告时转换为度每米：

`kappa_deg = kappa_rad * 180 / pi`

本研究区分：

- 坡度角 `theta`；
- 坡度变化率 `kappa`；
- 坡度符号；
- 坡度变化的空间过渡长度；
- 机器人速度；
- 摩擦系数。

不把 terrain difficulty 或坡度比直接当作角度报告。

### 5.3 坡度范围

#### 训练范围

训练地形限制为：

- `theta in [-30 deg, +30 deg]`
- 训练剖面包含固定坡和连续变化坡；
- 不向 actor 提供 terrain generator 的真实坡度标签；
- 训练时不包含 `+/-35 deg` 和 `+/-40 deg` 的目标测试样本。

建议训练坡度集合：

- 固定坡：`-30, -20, -10, 0, +10, +20, +30 deg`；
- 连续坡：平滑随机剖面；
- 最大坡度变化率：`|kappa| <= 20 deg/m`；
- 过渡长度：暂定 `4--8 m`。

#### 主要测试范围

主要测试范围为：

- `theta in [-35 deg, +35 deg]`；
- 固定坡：`-35, -25, -15, 0, +15, +25, +35 deg`；
- 连续变化坡；
- 至少包含 `-35 -> +35 deg` 和 `+35 -> -35 deg` 的符号转换；
- 使用训练中未出现的坡度变化率和过渡长度组合。

#### 极限探索范围

`+/-40 deg` 只作为探索性测试，不作为 v0.1 的主要统计结论。

极限测试至少需要单独记录：

- 摩擦系数；
- 实际速度；
- 是否从静止开始；
- 坡面长度；
- 上坡或下坡方向；
- 是否发生速度饱和；
- 是否发生 actuator saturation。

不能把一次通过 `+/-40 deg` 写成具有普遍性的能力结论。

### 5.4 坡度空间变化率

v0.1 采用以下预注册范围：

- 训练：`|kappa| <= 20 deg/m`；
- 主要测试：加入训练中未出现的 `|kappa| = 25 deg/m`；
- 扩展测试：`|kappa| = 30 deg/m`；
- 连续剖面必须经过平滑处理；
- 不使用数值不连续的坡度跳变作为主要测试。

暂不声称上述变化率具有生物力学或实际道路代表性。它们只是用于构造受控实验变量，最终数值需根据 G1 尺寸、地形长度和仿真稳定性检查。

### 5.5 速度命令

主要实验固定：

- `vx = 0.5 m/s`
- `vy = 0`
- `yaw_rate = 0`

次要速度测试：

- `vx = 0.3 m/s`
- `vx = 0.5 m/s`
- `vx = 0.7 m/s`

所有模型使用完全相同的 command distribution。

v0.1 不研究主动速度规划。若策略通过主动降速获得更高成功率，必须同时报告实际速度，不能只报告成功率。

### 5.6 摩擦系数

初始训练协议暂定为：

- 训练：`mu ~ Uniform[0.8, 1.1]`；
- 同一 episode 内固定；
- 测试主集：`mu in {0.8, 0.95, 1.1}`；
- 扩展测试：`mu = 0.7`；
- `mu = 0.7` 和 `+/-40 deg` 不作为同一主要结论的依据。

上述范围是实验协议暂定值，不是当前仓库已经验证的运行配置。必须确认 Isaac Lab 中实际 material friction 的语义、组合规则和随机化生命周期。

### 5.7 训练地形

训练地形由以下两类组成：

1. 固定有限长度坡面；
2. 沿前进方向连续变化的平滑坡度剖面。

训练剖面应独立随机生成，并保存：

- terrain seed；
- `theta(s)`；
- `kappa(s)`；
- 坡面长度；
- 过渡长度；
- 摩擦系数；
- 地形类别；
- 机器人初始位置。

训练和测试必须按剖面生成族、参数区间和随机种子分离，不能只随机切分同一条地形。

### 5.8 测试地形

测试集包含：

- 训练范围内固定坡；
- 未见的 `+/-35 deg` 固定坡；
- 未见变化率；
- 未见过渡长度；
- 上坡到下坡；
- 下坡到上坡；
- 缓变坡；
- 快速但连续的坡度转换；
- 不同摩擦系数组合；
- 0 度和轻微起伏地形作为退化检查。

测试集由固定清单生成，训练过程中不可读取。

## 6. 信息边界

### 6.1 训练时 actor 可获得的信息

v0.1 主实验中，actor 只接收：

- 当前本体观测；
- 当前速度命令；
- 上一步原始 action。

第一版保持与 InstinctLab Flat 任务一致，不加入：

- 深度图；
- LiDAR；
- elevation map；
- true slope；
- true slope variation rate；
- terrain generator metadata；
- future terrain profile；
- true friction；
- contact force oracle。

当前 Flat policy 的静态输入维度为 96，但必须通过运行时 `obs_format` 再次确认。

### 6.2 测试时 actor 可获得的信息

测试时 actor 获得的信息与训练时一致：

- 本体观测；
- command；
- 上一步 action。

测试 evaluator 可以知道地形真值，但该信息只能用于：

- 计算坡度和变化率；
- 划分坡顶、坡底和转换窗口；
- 统计滑移、冲击和失败位置；
- 生成诊断图。

evaluator 的地形真值不能传入 actor。

### 6.3 仅用于 oracle ablation 的特权信息

oracle ablation 可以使用：

- 当前真实坡度 `theta(s)`；
- 当前真实坡度变化率 `kappa(s)`；
- 前方有限距离的真实坡度预瞄；
- terrain heightfield 或 mesh 真值；
- 真实摩擦系数；
- 真实接触力和接触状态。

oracle ablation 的作用是区分：

- 控制架构能力；
- 感知信息限制；
- 本体历史适应能力。

oracle policy 不属于主要方法，也不能用于证明实际部署可行。

### 6.4 视觉和 LiDAR 范围

v0.1 不回答深度相机或 LiDAR 的最终必要性。

视觉/LiDAR 只在 MoE 去留实验完成后进入后续实验，并且需要单独报告：

- 传感器输入；
- 延迟；
- 噪声；
- dropout；
- 坐标系；
- history；
- 计算开销；
- 仿真与真实传感器差异。

## 7. 模型定义

### 7.1 S-small

S-small 是当前 Flat 任务规模的单一 MLP policy：

- 输入：与主实验完全相同；
- actor：`96 -> 256 -> 128 -> 128 -> 29`；
- 激活：ELU；
- action distribution：沿用当前 PPO/Gaussian 实现；
- critic：统一的公共 critic `C0`；
- 不包含专家、门控、teacher、蒸馏或辅助损失。

S-small 用于确认任务是否能够被普通小型策略学习，不作为 MoE 公平比较的唯一对手。

### 7.2 S-match

S-match 是参数量和 actor 推理计算量与 M4 匹配的单一 MLP。

定义方式：

1. 先实例化 M4；
2. 统计 M4 actor experts、gate 和共享输入处理部分的可训练参数量；
3. 统计 actor forward 的 FLOPs 或至少统计线性层乘加量；
4. 在单一 MLP 的候选隐藏层宽度中选择与 M4 最接近的结构；
5. 参数量和推理计算量误差目标为不超过 5%；若不能同时满足，必须报告两者并使用预先确定的优先级；
6. S-match 只扩大 actor，不改变公共 critic `C0`；
7. S-match 不使用任何 MoE gate 或 expert loss。

S-match 的数值隐藏层宽度必须在代码运行时从实际 M4 配置计算后冻结。不能根据结果重新选择宽度。

### 7.3 M4

M4 是 v0.1 的主要 MoE 模型：

- actor 包含 4 个结构相同的 experts；
- 每个 expert 接收相同的 actor observation；
- gate 接收相同的 observation；
- gate 输出 4 维 softmax 权重；
- actor 输出为 4 个 expert action 的加权和；
- 使用 dense mixture；
- 每一步计算全部 4 个 experts；
- 不使用 hard routing；
- 不使用 top-k；
- 不使用坡度标签；
- 不使用专家职责监督；
- 不使用 gate smoothness；
- 不使用 expert diversity loss；
- critic 使用公共 critic `C0`；
- PPO 只使用标准 PPO 损失。

M4 的 gate 和专家结构必须以外部 `instinct_rl` 实际源码为准。当前审计确认已有 dense MoE 实现，但未确认目标任务运行时的完整张量形状。

### 7.4 actor-only MoE

“actor-only MoE”是结构分类，不是额外模型名称：

> 只有 actor 使用 MoE，critic 不使用专家分解。

在本记录中，actor-only MoE 的具体实例就是 M4。

它的目的，是把 actor 专家结构与 critic 专家结构分开，避免把多 critic 容量误认为 actor MoE 的作用。

### 7.5 shared-critic MoE

shared-critic MoE 是 M4 的具体训练配置：

- actor：4 experts + dense gate；
- critic：一个公共 critic `C0`；
- 所有模型使用同一个 `C0`；
- `C0` 的输入、层数、隐藏宽度、归一化方式和参数量固定；
- `C0` 不读取专家输出；
- `C0` 不接收额外 terrain oracle。

因此，v0.1 中：

- actor-only MoE；
- shared-critic MoE；
- M4；

指的是同一个公平主实验模型，不重复统计为三个独立方法。

### 7.6 M4-full

M4-full 指当前外部 `instinct_rl` 中可能存在的 actor-MoE + critic-MoE 配置。

它可以作为诊断基线，但不用于主要 H0/H1 比较，因为其 critic 容量和结构不同。

如果报告 M4-full，必须单独标注：

- actor expert 数量；
- critic expert 数量；
- gate 参数；
- 总参数量；
- FLOPs；
- 是否与主实验共享 critic。

## 8. 公平性约束

S-small、S-match 和 M4 主实验必须保持一致：

- actor 输入；
- action 维度；
- observation normalization；
- reward term 列表；
- reward 权重；
- termination/reset；
- critic `C0`；
- PPO 实现；
- rollout length；
- discount factor；
- GAE；
- learning rate；
- optimizer；
- entropy 设置；
- total environment transitions；
- parallel environment count；
- terrain distribution；
- friction distribution；
- dynamics randomization；
- observation noise；
- actuator delay；
- episode timeout；
- evaluation seeds；
- checkpoint selection rule。

禁止：

- 为 M4 单独调整 reward；
- 为 S-match 单独增加 curriculum；
- 为 M4 使用不同 critic；
- 用 M4 更长训练时间；
- 只选择 M4 最好的 checkpoint；
- 用不同测试地形；
- 用不同实际速度作为成功率比较依据。

## 9. Reward 和 critic 约束

所有主实验模型共享同一 reward。

reward 只允许包含已经预先确定的通用项，例如：

- 速度跟踪；
- 相对局部支撑面的姿态稳定；
- 足端滑移；
- 足端接触相关项；
- 动作变化率；
- 关节偏离；
- 加速度；
- 力矩或力矩代理；
- 非法接触和跌倒终止。

当前 Flat 配置中的 world-horizontal orientation penalty 必须先审查。若它会把顺坡站立错误惩罚为姿态偏差，则该项不能直接用于坡地实验。

任何坡面姿态 reward 的修改必须：

- 在所有模型中一致；
- 在首次模型训练前冻结；
- 记录到生成的 `env.yaml`；
- 不根据模型结果重新调节。

公共 critic `C0` 的输入只使用主实验定义的 critic observation。主实验不向 critic 提供 true slope、true friction 或 future terrain。

## 10. PPO、训练预算和随机化

### 10.1 PPO

沿用外部 `instinct_rl` 的 PPO 实现和当前任务配置。

v0.1 不加入：

- auxiliary expert loss；
- load balancing loss；
- contrastive loss；
- distillation loss；
- gate regularization；
- risk objective；
- multi-critic loss。

rollout length 暂定为当前配置中的 24 个 environment steps，必须从运行时 agent 配置确认。

### 10.2 训练预算

内部 MoE 去留筛选暂定：

- 每个模型每个 seed：20,000,000 environment transitions；
- 训练 transition 数量而不是 iteration 数量作为预算；
- 每次 PPO rollout 的长度一致；
- 若环境吞吐不同，不得因此增加某个模型的训练 transition；
- 不以 wall-clock 时间作为主要训练预算；
- 如需扩大预算，所有模型同时扩大，并在扩大前记录原因。

20M 只是筛选预算，不足以证明最终收敛或论文性能。

### 10.3 并行环境

初始筛选：

- `num_envs = 256`；
- 单 GPU；
- 暂不使用多节点或 DDP；
- 所有模型相同；
- 记录显存、step throughput 和实际 transition 数量。

若显存不足，必须同时降低所有模型的环境数；不能只降低 MoE 或单策略的环境数。

### 10.4 随机种子

筛选使用：

- `42`
- `43`
- `44`

每个 seed 从独立初始化开始。

正式论文实验至少扩展到 5 个 seed。v0.1 的 3 个 seed 只用于 go/no-go，不作为最终统计结论。

### 10.5 动力学和观测随机化

所有模型使用同一套随机化：

- 质量；
- 质心；
- 材质；
- PD gains；
- joint zero offset；
- 初始状态；
- observation noise；
- actuator delay；
- episode terrain seed；
- friction。

具体范围以实际生成的配置和运行时实现为准。若某一随机化项无法运行，必须先修复或从所有模型中同时移除，不能只对单一模型改变。

## 11. 指标

### 11.1 主指标

主要测试集上的：

**每 100 m 行走距离的失败次数**

失败包括：

- 摔倒；
- 非法接触终止；
- 无法继续前进；
- 速度保持失败并达到预设终止条件。

主指标必须分别统计：

- 固定坡；
- 缓变连续坡；
- 快速连续坡；
- 上坡到下坡；
- 下坡到上坡；
- 坡顶窗口；
- 坡底窗口；
- 不同摩擦系数。

### 11.2 次指标

- episode success rate；
- 最大连续行走距离；
- 实际前进速度；
- 速度跟踪误差；
- 机身姿态相对局部坡面法向的误差；
- 足端滑移距离；
- 足端滑移比例；
- 接触冲量；
- 接触力变化；
- 关节力矩；
- 力矩平方和；
- 机械功或明确说明定义的能耗代理；
- action rate；
- 关节加速度；
- 恢复时间；
- 扰动后的恢复成功率；
- 推理延迟；
- actor 参数量；
- actor forward FLOPs；
- GPU 显存和吞吐。

上坡和下坡的能耗必须分开报告。负机械功不能直接当作可回收电能。

### 11.3 MoE 诊断指标

- gate entropy；
- 每个 expert 的平均使用率；
- 每个 terrain profile 的 gate trajectory；
- gate 与 `theta(s)` 的关系；
- gate 与 `kappa(s)` 的关系；
- gate 与 contact phase 的关系；
- expert action pairwise distance；
- expert action variance；
- gate switch count；
- gate dwell time；
- route-induced action change；
- 单 expert 禁用后的性能变化；
- 均匀 gate 后的性能变化；
- gate 随机交换后的性能变化。

gate 与坡度的相关性只能作为诊断，不能单独证明专家职责分化。

## 12. 失败判据和继续研究判据

### 12.1 实验无效判据

出现以下任一情况，不能作 MoE 结论：

- 0 度平地 S-small 无法稳定学习；
- 任务在 Isaac Sim 中无法稳定创建；
- 观测 shape 在不同 reset/env_id 下不一致；
- 训练出现无法解释的 NaN 或 reset 错误；
- 模型没有使用同一 reward 或 critic；
- checkpoint 选择规则不一致；
- 训练 transition 数量不一致；
- 测试地形没有固定；
- 3 个 seed 中存在未记录的人工调参。

### 12.2 MoE 暂停判据

满足以下任一情况，暂停继续增加 MoE 模块：

- M4 不优于 S-match；
- M4 只优于 S-small，但不优于参数匹配的 S-match；
- M4 优势只出现在训练地形；
- M4 通过明显降低实际速度获得更高成功率；
- M4 优势主要来自 M4-full 的 expertized critic；
- M4 在不同 seed 上方向不稳定；
- 均匀 gate、随机 gate 或单专家干预几乎不改变结果；
- 所有 expert 输出高度相似；
- 结果差异小于 seed 间变化；
- M4 的计算或延迟明显超出 S-match 的预算。

此时研究重点转向参数匹配的单一条件策略、坡度物理先验或感知必要性边界，而不是继续添加 MoE 模块。

### 12.3 继续研究判据

只有当以下条件同时满足时，才进入 E3：

1. M4 在 3 个 seed 上相较 S-match 方向一致；
2. 主要测试集的每100m失败率相对降低至少20%，或成功率提高至少5个百分点；
3. 该差异集中在未见连续坡度过渡或高变化率区域；
4. 实际速度下降不超过10%；
5. 公共 critic 条件下仍然存在差异；
6. gate/专家干预能够消除或改变相应区域的收益；
7. 结果不依赖某一个单独地形 seed。

这些是预设的继续研究门槛，不是对结果的预判。

## 13. 当前禁止写进论文的结论

在 E2 完成前，禁止写：

- “MoE 优于单一策略”；
- “MoE 是连续坡度行走所必需的”；
- “专家分别学会了上坡、下坡和过渡”；
- “该方法解决了未知地形行走”；
- “该方法支持深度相机或 LiDAR 部署”；
- “该方法完成 sim-to-real”；
- “能够稳定通过 +/-35 deg”；
- “能够完成 +/-40 deg”；
- “当前是首个连续坡度人形行走方法”；
- “InstinctLab 已经提供了完整控制算法”；
- “现有 InstinctLab MoE 配置就是本文方法”；
- “gate 权重聚类证明了专家功能分工”；
- “平均 reward 提升代表稳定性提升”；
- “ONNX 导出代表真实机器人闭环可用”。

可以写的暂定表述只有：

> 本研究拟通过参数量和训练预算匹配的受控实验，检验 MoE 在空间变化坡度人形行走中的实际作用。

## 14. 后续实验编号和顺序

### E0：仓库和环境运行验证

目标：

- 使用 Python 3.11 环境；
- 创建 Flat 环境；
- 确认任务注册；
- 确认实际 obs/action shape；
- 确认 reset、done 和 timeout；
- 完成至少 2 次 PPO update；
- 生成可加载 checkpoint。

E0 失败时不进入模型比较。

### E1：单策略任务基线

模型：

- S-small。

测试：

- 平地；
- 固定坡；
- 连续坡度；
- 训练范围内和目标测试范围。

目标：

- 确认任务难度；
- 确认 reward 没有错误惩罚顺坡姿态；
- 确认失败指标和 evaluator 可用。

### E2：公平 MoE 去留实验

模型：

- S-small；
- S-match；
- M4，即 actor-only/shared-critic MoE。

条件：

- 相同 critic；
- 相同 PPO；
- 相同 transition 数；
- 相同 randomization；
- seed 42/43/44。

E2 是当前冻结主线的核心实验。

### E3：MoE 机制诊断

只有 E2 满足继续研究判据后执行：

- M4-full；
- 单专家启用；
- 均匀 gate；
- 随机 gate；
- gate 冻结；
- 专家禁用；
- 专家输出和 gate trajectory 分析。

目标是判断收益来自 actor routing、critic 容量还是普通模型容量。

### E4：oracle slope ablation

加入 true slope、true slope variation rate 和有限预瞄。

目标：

- 区分控制架构限制与感知限制；
- 判断坡度信息是否足以改变模型排序；
- 不把 oracle 结果作为部署结果。

### E5：外部感知扩展

按以下顺序选择一种模态：

1. 局部高度扫描；
2. 深度相机；
3. LiDAR。

所有模态需要重新记录：

- 输入维度；
- 历史；
- 延迟；
- 噪声；
- dropout；
- 坐标；
- 计算量；
- 训练/测试分布。

### E6：鲁棒性扩展

加入：

- 低摩擦；
- actuator delay；
- observation dropout；
- 外部冲量；
- 持续外力；
- 足端扰动；
- 载荷和质心变化。

此阶段才评价恢复能力、滑移和冲击尾部指标。

### E7：硬件在环和真实机器人

只有仿真结果、输入协议、动作尺度、normalizer、历史填充和实时延迟全部核实后执行。

InstinctLab 本身不包含完整硬件闭环；需要单独审查外部 `instinct_onboard`。

## 15. 待 Codex 从仓库验证的事实

以下内容当前不能由静态审计替代运行验证：

1. 目标坡度任务能否在 Isaac Sim 中正常创建；
2. 现有 terrain generator 能否生成有符号、空间连续且可记录 `theta(s)`/`kappa(s)` 的坡度剖面；
3. 现有反向金字塔坡度参数的实际几何角度；
4. terrain mesh、heightfield 和环境原点的坐标约定；
5. G1 运行时关节排序是否与 29 维动作顺序一致；
6. Flat 任务运行时 actor/critic 的真实 observation format；
7. 96 维 policy 和 99 维 critic 静态推导是否与运行时一致；
8. 外部 `instinct_rl` 当前 MoE 配置的 expert 数、层数、参数量和 FLOPs；
9. dense MoE 的 gate 是否确实对全部 expert 输出加权；
10. 当前 actor 是否存在 action clipping；
11. action scale、default joint position 和 actuator limit 的完整运行语义；
12. slope-relative orientation reward 的实现位置和可行性；
13. 当前 reward 中 world-horizontal orientation 项在坡面上的行为；
14. friction randomization 在 Isaac Lab 中的实际范围和生命周期；
15. mass、CoM、PD、joint offset、delay 和 observation noise 的有效范围；
16. 256 个并行环境下的显存和吞吐；
17. 不同 env_id 独立 reset 时传感器和 terrain state 是否正确；
18. episode evaluator 是否按单个 env_id 正确统计；
19. 失败位置能否映射回 `theta(s)` 和 `kappa(s)`；
20. checkpoint、normalizer 和配置是否能够被同一模型正确恢复；
21. 训练后 S-small、S-match 和 M4 是否真正使用同一 critic；
22. 训练和评估时 command、observation normalization 和 action transform 是否一致；
23. 当前代码中已知的 subset sensor noise、AsyncCircularBuffer、terrain metadata 和评估统计问题是否会影响本实验；
24. 是否存在可用的完整 checkpoint；当前审计未找到可直接回放的 `.pt` 文件；
25. 目标实验是否需要修改外部 `instinct_rl`，而不仅是 InstinctLab 配置。

## 16. 待验证的候选论文贡献

只有在实验完成后，才从以下候选中选择实际成立的贡献：

1. 一个用于比较单策略与 MoE 的空间连续有符号坡度测试协议；
2. 对坡度幅值、空间变化率、坡度符号转换和失败位置的系统分析；
3. 关于 MoE 在何种坡度过渡条件下有效或无效的受控实证结论；
4. 参数量和训练预算公平匹配下的模型比较；
5. 对 actor MoE、critic MoE、专家路由和专家分工的干预分析；
6. 感知预瞄对不同坡度变化率和速度条件的必要性分析；
7. 在仿真中对稳定性、滑移、冲击、能耗和速度之间关系的统一报告。

当前不预注册以下内容为贡献：

- 使用 MoE 本身；
- 增加专家数量；
- 使用深度相机或 LiDAR；
- 使用 domain randomization；
- 使用 PPO；
- 使用 ONNX；
- 训练出能够行走的 G1；
- 单次通过 35 度或 40 度坡面；
- 专家权重可视化；
- 将已有 InstinctLab MoE 配置重新命名。
