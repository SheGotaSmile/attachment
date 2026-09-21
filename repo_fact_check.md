# InstinctLab 研究决策事实核查

核查日期：2026-09-17。范围：当前工作树与本机实际外部依赖。本文只核实事实，不实现算法，不添加 MoE、相机、LiDAR，不修改已有审计或研究记录。

## 0. 证据、现场与结论边界

先执行 `git status --short --untracked-files=all`，再阅读已有审计和研究记录，并使用 `rg` 定位实际配置、函数体和调用方。初始仅有两个未跟踪用户文件：`docs/repo_audit.md` 和 `docs/Research Decision Record v0.1.md`；没有已跟踪源码变更。此次仅新增本文及 `docs/experiment_protocol_v0.1.md`。

现场版本记录（来自 `git rev-parse HEAD`，版本输出本身没有源码行号）：

| 工作树 | HEAD | 初始状态 |
| --- | --- | --- |
| `/home/xiexuhui/InstinctLab` | `ba28d3d2655b15a19b729476a630937a19610a3b` | 上述两个未跟踪文档 |
| `/home/xiexuhui/instinct_rl` | `ba45ed231ebbf0a4099cd31d607e2886814fd165` | clean |
| `/home/xiexuhui/IsaacLab-Instinct` | `f73c33173801f5f8afea4142482e47b7710c2b75` | clean |

两个原文件的初始 SHA-256 分别为 `dda971ff9f02d7c40cf165e300bd134c41c7cb4cc2cbffdddeac7b69e0c3ea4b`、`83ace85c303689a4aba478380bd60c78b995576cad787a140eb970bb770eac42`。它们只用于核查结束时确认未覆盖用户文件。

本轮完成静态源码核查、URDF XML 解析和现有网络 CPU 前向；没有启动 Isaac Sim、训练、加载训练 checkpoint 或执行机器人部署。以下区分“源码确认”“CPU 局部验证”“未找到”“待运行验证”。“未找到”限定于第 14 节检索范围，并附最近的实际实现位置；不存在的功能没有可引用的实现行号，不虚构行号。

研究记录本身说明“研究决策，不是实验结果”，因此下文多数差异是实现前置条件，而不是把拟议实验误判成虚假结果。依据：[研究记录:3](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:3>)、[研究记录:221](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:221>)。已有审计只作定位线索，事实以链接的函数体为准。

结论：当前可作为 Flat PPO 基线和扩展组件库；不能直接作为研究记录的完整 E2 实验。关键缺口是 actor-only/common-critic M4、受控且可记录的连续坡度实验地形、独立逐环境评估和复杂度匹配工具。证据分别见第 4、8、10、11 节。当前 Parkour 的深度历史、AMP、MoE critic 也违反主实验的信息及算法冻结边界：[Parkour 策略:32](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:32)、[Parkour 算法:44](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:44)、[研究记录:11](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:11>)。

## 1. G1 自由度、动作维度与控制频率

**源码确认；URDF 已解析；仿真运行时待确认。** Flat 在 `G1_CFG` 中选择 Popsicle 资产，scene 实际使用该配置，不是根据文件名猜测：[flat_env_cfg.py:23](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:23)、[同文件:47](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:47)。

| 项目 | 实际结果 | 证据 |
| --- | --- | --- |
| 关节自由度 | XML 统计 29 个 revolute、10 个 fixed；双腿各 6、腰 3、双臂各 7；固定手掌没有手指动作自由度 | [URDF 左腿:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf:65)、[腰:452](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf:452)、[左臂:618](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf:618)、[右臂:822](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf:822) |
| 浮动基座 | `fix_base=False`；29 是受控关节数，不包含基座 6 个刚体自由度；不是 35 维动作 | [unitree_g1.py:536](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:536) |
| 动作 | 全部关节的 29 维位置偏移指令，非力矩 policy | [flat_env_cfg.py:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:65)、外部 [joint_actions.py:197](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:197) |
| 频率 | `dt=0.005 s`，physics 200 Hz；`decimation=4`，控制 50 Hz；默认 episode 20 s，即 1000 个控制步 | [flat_env_cfg.py:357](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:357)、外部 [物理循环:182](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:182) |
| 实际执行器 | Popsicle 绑定 `beyondmimic_g1_29dof_actuators`，使用 `ImplicitActuatorCfg`；不能把同文件另一个 delayed 字典当作 Flat 已启用 | [unitree_g1.py:269](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:269)、[实际绑定:573](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:573)、[另一套配置:391](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:391) |

与研究记录 5.1 的 29 维、50 Hz 一致，但只确认当前 Flat 默认值。运行时仍须保存 `robot.joint_names`、`num_joints`、action term 的解析后 joint IDs/scale/offset，不能把 URDF 文本顺序直接当作 policy 顺序。依据：外部 [JointAction 关节解析:65](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:65)、[研究记录:108](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:108>)。

## 2. 当前观测空间和动作空间

**Flat 已确认；不同任务不能混用。** Flat 注册绑定 `G1FlatEnvCfg` 和 `G1FlatPPORunnerCfg`：[任务注册:6](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py:6)。令 `N` 为并行环境数。

| Flat 分量 | policy | critic | 定义 |
| --- | --- | --- | --- |
| base linear velocity | 无 | `(N,3)` | [Flat:115](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:115) |
| base angular velocity | `(N,3)` | `(N,3)` | [Flat:98](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:98) |
| projected gravity | `(N,3)` | `(N,3)` | [Flat:99](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:99) |
| command `(vx,vy,wz)` | `(N,3)` | `(N,3)` | [Flat:103](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:103)、[critic:120](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:120) |
| relative joint position | `(N,29)` | `(N,29)` | [Flat:104](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:104) |
| joint velocity | `(N,29)` | `(N,29)` | [Flat:105](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:105) |
| previous raw action | `(N,29)` | `(N,29)` | [Flat:106](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:106)、[critic:123](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:123) |
| wrapper 拼接后 | `(N,96)` | `(N,99)` | [vecenv_wrapper.py:208](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:208)、[展平拼接:222](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:222) |

没有配置观测历史，actor 有噪声、critic 无噪声；没有坡度、摩擦真值或外部感知输入：[Flat policy:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:97)、[Flat critic:114](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:114)。历史默认值为 0，见外部 [manager_term_cfg.py:184](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/managers/manager_term_cfg.py:184)。

动作语义是 `q_target[j] = q_default[j] + scale[j] * a[j]`。声明中的 `scale=0.5` 被 `__post_init__` 覆盖为逐关节 `0.25 * effort_limit_sim / stiffness`，不能报告为统一 0.5：[Flat:364](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:364)、[scale 计算:523](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:523)、外部 [动作处理:169](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:169)。网络均值末层为线性、训练 Gaussian 采样，Flat 未设动作 clip，因此不能称为天然 `[-1,1]^29`：[actor_critic.py:112](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:112)、[采样:159](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:159)。

普通网络静态结构为 actor `96->256->128->128->29`、critic `99->256->128->128->1`、ELU。CPU 实际调用现有类得到 `(4,96)->(4,29)`、`(4,99)->(4,1)`；不是运行时环境 shape 验收：[Flat PolicyCfg:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:12)、外部 [evaluate:174](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:174)。

## 3. 奖励函数及姿态参考系

**Flat 奖励存在；局部坡面姿态参考未接入。** 以下为当前权重，不是建议的新权重。

| 项目 | weight | 代码 |
| --- | --- | --- |
| termination penalty | -200 | [Flat:143](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:143) |
| xy velocity / yaw angular velocity tracking | 各 +1，`std=0.5` | [Flat:144](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:144) |
| feet air time / feet slide | +1 / -0.1 | [Flat:154](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:154) |
| flat orientation / stand still / ankle position limits | -1 / -0.8 / -1 | [Flat:172](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:172) |
| hip / arm / torso / knee position deviation | -0.1 / -0.1 / -0.1 / -0.05 | [Flat:179](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:179) |
| vertical velocity / action rate / joint acceleration / joint torque | -0.1 / -0.05 / -2e-7 / -4e-6 | [Flat:212](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:212) |

参考系直接由运算确定：

- xy 跟踪先用 `yaw_quat(root_quat_w)` 把 world 线速度旋到重力对齐的 yaw frame，再取 xy；没有投影到坡面切向：[locomotion rewards.py:47](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/mdp/rewards.py:47)。
- yaw 跟踪使用 `root_ang_vel_w[:,2]`，是 world Z 角速度：[同文件:60](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/mdp/rewards.py:60)。
- `flat_orientation_l2 = sum(projected_gravity_b[:,:2]^2)`，鼓励机身相对重力竖直，不读取地形法向；纯 pitch 为 `alpha` 时该项约为 `sin(alpha)^2`。它不是 slope-relative orientation reward：[外部 rewards.py:91](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/rewards.py:91)。
- `contact_slide` 使用接触力历史门限和足端 world xy 速度，不是完整坡面切向滑移距离：[regularizations.py:780](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/rewards/regularizations.py:780)。

研究记录已要求审查此问题，不能把它直接视为顺坡姿态参考已实现：[研究记录:477](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:477>)、[研究记录:487](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:487>)。建议在 E1 前共同冻结“相对重力直立”还是“相对支撑面姿态”的目标、采样位置和期望倾角；不能假定坡面法向对齐一定是正确的人形姿态。本轮不改 reward。

## 4. 地形生成器是否支持连续坡度

**部分支持几何组件；研究所需完整协议未找到。** “坡度参数能连续取值”“空间高度连续”“坡度角本身连续且变化率可控”是不同条件。

| 实际实现 | 能确认什么 | 不能据此声称什么 |
| --- | --- | --- |
| Flat `terrain_type="plane"` | 当前主基线是平地 | 已有坡度任务。证据：[Flat:35](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:35) |
| Parkour active map 的反向金字塔坡 | 占比 0.10，`slope_range=(0,0.7)`，difficulty 对坡度比插值；另混有起伏、沟隙、台阶、箱体 | 整条道路恒定角度，或满足指定 `theta(s), kappa(s)`。证据：[Parkour:45](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:45)、[active slope:261](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:261)、[生成:78](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:78) |
| `perlin_slope_terrain` | 角度参数转 tan，两个线性坡段与平台；`up_down=False` 翻转高度 | 坡段连接处角度连续；代码未做连接平滑。证据：[hf_terrains.py:991](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:991)、[赋值:1033](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:1033) |
| `perlin_wave_terrain` | 已有 sin/cos 波形高度，理想公式有变化坡度；最终量化为 heightfield/mesh | 按研究记录角度、最大变化率、剖面族和种子分离的实验。证据：[hf_terrains.py:339](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:339)、[返回:376](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:376) |
| `perlin_tilted_ramp_terrain` | degree 参数生成横向斜面，可沿另一轴交替 | 沿机器人路径平滑连续的有符号纵坡。证据：[hf_terrains.py:910](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:910)、[分段交替:918](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:918) |

`rg` 对后三种配置仅找到定义、导出与函数注解，未找到当前任务中的实例化；当前 Parkour 实际绑定的是另一套 `ROUGH_TERRAINS_CFG`：[SceneCfg:290](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:290)。因此应写“有可复用地形组件”，不能写“空间变化坡度研究环境已经就绪”。

按 episode 保存 `theta(s)`、`kappa(s)`、路径方向、独立 train/test manifest 的完整链路未找到。已有 `FiledTerrainGenerator` 只记录子地形 cfg/difficulty/seed，且 `get_subterrain_cfg()` 使用 `torch.Tensor` 却没有导入 torch，调用此查询方法有 NameError 风险：[terrain_generator.py:21](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_generator.py:21)、[查询:42](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_generator.py:42)。这不是 active Parkour 生成路径已失败的证据。

## 5. 坡度角、坐标系和正负方向

**仓库没有统一的研究坡度定义；必须按生成函数区分。**

| 来源 | 定义、坐标及符号 | 证据 |
| --- | --- | --- |
| heightfield 坐标 | 数组第一轴对应 local X，第二轴对应 local Y；高度乘 `vertical_scale` 成 Z，网格再平移到 tile/world | 外部 [height_field/utils.py:123](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/terrains/height_field/utils.py:123)、[顶点:155](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/terrains/height_field/utils.py:155)、[tile 变换:333](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/terrains/terrain_generator.py:333) |
| pyramid `slope_range` | 无量纲升高/水平距离比，不是 deg/rad；inverted 把参数取负使中心下凹。实际 `hf=height_max*xx*yy` 再 clip，沿不同位置/方向局部导数不同；靠近中心与离开中心正负相反 | [hf_terrains.py:78](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:78)、[网格公式:94](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:94) |
| `slope_angle` | 输入 degree，`deg2rad` 后 tan；高度沿 local +Y 先上后下，`up_down=False` 反转；这是几何方向，不自动等于机器人 +X 前进方向 | [配置:175](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains_cfg.py:175)、[实现:1013](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:1013)、[坡段:1033](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:1033) |
| `tilt_angle` | 输入 degree；斜率作用于 local X；左右两侧相反，沿 Y 可切换左右倾斜 | [配置:160](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains_cfg.py:160)、[实现:924](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:924) |
| 研究定义 | `theta=atan(dh/ds)`，沿约定水平前进路径增高为正，降低为负；`kappa=dtheta/ds`。把 world/tile/path/body 坐标联系起来的标签接口未找到 | [研究记录:110](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:110>)；最近实现是上面的局部 heightfield 公式 |

`atan(0.7)` 约 34.99 deg 只是坡度比到角度的换算，不说明 active map 已覆盖 `[-35,+35] deg` 的实验样本。`slope_threshold=1.0` 则是 heightfield-to-mesh 的陡面处理阈值，不是任务角度上限：[Parkour:53](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:53)、外部 [utils.py:132](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/terrains/height_field/utils.py:132)。

建议未来固定路径坐标，而不是每一步随机器人朝向重定义正负；否则机器人转头会改变同一地形的标签。还应验证量化和三角面插值后的实际法向，不只报告生成器请求角度。此为协议建议，不是现成实现。

## 6. 是否存在高度扫描器

**存在于其他任务；Flat 未找到。**

| 任务 | 已存在的扫描和用途 | 证据 |
| --- | --- | --- |
| Flat | scene 只有 contact sensor，没有 height scanner；policy/critic 没有 height_scan | [Flat:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:34)、[observations:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:97) |
| Parkour | 左右脚 RayCaster，yaw 对齐，grid `0.12 x 0.0 m`、resolution `0.12 m`；用于 `feet_at_plane` reward，不可自动计为 actor 输入 | [scanner:318](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:318)、[reward:731](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:731) |
| Perceptive | torso RayCaster，yaw 对齐，grid `1.6 x 1.0 m`、resolution `0.1 m`；17x11=187 射线，critic 的 scan 为 `(N,187)`，policy 的高度项被注释 | [scanner:109](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:109)、[policy 注释:237](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:237)、[critic:301](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:301)、外部 [grid 生成:45](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns.py:45) |

高度扫描值实际为 `sensor.pos_w.z - ray_hits_w.z - offset`，默认 offset 0.5；它不是坡度角或已融合的全局 elevation map：[外部 observations.py:292](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/observations.py:292)。研究主线禁止加入该输入，既有组件的存在不改变冻结范围：[研究记录:270](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:270>)。

## 7. 扫描器频率、历史堆叠和延迟

**高度扫描和深度图必须分开。**

| 对象 | 刷新、历史、延迟的真实配置 | 证据 |
| --- | --- | --- |
| Parkour 脚高度扫描 | `update_period=0.02 s`，名义上 50 Hz；未配置扫描历史或显式随机延迟 | [Parkour:318](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:318)、外部 [sensor 默认:34](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/sensors/sensor_base_cfg.py:34) |
| Perceptive torso 高度扫描 | 未覆盖 `update_period`，继承 0.0；每 physics update 可过期，但默认 lazy 按需取数，不能称为实测 200 Hz。critic 每控制步消费，控制为 50 Hz；height_scan term 没设历史或延迟 | [Perceptive:109](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:109)、[term:301](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:301)、[dt:730](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:730)、外部 [lazy 默认:80](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:80)、[过期判断:184](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/sensors/sensor_base.py:184) |
| Parkour 深度图 | camera 周期 0.02 s；buffer 37 帧；输出 8 帧、stride 5、延迟范围 0/1 帧。名义相邻输出 0.1 s、覆盖 0.7 s，延迟 0/0.02 s；实际 timestamp 待仿真验证 | [camera:368](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:368)、[buffer:395](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:395)、[ObsTerm:448](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:448)、[帧索引:136](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:136) |
| 本体历史 | Parkour policy 多个 term 各 8 帧；Flat 没有此设置。这不是高度扫描历史 | [Parkour:417](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:417)、[Flat:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:97) |

已有高度扫描器的随机丢帧、扫描历史堆叠、真实端到端时延标定未找到。这里“未配置延迟”不是“物理系统零延迟”的证明；也不能将 actuator 延迟当作 sensor 延迟。传感器调度依据：外部 [scene.update:498](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/scene/interactive_scene.py:498)。

## 8. MoE 到底作用于哪一侧

**Flat 两侧都不是 MoE；现有 MoE 类是 actor 和 critic 都 MoE。actor-only/shared-critic 的独立可选实现未找到。**

| 配置/实现 | 实际行为 | 证据 |
| --- | --- | --- |
| Flat | `InstinctRlActorCriticCfg` 默认类 `ActorCritic`，普通 MLP | [Flat PolicyCfg:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:12)、[rl_cfg.py:13](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:13) |
| Parkour | 4 专家、actor/critic hidden `[256,128,64]`、各自 depth encoder，WasabiPPO；不是研究主线 M4 | [instinct_rl_amp_cfg.py:32](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:32)、[rl_cfg.py:154](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:154)、外部 [all_mixer.py:20](/home/xiexuhui/instinct_rl/instinct_rl/modules/all_mixer.py:20) |
| WholeBody | 默认 MoE 8 专家，gate hidden `[128,64]`；不是四专家主实验 | [WholeBody cfg:107](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py:107)、[默认选择:150](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py:150) |
| 外部 `MoEActorCritic` | `_build_actor` 和 `_build_critic` 分别构造 `MoeLayer`，gate 不共享；不是依据类名推断 | [moe_actor_critic.py:38](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:38)、[critic:51](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:51) |
| 外部 `MoeLayer.forward` | 全量计算所有 experts，softmax gate 加权；dense，不是 hard/top-k routing | [moe.py:48](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:48) |

与研究记录的主要差异：M4 要求只有 actor MoE、critic 固定 C0，而现有类对应其 M4-full 类型。[研究记录:368](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:368>)、[shared critic:400](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:400>)、[M4-full:419](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:419>)。建议把记录中“可能存在”的双侧 MoE 更新为“已确认存在，非主实验 M4”；actor-only 接入必须另行授权后实施，本轮不做。

## 9. 专家输出和 gate 张量

**函数体确认；CPU 验证；完整 Parkour 环境未验证。** 一般 `MoeLayer` 输入 `(N,D)`，gate logits/softmax `(N,E)`，每专家 `(N,O)`，stack `(N,E,O)`，einsum 后 `(N,O)`：[moe.py:27](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:27)、[forward:48](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:48)。

| 当前 Parkour（E=4，29 actions，1 reward group） | actor | critic |
| --- | --- | --- |
| gate logits / weights | `(N,4)` | `(N,4)` |
| expert stack | `(N,4,29)` | `(N,4,1)` |
| mixture | `(N,29)`，作为 Gaussian 均值 | `(N,1)` |

配置来源：[4 experts:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:34)、[1 reward group:799](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:799)、[critic 输出:51](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:51)。多奖励任务还会为各 reward 构造独立 critic，不能一律套用一个 scalar value：[actor_critic.py:82](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:82)。

本轮 CPU 用已有 `MoEActorCritic`、Flat-sized `(96,99)` 输入、hidden `[256,128,128]`、4 experts、空 gate hidden 做纯函数诊断，得到 actor `(4,4,29)`、critic `(4,4,1)`、两侧 gate `(4,4)` 且每行和为 1。这个临时实例不是 Parkour 配置，也不是已完成的 actor-only M4，不作为 M4 参数匹配基准。forward 只返回混合输出，没有把 gate/expert 中间张量自动写入 runner 日志：[moe.py:53](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:53)。

## 10. 参数量、FLOPs 与推理时间统计

**专用自动化入口未找到；可对现有模块做离线计数。**

| 项目 | 当前事实 | 证据 |
| --- | --- | --- |
| 参数量/匹配 | 未找到自动统计或 <=5% 搜索工具。WholeBody 有 `EquivalentMlpPolicyCfg=[1024,1024,640]`，但同时扩大 actor/critic，注释的“约 2.5M”不是实测结果，也不是主实验 S-match | [WholeBody:117](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py:117)、[研究记录:350](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:350>) |
| FLOPs/MACs | 未找到专用计算工具/结果产物；线性层和 dense 全专家计算可供后续统计 | [actor_critic.py:112](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:112)、[moe.py:50](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:50) |
| timing | 有整段训练 `cProfile`，runner 输出 collection/learning 用时和 steps/s；未找到 actor-only latency benchmark；训练吞吐不能替代单策略实时延迟 | [train.py:219](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:219)、[runner:385](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:385) |

本轮离线计数现有 Flat MLP：actor 均值网络 77,981 参数、critic 75,137、Gaussian std 另 29。由各 Linear 的 `(in+1)*out` 或 `sum(p.numel())` 得到；actor 线性层 MACs 为 77,440/单样本，不含 bias、ELU、normalization。定义依据：[Flat 架构:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:12)、[actor 构建:112](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:112)、[std:95](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:95)。这是本次诊断所得，不代表仓库已有持久化统计功能，也没有据此选择 S-match 宽度。

## 11. 固定坡度、变化坡度及独立 evaluation

**坡面构件和独立播放入口存在；研究级独立 evaluator 未找到。**

- 固定参数斜坡组件存在，但未找到注册的 G1 slope locomotion 主实验任务；不能只改 Flat 的 `terrain_type` 字符串就获得研究协议。证据：[Flat 注册:6](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py:6)、[slope cfg:175](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains_cfg.py:175)。
- 波形/上下坡组件存在，但受控 `theta/kappa` 训练集、固定测试清单、失败位置映射和每 100 m 失败率未找到。已有 Flat monitor 配置为空；其他任务的 monitor 不能自动替代坡度 evaluator：[Flat:332](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:332)、[波形公式:360](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:360)。
- 通用 `play.py` 独立启动环境、选择 checkpoint、恢复模型和 normalizer，再确定性推理；它不做 PPO update，但没有固定测试矩阵、逐 env episode 聚合或成功率协议：[play.py:101](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:101)、[load/inference:159](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:159)、[loop:185](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:185)、外部 [normalizer:474](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:474)。
- Flat Play 并不是“固定 command、无噪声标准评估”：它把 command resample 改为 2 s、关掉外力与 push，但关闭 observation corruption 的语句被注释；material/mass startup randomization 仍继承。通用 play 也没有 `--seed` 参数：[Flat Play:390](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:390)、[play CLI:15](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:15)、[共享 CLI:18](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:18)。
- Shadowing 专用 play 的汇总触发条件是 `dones.sum()==dones.numel()`，不能直接用于异步结束的 N 个 locomotion 环境：[shadowing/play.py:298](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/play.py:298)。父环境 step 会先 reset 再返回 obs，评估终止位置必须在 reset 前保存，不能拿返回的下一 episode 初始位置当失败位置：外部 [manager_based_rl_env.py:215](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:215)。

研究中的固定 `vx=0.5,vy=0,wz=0` 与当前 Flat command 不同：当前范围 x `[-0.5,1]`、y `[-0.5,0.5]`、yaw rate `[-1.5,1.5]`，含 standing 和 heading mode：[Flat:77](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:77)、[研究记录:193](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:193>)。未来不但要冻结 ranges，还要冻结 standing/heading 比例及初始 yaw。

## 12. 外力、摩擦和传感器噪声

**配置存在，但当前有效范围/生命周期不等于研究协议。**

| 项目 | Flat 当前有效配置 | 证据 |
| --- | --- | --- |
| 外力/力矩 | reset 有施加接口，range 都是 0；非零外力未启用 | [Flat:281](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:281) |
| 推扰 | 每 10-15 s 设置 world xy 速度到 `[-0.5,0.5]`；不是指定 N 或 N*s 的力脉冲 | [Flat:313](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:313) |
| 摩擦 | robot static `[0.25,0.8]`、dynamic `[0.2,0.6]`、restitution `[0,0.8]`；startup，64 buckets；ground static/dynamic 均 1、multiply | [Flat:261](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:261)、[ground:39](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:39) |
| 摩擦采样语义 | 初始化采样材质桶，再向 collision shapes 分配材质；不是每 episode 独立采一个统一 `mu`。默认 `make_consistent=False`，不能假定 dynamic <= static | 外部 [events.py:225](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/events.py:225)、[分配入口:243](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/events.py:243) |
| 质量/初态 | torso mass startup 加 `[-5,5]`；reset root pose/velocity 和 joint position/velocity 随机 | [Flat:272](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:272)、[reset:290](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:290) |
| policy observation 噪声 | angular velocity ±0.2，gravity ±0.05，joint position ±0.01，joint velocity ±1.5；critic corruption=False | [Flat:98](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:98)、[critic:125](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:125) |
| CoM、PD、joint offset、delay | Flat 未找到这些随机化项；资产存在另一套 delayed actuator，但 Flat 未绑定 | [Flat Events:260](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:260)、[unitree_g1.py:573](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:573) |

其他任务确有 CoM、PD gains、joint offset、camera offset、mass 配置，但不能据此说 Flat 已全部启用；例如 Perceptive 的摩擦 static `[1.25,2.0]`、dynamic `[1.2,1.8]`，也不是研究值：[perceptive_env_cfg.py:430](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:430)、[后续随机化:442](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:442)。

与研究记录的 `Uniform[0.8,1.1]`、测试 `{0.8,0.95,1.1}` 不符。startup 的材质虽在 episode 内保持，但不代表每个 episode 重采，更不代表统一实际接触 mu。建议先定义 static/dynamic、左右脚是否同值、combine mode 和采样时机，并从 PhysX runtime 读回检查；不要直接把“改为 reset”视作无成本操作，外部实现使用 CPU 并有限量材质桶：[研究记录:211](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:211>)、外部 [events.py:172](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/events.py:172)。

## 13. seed、配置、checkpoint 和 Git commit 保存

**入口基本支持；完整复现仍有缺口。**

| 产物 | 已确认路径/行为 | 边界与证据 |
| --- | --- | --- |
| seed | `--seed` 覆盖 agent，train 赋给 env；环境初始化调用 seed；配置 YAML 含 seed | [cli_args.py:67](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:67)、[train.py:131](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:131)、外部 [manager_based_env.py:94](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_env.py:94)、[configure_seed:505](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_env.py:505) |
| 配置 | `<run>/params/env.yaml`、`agent.yaml` | 保存构建后配置；不是所有随机 draw/地形 mesh 的快照。[train.py:213](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:213) |
| checkpoint | 周期保存及 learn 结束保存 `model_<iteration>.pt` | 包含模型/optimizer、normalizer、iter；入口存在不代表已有可加载文件。[runner:195](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:195)、[save:413](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:413)、[PPO state:310](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/ppo.py:310) |
| 恢复 | `runner.load` 加载算法状态、normalizer、iteration | 缺 normalizer 仅 warning，协议必须将其视为验收失败；不是严格逐 bit 中断续训。[runner:433](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:433) |
| Git | runner 默认记录 instinct_rl；train 添加 InstinctLab；learn 保存 `git/<repo>_<shortsha>.diff`，内容为 status 与 tracked diff | 没有默认注册 IsaacLab-Instinct；未跟踪文件只在 status 列名，正文不包含在 diff。[runner:107](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:107)、[train:206](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:206)、[utils.py:99](/home/xiexuhui/instinct_rl/instinct_rl/utils/utils.py:99) |

需要明确的限制：

- `--experiment_name` 声明了但 `update_instinct_rl_cfg` 未应用；使用 `--logroot`、`--run_name` 或已经验证的 Hydra override，不依赖这个无效 CLI 选项：[cli_args.py:21](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:21)、[更新逻辑:66](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:66)、[train logroot:152](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:152)。
- checkpoint 中未找到 Python/NumPy/Torch RNG 全状态、完整 physics/env 状态或每个 episode 地形 manifest；seed 存在不等于 bitwise reproducibility。训练显式 `cudnn.deterministic=False`：[PPO state:310](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/ppo.py:310)、[runner save:418](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:418)、[train:105](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:105)。
- 现场 `rg --files --hidden --no-ignore -g '*.pt' -g '!**/.git/**' .` 无结果，当前仓库可回放的 `.pt` checkpoint 未找到；未搜索用户机器全部目录。加载入口仍存在：[play.py:104](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:104)。

## 14. 缺失项检索记录与决策修订建议

检索范围是本仓库 `scripts/`、`source/instinctlab/instinctlab/`，算法再查本机 `/home/xiexuhui/instinct_rl/instinct_rl/`；不以 README 或 docs 中的计划文字作为实现命中。代表命令如下，退出码 1 表示无匹配而非工具故障：

```bash
rg -n 'PerlinWaveTerrainCfg|PerlinSlopeTerrainCfg|PerlinSlopeUpDownTerrainCfg|PerlinTiltedRampTerrainCfg' source scripts
rg -n 'theta|kappa|slope_profile|terrain_profile|failures_per|per_100|evaluation|evaluator|eval_seed' scripts source/instinctlab/instinctlab -g '*.py'
rg -n 'FLOPs|flops|FLOP|flop|thop|fvcore|ptflops|numel\(|inference_time|inference_latency|benchmark|cprofile' scripts source/instinctlab/instinctlab /home/xiexuhui/instinct_rl/instinct_rl
rg -n 'MoE|Moe|expert|gate|_build_actor|_build_critic' source/instinctlab /home/xiexuhui/instinct_rl/instinct_rl
rg -n 'height_scanner|update_period|history_length|delay' source/instinctlab/instinctlab/tasks
rg --files --hidden --no-ignore -g '*.pt' -g '!**/.git/**' .
```

结果已经逐项与定义/调用方交叉检查：`theta` 命中主要是 mesh 随机旋转或 quaternion 插值；`slope_profile` 是第 4 节的线性坡段；`numel()` 主要统计普通 tensor；都不能算作目标研究功能。

| 研究记录条款 | 差异/待补齐 | 建议，不在本轮实施 |
| --- | --- | --- |
| [5.7:223](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:223>) | 第 4-5 节：没有完整有符号、连续坡度及 metadata 协议 | 标注“待接入”；先确定 path 坐标、量化误差、平滑性与 manifest，再新建独立 slope task |
| [6.1:264](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:264>) | 第 2、6-7 节：Flat 符合无感知边界；Parkour 不符合 | 主实验从 Flat 起步，不直接复用 Parkour obs/AMP |
| [7.3:368](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:368>) | 第 8 节：已有类同时 MoE actor/critic，不是 M4 | 把现有双侧类只列为 M4-full；独立验收 actor-only/common-critic 后才能比较 |
| [7.2:350](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:350>) | 第 10 节：匹配工具未找到，WholeBody equivalent 配置不满足 C0 不变 | 先冻结真实 M4，再计数和搜索 S-match，保存报告 |
| [9:471](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:471>) | 第 3 节：重力竖直姿态/世界 xy 滑移，不是支撑面参考 | 定义 reference、检查 reward 效果后统一冻结；不是仅换名称 |
| [5.5:193](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:193>)、[5.6:211](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:211>) | 第 11-12 节：command、摩擦范围、采样生命周期不同 | 固定 heading/standing/reset 和实际接触材质语义 |
| [10.5:553](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:553>) | 第 1、12 节：Flat 没有全部列出的随机化；未启用 delayed actuator | “已启用/未启用/计划启用”分栏；不把换低层 actuator 隐藏为普通参数调整 |
| [11:570](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:570>) | 第 11 节：没有独立坡度评估器，返回 obs 已经过 reset | 单 env_id 记录终止前状态、失败原因、距离暴露量，单独冻结评估矩阵 |
| [10.2:520](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:520>) | 256 env x 24 steps = 6144 transitions/update，20,000,000 不能整除 | 协议明确统一取整预算及实际 transitions，不把 iteration 当 transition。rollout 依据：[Flat PPO:48](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:48) |
| [12.3:675](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:675>)、[E3:765](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:765>) | E3 准入要求干预证据，但干预又排在 E3，存在流程循环 | 区分 E2 性能筛选通过与 E3 机制验证通过；不得在 E2 前增加 gate loss |

此外，研究记录尚未固定失败速度阈值、停滞时长、路径距离口径、坡面法向采样点、S-match 优先级和完整 M4 hidden sizes。它们应在实验前批准，不应在看见模型结果后补选：[研究记录:364](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:364>)、[失败定义:578](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:578>)。配套协议提出可审批的具体口径，不冒充仓库默认行为。

## 15. 当前仍无法确认

1. Isaac Sim 创建后真实关节排列、动作 scale、观测格式及部分 env reset 是否通过；只有网络 CPU shape 已验证。验收入口：[train.py:182](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:182)、[wrapper get_obs_format:208](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:208)。
2. 地形离散网格、边界与地形原点上的实际坡角/法向，以及碰撞材质有效 mu；必须测 mesh 和 PhysX runtime，不能仅看 cfg。依据：[heightfield mesh:155](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/terrains/height_field/utils.py:155)、[material event:243](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/events.py:243)。
3. 高度扫描实际更新 timestamp、reset 后读数、无命中时的非有限值行为；配置频率不等于测量结果。依据：[sensor update:184](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/sensors/sensor_base.py:184)、[height_scan:292](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/observations.py:292)。
4. 完整 checkpoint 往返、256 env 显存/吞吐、GPU 推理延迟、PPO 数值稳定性及任何坡度通过率。本轮没有运行这些验证；入口：[runner load:433](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:433)、[train learn:230](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:230)。

本轮产物是事实报告与执行协议，不是已通过 E0/E1/E2 的实验结果。
