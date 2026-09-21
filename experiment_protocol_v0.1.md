# Experiment Protocol v0.1：有符号坡度策略比较

日期：2026-09-17。状态：执行草案，E0 命令可使用现有入口；E1 坡地与 E2 仍有前置条件，尚未开始训练。本次只写文档，不修改源码、算法、现有配置或用户文件，不实现 MoE，不加入相机/LiDAR。

依据：[Research Decision Record v0.1:3](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:3>)、[冻结规则:11](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:11>)、[事实核查](/home/xiexuhui/InstinctLab/docs/repo_fact_check.md:1)。本协议所有“建议值/拟定”均为等待实验前冻结的设计，不是当前代码默认值，也不表示已获实施新算法的授权。

## 1. 当前能做什么，以及阶段准入

| 阶段 | 内容 | 当前状态与验收依据 |
| --- | --- | --- |
| E0 | Flat 环境、真实张量、异步 reset、至少 2 次 PPO update、checkpoint 往返 | 有现成入口，待运行；[train.py:182](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:182)、[runner learn:155](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:155) |
| E1-flat | 原始 S-small 平地学习，确认稳定性和奖励行为 | 现有任务可做；不是完整坡度协议。[Flat PPO:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:12) |
| E1-slope | S-small 固定坡/连续坡、reward 与 evaluator 验收 | 前置条件未满足：独立坡地任务、theta/kappa manifest、逐 env 评估未找到。最近地形组件：[hf_terrains.py:991](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:991) |
| E2 | S-small、S-match、M4；相同 C0、PPO、transitions、随机化 | 前置条件未满足：现有 MoE 同时作用于 critic；匹配工具未找到。[moe_actor_critic.py:38](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:38)、[critic:51](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:51) |
| E3 及以后 | 路由干预、oracle 或外部感知 | 保持冻结；依研究记录的阶段要求另行决策。[研究记录:763](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:763>) |

不得用 Parkour 直接替代 E2：其配置启用了深度 encoder、双侧 4 专家及 WasabiPPO，改变了信息、critic 和学习目标。[Parkour cfg:32](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:32)、[算法:44](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:44)。

## 2. E0：不修改代码的最小复现

以下为待执行命令；本轮没有运行仿真。所有命令在 `/home/xiexuhui/InstinctLab` 内执行，训练产物写入新建临时目录，不删除或覆盖已有日志。使用已有 Python 3.11 环境路径，先确认包位置。已有审计定位结果仅供核对：[repo_audit.md:299](/home/xiexuhui/InstinctLab/docs/repo_audit.md:299)。

```bash
cd /home/xiexuhui/InstinctLab
git status --short --untracked-files=all
git rev-parse HEAD
git -C /home/xiexuhui/instinct_rl rev-parse HEAD
git -C /home/xiexuhui/IsaacLab-Instinct rev-parse HEAD
protocol_python=/home/xiexuhui/miniconda3/envs/instinctlab/bin/python
"$protocol_python" --version
"$protocol_python" -m pip show isaaclab isaacsim instinctlab instinct-rl torch
"$protocol_python" scripts/list_envs.py
```

通过标准：实际 interpreter、editable package 路径及三个提交与记录一致，列出 `Instinct-Locomotion-Flat-G1-v0` 及 Play 变体。`list_envs.py` 本身启动 headless AppLauncher，不额外传不存在的参数：[scripts/list_envs.py:13](/home/xiexuhui/InstinctLab/scripts/list_envs.py:13)、[任务注册:6](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py:6)。失败时记录首个异常、环境版本、资源路径，不继续扩大环境数。

### 2.1 两次更新与产物检查

```bash
protocol_runs=$(mktemp -d /tmp/instinctlab-protocol-e0.XXXXXX)
"$protocol_python" scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-v0 \
  --headless --device cuda:0 --num_envs 4 --seed 42 \
  --max_iterations 2 --logroot "$protocol_runs" --run_name E0_s42
```

这些参数由 [train.py:21](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:21)、[覆盖配置:124](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:124)、[cli_args.py:24](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:24) 解析。默认 Flat rollout=24；这次仅 192 environment transitions，不能用来判断步态好坏或学习收敛。[Flat PPO:48](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:48)。

验收保存的 `params/env.yaml`、`params/agent.yaml`、`model_2.pt`、TensorBoard 文件及 `git/`。两次 update 应完成且 loss/obs/action/reward 有限；normalizer 和 optimizer 状态必须存在。保存实现：[train.py:213](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:213)、[runner final save:205](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:205)、[PPO state:310](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/ppo.py:310)。只看进程退出码或历史日志不够。

### 2.2 真实 shape 与 reset 验收

需要记录的运行时字段如下；可以使用现有 `--debug` attach，在 `train.py` 创建 wrapper 后、learn 前检查，不需要编辑源文件。调试入口会等待客户端连接，仅在准备好调试器时使用：[train.py:90](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:90)、[wrapper:201](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:201)。

| 字段 | 预期及拒绝条件 |
| --- | --- |
| `env.get_obs_format()` 和实际返回观测 | policy `(4,96)`、critic `(4,99)`；term 顺序与维度保存到验收记录；任一 reset/env_id 改变 shape 则停止。依据：[wrapper:208](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:208) |
| `env.num_actions`、joint names、scale/offset | 29 actions；记录解析后的全顺序；确认不是统一 scale=0.5。依据：[Flat:364](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:364)、[action:169](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:169) |
| `step_dt`、`physics_dt`、decimation | 0.02、0.005、4；不要用 wall-clock FPS 代替控制 Hz。依据：[Flat:357](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:357) |
| `terminated`、`truncated`、reset env IDs | 非法接触和 timeout 分开；只重置选中的环境；动作历史清零且没有 NaN。依据：[Flat terminations:232](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:232)、[父环境 reset:215](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:215) |

两次更新只有 48 个控制步，无法覆盖 1000 步 timeout。reset 验收应另做至少 1000+ 控制步的观察，并在调试会话单独让一部分 env 到达 timeout，确认其余环境继续；若自然跌倒一直先于 timeout，不能把“运行够 20 s”当 timeout 已验收。现有父环境公开 reset 支持 env_ids；使用调试表达式或外部交互检查记录，不在本轮新建测试脚本。[manager_based_env.py:342](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_env.py:342)、[episode 计数:200](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:200)。

### 2.3 加载与继续更新

将下列 `protocol_run` 替换为本次产生的精确绝对目录，不使用“最新 checkpoint”跨实验模糊匹配。

```bash
protocol_run='/absolute/path/to/the/E0_run_directory'
timeout --signal=INT 120s "$protocol_python" scripts/instinct_rl/play.py \
  --task Instinct-Locomotion-Flat-G1-Play-v0 \
  --headless --device cuda:0 --num_envs 4 \
  --load_run "$protocol_run" --checkpoint model_2.pt --agent_cfg

"$protocol_python" scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-v0 \
  --headless --device cuda:0 --num_envs 4 --seed 42 \
  --resume --load_run "$protocol_run" --checkpoint model_2.pt \
  --max_iterations 1 --logroot "$protocol_runs" --run_name E0_resume_s42
```

通用 play 没有非视频有限步 CLI，上述 timeout 包含启动时间；退出码 124 不直接等于仿真失败，要确认实际进入 step。使用 `--agent_cfg` 恢复模型配置；此 E0 原始 Flat 输入与 Play 相同。`--env_cfg` 会覆盖前面应用的 num_envs/device，需单独核验而非盲目加入。依据：[play.py:92](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:92)、[配置加载:120](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:120)、[播放循环:185](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:185)。

验收：同一权重+同一 normalizer 对同一输入给相同确定性 action；没有 missing-normalizer 警告；resume 从 iter=2 开始再更新一次并生成 `model_3.pt`。这里是权重/优化器恢复验证，不声称 physics/RNG 精确续跑：[runner load:433](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:433)、[迭代起点:151](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:151)。

可视化只用于检查：去掉 `--headless` 打开 GUI；或添加 `--video --video_length 300 --video_start_step 0` 录制。视频路径和录完后的 `code -r` 调用来自 [play.py:130](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:130)、[play.py:213](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:213)，后者缺 VS Code CLI 时可能导致收尾报错。TensorBoard 可用 `tensorboard --logdir "$protocol_runs"`，writer 依据：[runner:120](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:120)。可视化不计入正式推理 benchmark。

## 3. E1/E2 前必须冻结的环境规格

以下是实验目标，尚未全部接入现有代码。基础保持 Popsicle 29 动作、50 Hz、policy 96 维、critic 99 维、无观测历史、无外部感知、无地形/摩擦 oracle；来源：[Flat actions:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:65)、[observations:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:97)、[研究信息边界:264](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:264>)。

| 项目 | 协议要求及当前差异 |
| --- | --- |
| command | 主集 `vx=0.5 m/s, vy=0, wz=0`；次级 0.3/0.7。关闭 standing/heading 随机分支，明确初始 yaw 沿路径。当前命令含站立、heading 和宽范围采样，需后续配置调整。[Flat:77](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:77)、[研究记录:193](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:193>) |
| actuator | 保持当前 Flat implicit actuator，不把未绑定的 delayed 配置默认为已启用。若以后要换 delayed actuator，必须单列低层控制变更，重新 E0。[unitree_g1.py:269](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:269)、[绑定:573](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:573) |
| reward | 三模型 term/weight 完全一致；先解决世界水平姿态与坡面参考的定义，先以 S-small 验收。当前 orientation 使用 projected gravity xy。[外部 rewards.py:91](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/rewards.py:91)、[研究记录:487](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:487>) |
| reset/termination | 固定可行初始位置、yaw 和高度相对地形的规则；超时、物理跌倒、停滞、越界分开。当前随机 world xy/yaw 与平地 root reset 不应直接套入斜坡。[Flat:290](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:290) |
| friction | 研究目标 train `[0.8,1.1]`、test `{0.8,0.95,1.1}`；建议明确为左右足相同 `mu_static=mu_dynamic=mu`、ground=1、multiply，episode 内恒定，并记录读回值。此为待冻结建议，当前 Flat 是两套不同范围、startup 分配材质桶。[Flat:261](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:261)、[研究记录:211](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:211>) |
| other randomization | manifest 逐项列 mass/CoM/PD/joint offset/initial state/noise/actuator delay：范围、distribution、startup/reset/step 生命周期、是否有效。Flat 没有全部项目；未启用的明确写 disabled，不自动从其他任务借用。[Flat events:260](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:260)、[研究记录:553](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:553>) |
| disturbance | 主评估建议关闭主动推扰，扰动另建配对测试；训练是否保留现有 velocity push 必须在 E1 前冻结。当前非零 force 未启用，push 是速度设置，指标不能用 N 表示。[Flat:281](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:281)、[push:313](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:313) |

对姿态 reward 的必要决策：定义法向来自支撑脚局部平面还是路径中心坡面，定义期望 body pitch；只用法向点积不足以约束 yaw。坡地滑移若需报告真实切向距离，应使用速度在切平面的投影，当前 world xy penalty 不等于该指标。[regularizations.py:800](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/rewards/regularizations.py:800)。本协议不预先认定 torso 必须与地形法向一致。

## 4. 地形定义、清单与可复现性（前置条件：未实现完整链路）

沿预先固定的水平参考路径定义 `s`（米），world Z 为高度；local path +s 为实验前进方向，上坡正、下坡负。保存 path-to-world 变换。机器人偏航或倒退不改变地形标签。计算用 rad，报告角度 deg 和变化率 deg/m：`theta(s)=atan(dh/ds)`，`kappa(s)=dtheta/ds`。这是研究定义，现有 pyramid slope 是 ratio，不能直接混用：[研究记录:110](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:110>)、[hf_terrains.py:78](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:78)。

建议先通过连续 `theta(s)` 定义剖面，再积分 `dh/ds=tan(theta)` 得到 h；保留解析标签与实际网格采样两份。固定坡测试段允许有起止缓冲，但只将稳态段计入 fixed-slope 指标，入口过渡另外标注。不得将拼接处坡角跳变的线性坡段当作连续坡主实验：[现有分段赋值:1033](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:1033)。

### 4.1 拟定数据划分

| 数据集 | 角度/族 | seed 与用途 |
| --- | --- | --- |
| Train | 固定 `-30,-20,-10,0,10,20,30 deg`；连续坡限制 ±30、`max|kappa|<=20 deg/m`，过渡长度 4-8 m | 模型种子 42/43/44；建议 terrain seed 分别 `[10000,19999]`、`[20000,29999]`、`[30000,39999]`；三模型同一 seed 使用同一 terrain 清单 |
| Validation | 训练范围内，独立剖面与 reset/noise seeds | 建议 `[40000,40999]`；只做 E1 难度和环境验收；不得通过最终 test 选 checkpoint |
| Test-ID | 训练范围内固定坡及未见剖面，0 deg/轻微起伏退化检查 | 建议 `[50000,50999]`；预先保存，与训练隔离 |
| Test-OOD 主集 | 固定 `-35,-25,-15,0,15,25,35 deg`；连续含 `-35->35`、`35->-35`，未见族/长度/变化率组合，含 `max|kappa|=25` | 建议 `[60000,60999]`；主结论数据 |
| Exploratory | `max|kappa|=30`、±40 deg、mu=0.7 分开标记 | 建议 `[70000,70999]`；不合并入主指标 |

角度和变化率来自 [研究记录:137](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:137>)、[变化率:181](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:181>)、[train/test 隔离:241](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:241>)。上表 terrain seed 区间是本协议建议；代码当前没有该划分。seed 区间不同也不能独自证明分布隔离，还要检查 profile hash、参数和生成族。

建议 train 用 quintic 过渡 `p(u)=6u^5-15u^4+10u^3`，test OOD 可用端点导数为零的另一平滑族，如 raised cosine；训练同族的未见 seed 仅归 Test-ID。两种族都必须由生成后 `theta/kappa` 实际验收，不能凭名字判“平滑”。这只是地形设计草案，没有加入代码。

变化率与长度不能独立随意指定。例如 quintic 的 `max|kappa|=1.875*|delta_theta|/L`，角度使用 deg 时单位为 deg/m；`-30->30` 若要 <=20，L 至少 5.625 m，L=4 m 会超限。raised cosine 则是 `pi*|delta_theta|/(2L)`。这些是数学推导，用于核查研究记录的 4-8 m、20/25/30 deg/m 是否同时成立，不是已生成数据。[研究记录:150](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:150>)、[研究记录:185](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:185>)。

地形资产尺寸还需单独设计：当前 Parkour tile 是 8x8 m，不能默认容纳 8 m 过渡加两端缓冲或多个过渡；Flat 20 s episode、0.5 m/s 理想水平距离仅 10 m。若增大 tile/episode，三模型必须一起冻结并记录。[Parkour:47](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:47)、[Flat:359](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:359)。

### 4.2 生成与保存的验收产物

每条 profile 必须有 `profile_id, split, generator_family/version, seed, horizontal_path_transform, s[], h[], theta_rad[], kappa_rad_per_m[], segment/windows, mesh_hash, requested_and_measured_angle, horizontal/vertical_resolution, collision_materials, initial_pose, goal, timeout`；把实际摩擦、随机化 draw、reset seed 关联到每个 episode。保存内容依据研究要求：[研究记录:230](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:230>)；现有 `FiledTerrainGenerator` 只保存 cfg/difficulty/seed，不能替代这些产物：[terrain_generator.py:21](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_generator.py:21)。

验收：读取 mesh 计算局部坡度/法向，报告量化误差和连接处最大变化率；验证边界、平台和墙未进入测量走廊。针对不同 env_ids 复现同一个 profile/seed，独立 reset 后资产、标签与 episode ID 一致。Flat 主实验 actor/critic 不读取该 manifest；仅生成器和 evaluator 使用真值。[研究记录:292](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:292>)、[critic 边界:496](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:496>)。

## 5. 模型、公共 critic 与复杂度匹配（M4/S-match 尚未具备）

| 模型 | 冻结目标 | 现有证据与缺口 |
| --- | --- | --- |
| S-small | actor `96->256->128->128->29`，ELU | 现有 Flat 架构；本轮 CPU 计数均值网络 77,981 参数。[Flat PolicyCfg:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:12) |
| C0 | 建议固定为 `99->256->128->128->1`，ELU，独立 critic normalization；所有方法相同结构/优化规则，但各训练各自权重 | 与 Flat 一致；75,137 参数。这里“公共”不表示跨三个训练共享一份正在更新的权重。[actor_critic.py:127](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:127)、[normalizers cfg:37](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:37) |
| M4 | 4 identical experts + softmax gate；同输入，dense action-mean mixture；C0 不专家化；仅 PPO loss | 专家/gate hidden sizes 尚需单独冻结；不能自动把 Parkour 的 `[256,128,64]` 当研究已定值。当前类 critic 也是 MoE，主模型未找到。[moe.py:48](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:48)、[moe_actor_critic.py:51](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:51)、[研究记录:386](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:386>) |
| S-match | 单一 MLP actor；匹配真实 M4 actor 参数和计算；C0 不变 | 宽度必须在 M4 实例化后搜索冻结；本轮不选、不实现。WholeBody equivalent 同时扩大 critic，不能照搬。[EquivalentMlpPolicyCfg:117](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py:117) |

M4 前置验收：actor gate `(N,4)` 每行和为 1，expert stack `(N,4,29)`，混合 `(N,29)`；critic 单个 `(N,1)`，无 gate；每步确实计算 4 experts。所有模型 std 参数 `(29,)` 处理一致，normalization、动作尺度及原始上一步 action 语义一致。[现有 dense 实现:48](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:48)、[Gaussian:159](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:159)。

### 5.1 计数口径

拟定 `complexity.json`（现有自动生成入口未找到）：分别保存 actor experts、gate、共享预处理、critic、std 的 trainable parameter 数；actor 推理预算包括 experts+gate+必需预处理，Gaussian std 单列，不悄悄把 critic 混进 actor count。normalizer 的运行统计属于 buffer，单列存储和计算开销。计数目标依据：[研究记录:356](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:356>)。

线性层每样本 MACs=`in*out`，带 bias 参数=`(in+1)*out`。统一报告 MACs；若报 FLOPs，声明一个 multiply-add 算 2 FLOPs，bias/ELU/softmax/加权求和及 normalization 是否额外计入。dense MoE 必须累加 4 个 expert、gate 和 mixture 的实际 forward。已有 cProfile 和训练 throughput 不替代此计数：[train.py:219](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:219)、[moe.py:50](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:50)。

建议预先冻结单 MLP 的候选搜索空间为 2-4 个 hidden 层、各层宽度为 8 的倍数且不超过 2048；只做计数，不训练候选。以 `abs(P_S-P_M)/P_M`、`abs(MAC_S-MAC_M)/MAC_M` 均 <=5% 为准；满足者优先最小化两误差的最大值，再最小化参数误差，再按较浅层数决定。找不到时暂停 E2 并先登记扩大空间或放宽标准，不能看结果后挑 S-match。[研究记录:360](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:360>)。这些搜索规则为待冻结建议，仓库未实现自动搜索。

### 5.2 latency benchmark

拟定统一 GPU/CPU、驱动、PyTorch、dtype、线程数、功率状态；`eval()` + `inference_mode()`，输入固定在目标设备。分 batch=1（部署相关）和 batch=256（训练吞吐相关），每种模型先 warmup 200 次，5 组各 2000 次重复；GPU 用 CUDA events 并在读计时前同步；CPU 用单调高分辨率时钟。报告 median/p95/p99、组间变化，区分纯 actor 与 normalization+actor；另测设备传输开销，不把 batched time/256 当单机器人的闭环时延。这是待实施测量规范；现有仓库专用推理计时入口未找到，最近是整体训练计时：[runner:385](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:385)。

## 6. PPO、训练 seed 与 transitions 预算

沿用 Flat 标准 PPO：rollout 24、epochs 5、minibatches 4、clip 0.2、value loss coef 1、entropy 0.008、lr 1e-3/adaptive、gamma 0.99、lambda 0.95、desired KL 0.01、max grad norm 1。最终以导出的 agent.yaml 和实际 optimizer 类为准；本地默认配置指定 AdamW，外部 PPO 构造默认 Adam，不能只读外部默认值就说当前用 Adam。[Flat AlgorithmCfg:20](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:20)、[本地 optimizer:195](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:195)、[外部 optimizer:89](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/ppo.py:89)。

筛选使用单 GPU、256 env、seed 42/43/44；三个方法同一 seed 配对、各自从头初始化，保持网络构建后环境随机源配对可核查。单一全局 seed 不保证不同网络消耗 RNG 后产生相同环境随机流，故 terrain/reset/noise 清单需要显式分离、保存和核验。现有 seed/训练入口：[train.py:131](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:131)、[模型构造:64](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:64)；研究要求：[研究记录:529](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:529>)。

20,000,000 / (256*24) = 3255.2083。建议把“20M 筛选上限”具体冻结为 **3255 次完整 update = 19,998,720 transitions**，三模型统一，少于名义上限 1280（0.0064%）；若要求至少 20M，则统一 3256 次=20,004,864，并在开始前修订预算。不得不同模型各自取整。runner 以 learning iterations 控制循环，并无精确 20M 截断参数：[runner:151](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:151)、[研究预算:520](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:520>)。

固定最后一个完成预算的 checkpoint 为主评估模型；中间 checkpoint 仅诊断，不用测试表现择优。保存间隔建议统一 100 updates，保持最终保存；文件中的 `iter` 与累计实际 transitions 分开记录，避免周期保存编号的边界差异。实现每轮先更新、再保存、再递增 iter：[runner:190](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:190)。resume 不能凭文件名简单估算已训练预算；主 E2 优先从头完整运行，中断运行单列。

若 OOM，全部模型统一降低 env 数并重新冻结预算算式；不得仅降低一种方法的并行度、延长其训练或给它额外 curriculum。[研究记录:539](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:539>)。

## 7. 独立 evaluation（所需实现未找到）

评估必须新进程加载精确 checkpoint + normalizer + 冻结环境配置，不更新网络或 normalizer，使用 Gaussian 均值，不开 `--sample`。`get_inference_policy()` 已提供 eval 模式与 normalization；完整坡度 evaluator 仍需后续补齐。[runner:474](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:474)、[play.py:165](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:165)。

建议主测试每个 `(terrain stratum, theta/transition, mu)` 单元固定至少 20 个 profile/reset/noise 案例，所有模型和训练 seeds 共享；各案例的目标距离和最长时间在模型比较前固定，不因某模型失败而添加案例。actor 观察噪声是否保留、随机化 draw 和扰动是否启用按第 3 节 manifest 固定。数字 20 是草案最低重复数，不能保证统计效力；E1 后可按预估失败率统一增加，增量决策应在 E2 前完成。[测试清单要求:258](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:258>)、[共同评估约束:434](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:434>)。

### 7.1 逐环境记账

每个 env_id 单独维护 episode ID、profile ID、起止步、终止原因、已行距离、目标/超时、初始和随机化状态、失败时 s/theta/kappa/mu。不能等待全部 env 同时 done。必须在 auto-reset 前捕获最终位置、速度、接触、reward term，reset 后返回的 obs 属于新 episode。[父环境 step:215](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:215)、[wrapper 合并 done:162](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:162)、[不适用的全 env done 汇总:298](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/play.py:298)。

拟定失败口径（E1 前冻结，现有 Flat 仅有 timeout/非法接触）：

- 非法接触/跌倒：现有 sensor term 的 raw reason，达到一次计一次失败；不能一个事件按多个 term 重复计数。
- 停滞/速度保持失败：建议启动宽限 2 s 后，滑动 2 s 窗口平均有符号路径速度 <0.1 m/s，且不处于计划结束缓冲，判失败。
- 路径走廊越界：判失败，走廊半宽在 manifest 给定；不得重置后算新距离弥补失败。
- 目标完成：成功结束。timeout 若未完成目标，按预注册规则算未完成失败；人为中断/基础设施故障单独作 censored，不算策略成功。

第一项可从现有 Flat term 继承；后三项是待实现/待冻结定义，不是仓库行为。[Flat terminations:232](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:232)、[研究失败定义:578](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:578>)。若仿真异常有系统性、只影响某模型，不得只剔除其失败样本，应停止并调查。

### 7.2 指标口径

主指标 `100 * total_failures / total_distance_m`。建议 distance 取沿固定水平路径的“最大已到达 s - 起始 s”，上限为路线长度，防止原地横移/往返累计里程；失败前短距离仍入分母。若用户希望坡面弧长，应另列 `integral ds/cos(theta)`，不能在结果出现后切换主距离口径。总距离为零时记 undefined 并报告失败数，不能加小常数掩盖崩溃。研究记录未明确距离语义，本段是待冻结的补充。[主指标条款:576](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:576>)。

固定坡、缓变坡、快速坡、正负转换、坡顶/坡底窗口、各摩擦值分别计算失败数与该区域距离暴露量；失败所在区域用终止前 s 标签。stratum 边界与窗口宽度应由 manifest 预定义；总表同时列每 cell 的完成率、尝试数、计划距离、实际距离，防止存活较长样本带来的选择偏差。[研究记录:585](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:585>)。

次指标至少保存实际路径速度、command error、局部姿态误差、切向滑移、接触冲量、关节 torque、action rate、acceleration、恢复时间、GPU 显存和吞吐。能耗代理建议分正机械功 `integral max(tau*qdot,0)dt` 和负功绝对量，按上坡/下坡分别报告，不把负功视作电能回收；冲量需有 physics-rate 力采样，只有控制末帧值时不得称精确冲量。这些指标并未全部实现，现有 Flat monitor 为空。[Flat monitor:332](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:332)、[次指标要求:596](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:596>)。

### 7.3 比较与停止

逐训练 seed 报 S-small/S-match/M4，并按共同测试案例配对；不能把 256 个 env 当 256 个独立训练 seed。报告三 seed 的方向、分层原始计数/距离，及按 seed、profile 分层重采样的不确定性；三 seed 仅用于筛选，不能夸大为充分统计证明。[研究记录:551](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:551>)。

保留研究门槛：每 100 m 失败率相对下降 >=20%，或成功率提高 >=5 个百分点；三 seed 方向一致，收益出现在未见过渡/高变化率，实际速度下降 <=10%，C0 公平。成功率替代路径须预先声明且仍完整报告主失败率；若 S-match 失败率为 0，相对下降未定义，不可宣布满足 20%。[研究记录:675](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:675>)。

研究记录把“干预能够改变收益”既作为 E3 进入条件，又把干预放在 E3。建议修订为 E2 性能筛选通过后仅进入 E3 验证；E3 干预证据完成前不宣称专家机制有效。这是流程修订建议，保持原文未改。[准入:682](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:682>)、[E3:765](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:765>)。

无效实验：E0 不通过、Flat S-small 无法学习、NaN/reset 错误、obs shape 漂移、critic/reward/预算不同、test 清单不固定、checkpoint 选择不同、模型独有的人工调参。任一发生先停止比较。[研究记录:642](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:642>)。若 M4 不优于 S-match、仅胜 S-small、收益来自降速或 MoE critic、seed 方向不稳定或超计算预算，暂停扩大 MoE。[暂停判据:656](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:656>)。

## 8. 每次运行需要归档的内容

| 产物 | 当前保存支持 | 本协议额外要求 |
| --- | --- | --- |
| `params/env.yaml`, `agent.yaml` | 已有：[train.py:213](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:213) | 核对最终 obs/action、events、reward、optimizer、seed 和 budget；保留完整命令 |
| `model_*.pt` | 已有模型/optimizer/normalizer/iter：[runner:413](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:413) | 主 checkpoint 的 SHA-256、实际 transitions、加载验收；禁止自动忽略缺失 normalizer |
| `git/*.diff` | 自动 InstinctLab+instinct_rl，短 hash 和 tracked diff：[runner:107](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:107)、[utils.py:99](/home/xiexuhui/instinct_rl/instinct_rl/utils/utils.py:99) | 三仓库完整 SHA、dirty 状态；归档未跟踪实验配置/协议正文与 hash；IsaacLab-Instinct 需额外记录 |
| 环境 manifest | 专用完整入口未找到；当前配置 dump 不能替代 | Python/package lock、GPU/driver/CUDA、dtype、determinism、资产 URDF/hash、任务 ID |
| terrain/evaluation manifest | 完整链路未找到；最近为 cfg/difficulty/seed：[terrain_generator.py:21](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_generator.py:21) | 第 4、7 节的 profile、真实标签、每个 env episode、split 和随机化样本 |
| complexity/latency report | 专用入口未找到；只有 cProfile：[train.py:219](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:219) | M4/C0/S-match 结构、逐模块 count、MAC/FLOP 定义、原始 timing、匹配偏差 |

不依赖 `--experiment_name` 命名实验：该参数没有被 update 函数应用；使用精确 `--logroot` 和 `--run_name`，运行完成后写入实际目录映射。[cli_args.py:21](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:21)、[更新:66](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:66)、[train logroot:152](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:152)。

## 9. 本轮结束时的状态

已完成事实核查、URDF 统计和现有网络 CPU 局部前向；E0 仿真、checkpoint 往返、E1/E2 训练与 evaluator 均未执行。当前可回放 `.pt` 未找到。后续先执行第 2 节；E1-slope/E2 必须先补齐并批准地形、reward、随机化、M4/C0、复杂度和 evaluator 的具体规格，再实施。不能把这两份文档当作算法实现或实验成功证明。对应入口和限制见 [事实核查第 11 节](/home/xiexuhui/InstinctLab/docs/repo_fact_check.md:179) 与 [研究记录 E0:712](</home/xiexuhui/InstinctLab/docs/Research Decision Record v0.1.md:712>)。
