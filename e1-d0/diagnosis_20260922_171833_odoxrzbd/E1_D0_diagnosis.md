# E1-D0：Flat G1 早期 `base_contact` 终止诊断

日期：2026-09-22（Asia/Shanghai）  
状态：诊断完成；没有修改原始 Flat 任务、奖励、执行器或终止阈值，没有启动长训练。

## 结论先行

`model_200` 的 32/32 `base_contact` 不是 32 次已证实的“物理跌倒”。在共享初态的独立短时复现中，首个越阈值接触稳定地出现在同侧 `wrist_yaw_link` 与 `hip_roll_link` 之间；过滤后的 PhysX 接触点约在离地 0.66 m 处，示例力 478 N。此接触属于机器人自碰撞，且触发时 root 高度仍约 0.78–0.81 m。原始 `base_contact` 判据把这类非足部 link 自接触计为非法接触，因此评估器的 `fall = base_contact` 只能表示终止项，不等同于倒地。

终止实现本身的函数调用和 body 映射没有发现索引错配。传感器历史在 reset 时确实清零；termination manager 的当前 `terminated` buffer 在 reset 调用后会暂时保留上一终止值，下一次 `compute()` 开始时才清零并重新计算。这是可观测的 buffer 生命周期，但不是本次跨 episode 接触污染：非法接触函数读取的 contact history 在 reset 后为零。

`model_200` 还表现出明显的动作/目标异常：共享 8 个案例的 raw action 最大绝对值 2.497，16 个 per-step 记录越过 soft joint limit，12 条越过 hard joint limit；`q_target = raw_action * scale + offset` 与执行器实际 position target 逐项一致，没有 action scale 或 joint order 漂移。`model_initial` 的 raw action 最大绝对值仅 0.255，未观察到目标越限。zero raw action 的确表示默认关节姿态的位置控制，不是关闭电机。

因此最符合证据的解释是：原始非法接触判据对 self-collision 很敏感，而 `model_200` 的早期策略更新产生了异常关节目标，使自碰撞更早发生；`5.72` 个控制步是接触终止时间，不是已确认跌倒时间。nominal 默认姿态、零速度、零命令、关闭噪声和随机化的单环境排查也在约 0.24 s 发生自接触，说明“原始随机化”不是唯一必要条件，但不能把该单环境工程排查当作原始任务性能。

## 1. `base_contact` 的实际含义

运行时解析结果保存在 [`runtime.json`](../logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/final_pairs/runtime.json)。真实调用为：

```text
isaaclab.envs.mdp.terminations.illegal_contact
  sensor_cfg = contact_forces, body_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,19,20,21,24,25,26,27]
  threshold = 1.0 N
```

函数实际计算 `norm(contact_sensor.data.net_forces_w_history)`，沿 history 维取最大值，再对 monitored body 取 any。它不是只读取当前 physics step，也不是只检查 torso：`history_length=3`、`update_period=0.005 s`、physics dt `0.005 s`，因此每次 control step 的判定覆盖最近三个 physics samples。Flat 的 29 个 sensor body 顺序如下：

```text
sensor: torso,
left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll/pitch/yaw,
right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/pitch/yaw,
waist_roll, waist_yaw, pelvis,
left_hip_pitch/roll/yaw, left_knee, left_ankle_pitch/roll,
right_hip_pitch/roll/yaw, right_knee, right_ankle_pitch/roll
```

它们映射到 robot body ID `[0,1,4,7,10,14,18,22,2,5,8,11,15,19,23,3,6,9,12,16,20,24,26,28,13,17,21,25,27,29]` 中相同名称的 29 个 rigid bodies；monitored IDs 是上面列出的 23 个，排除了两侧 ankle roll link 和 ankle pitch link 以外的足部支撑 link 仍然存在于 sensor，但没有被 `base_contact` 监测。运行时 joint/action 顺序也是 29 个 robot joints，`action_dim=29`。

资产运行时启用了 articulation self-collision。reset 调用 `scene.reset`，ContactSensor 的 `net_forces_w`、`net_forces_w_history` 和可用 force matrix 均清零；诊断记录的每次 reset 后 history/current 最大值都是 0。Termination manager 的 `_terminated_buf`/`_truncated_buf` 会在下一次 `compute()` 开头清零，因而 reset 返回后的短窗口里可以看到上一 episode 的 `terminated=true`，但这不是下一步的 contact 判定。

## 2. 共享初态短时对照

每个 original 对照使用 8 个预先固定案例，保存 root/joint state、command、material、mass、inertia、COM、interval timers、external wrench、初始 observation 和 contact buffers。三控制器的 root/joint/command/material/质量/惯量/COM/计时器/wrench/observation 逐项相同；contact buffer 在 lazy sensor 首次读取时可能更新，故只对 monitored bodies 的初始力作一致性核对（均为 0），不宣称足部 buffer 的逐字节配对。异步 reset 后观测噪声和后续随机数消耗不作严格配对声明。每案在首次终止或 2 s 停止；下表的 per-case 记录是首次 episode。

| 控制器/条件 | 平均控制步 | base_contact | raw action 最大绝对值 | hard target 越限记录 | 首次 link |
|---|---:|---:|---:|---:|---|
| zero raw action，original | 10.88 | 8/8 | 0 | 0 | wrist_yaw + hip_roll |
| `model_initial.pt`，各自 normalizer | 16.13 | 8/8 | 0.255 | 0 | wrist_yaw + hip_roll |
| `model_200.pt`，各自 normalizer | 5.25 | 8/8 | 2.497 | 12 | wrist_yaw + hip_roll |
| nominal default pose/zero speed/zero command/no noise | 12 | 1/1 | 0 | 0 | wrist_yaw + hip_roll |

原 evaluator 的 32 episode 统计仍为 `model_initial=14.5` 步、`model_200=5.71875` 步，均为 32/32 contact；诊断首批 8 个 env 的每个首次 episode 长度与原 `evaluation_v2` 的 env 顺序一致。nominal 单环境延迟 reset 仅用于观察，最多 2 s；它在 0.24 s 记录了 self-contact，不能称为稳定站立或稳定行走。

按对象过滤的 PhysX 数据见 [`filtered_contacts.jsonl`](../logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/final_pairs/filtered_contacts.jsonl)，所有代表性首次触发均是 robot-to-robot；GPU filter 对 ground terrain 输出“不支持”的警告，所以该 API 不提供完整的 ground 对象分类。已确认的代表性触发对象是自碰撞；其余没有过滤对象数据的接触标记为“未确认”，没有从 net force 推断对象。

代表性几何逐帧图和 GIF：[`representative_collision.gif`](../logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/representative_collision.gif)。它由运行时 USD collision primitives（capsule/sphere 等）和 body poses 重建，不是相机视频；完整数值记录在 [`representative_trace.csv`](../logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/representative_trace.csv)、[`physics_trace.jsonl`](../logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/final_pairs/physics_trace.jsonl) 和 [`control_trace.jsonl`](../logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/final_pairs/control_trace.jsonl)。每个 physics row 含 root pose、joint q/dq、raw action、scale/offset 后 q_target、force/history、joint limit/saturation proxy 和碰撞几何位置；control row 另外含 termination flags。

## 3. PPO、normalizer 和 reward 链路

静态代码与一次独立、最多 2 updates 的诊断记录见 [`ppo_v2/result.json`](../logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/ppo_v2/result.json)。结论如下：

- adaptive KL 使用当前 minibatch 的 `mu/std` 与 rollout 中保存的 `old_mu/old_sigma`，公式为 Gaussian KL；它在每个 minibatch 更新 `self.learning_rate`，并同步写入 optimizer 的每个 `param_group['lr']`。诊断中变量和 optimizer LR 一致；KL 在首个 update 的后续 minibatch 约 `0.018–0.055`，第 15 个 minibatch 已到 `1e-5` floor。这个“很早到 1e-5”有直接证据，但它是 adaptive schedule 对高 KL 的反应，尚不足以判定实现错误。
- rollout 的 old log-prob、old mu/std 与同一 transition 的 action 对应：复核误差分别 `7.63e-6`、`2.7e-7`、`0`（sigma）；未发现 old/new 数据错位。
- policy/critic EmpiricalNormalization 在 rollout 中以 training mode 更新，checkpoint 保存 count=307200；独立评估加载各自 checkpoint 的两个 normalizer 并冻结，状态未变化。`model_initial` count=0，没有混用 model_200 统计。
- `act_inference` 的 mean action 与 actor MLP 直接输出一致；评估不采样、不 update。动作 affine 是 `q_target=raw*scale+default_joint_pos`，实际 target 逐项一致。
- RewardManager 每个 term 乘 `dt=0.02`；`termination_penalty=-200` 因而终止步贡献 `-4.0`，没有发现额外 dt 缩放。它解释了两套评估 return 接近 `-4`，不能用 return 单独判断跌倒程度。

## 4. 已确认、支持、待验证、排除

**已确认事实**

- 32/32 是原始 `base_contact` termination，0/32 timeout；5.71875 是终止控制步平均数。
- 函数为 `illegal_contact`，读取三步 history 最大值，阈值 1 N；body 名称与 ID 已在运行时记录。
- 代表性触发是腕部和髋部 link 的自碰撞，触发时 root 通常仍在约 0.78 m；因此“终止”已确认，“已经倒地”未确认。
- zero action 仍驱动默认位置目标；默认 nominal 也会自碰撞，说明原始执行器/姿态启动本身存在短时自碰撞风险。
- model_200 比 model_initial 产生更大 raw action 和目标越限，并更早触发 self-contact。

**有证据支持的原因**

- `base_contact` 监测集合包含大量非足部 link，且启用了 self-collision；自碰撞会被判成非法接触。
- model_200 的动作分布/目标范围异常，是 5.72 步退化的直接伴随因素；但它与终止判据的因果贡献需要受控修复实验确认。
- 随机 reset/命令会改变触发时间，但不是唯一原因；nominal 仍触发。

**待验证假设**

- 应否把 wrist/hip 的正常 self-contact 通过资产碰撞几何或终止监测集合处理掉，需要单独工程决策和回归测试；本轮没有改阈值或任务。
- 只保留 foot/ground 过滤并取消 self-collision 后，原始任务能否维持 2 s，需要新诊断或修复后 pilot 验证。
- 更长训练预算能否学会避免自碰撞，当前没有证据支持或否定。

**已排除原因**

- 没有发现 sensor body name/robot body ID 的索引错配。
- 没有发现 reset 后 contact history 跨 episode 残留；reset 后为 0。
- 没有发现 model normalizer 混用、mean action 采样差异、joint order/scale 不一致、old log-prob 对错 transition 或 optimizer LR 变量不一致。
- 不能把 GPU filter 对 ground 的警告当成根因；它只限制对象分类，self-contact 过滤数据仍可读。

## 5. 下一步建议和明确回答

A. **是否找到实现错误？** 找到一个应记录并处理的语义风险：`fall` 被 evaluator 直接命名为 `base_contact`，而原 termination 实际包括非足部 self-contact；这不是 sensor 索引实现错误。PPO/normalizer 链路未发现已确认实现错误。

B. **是否需要修复后重跑相同 pilot？** 需要先做最小、独立的碰撞/termination 语义修复（明确排除正常 self-contact 或重建可区分的 ground-contact 判据），然后重跑相同 pilot；本轮不直接修改或重跑。

C. **是否可以保持原任务，仅增加训练预算？** 目前不建议只增加预算。model_200 已有目标越限且 nominal 也触发 self-contact；应先完成碰撞判据回归，再决定是否保持任务并增加预算。

D. **仍未确认什么？** 未确认所有 32 条原评估轨迹的每次 contact 对象；未确认一个仅由地面接触触发的样本；未确认碰撞几何修复后是否能 2 s 支撑、也未确认长预算是否可学习。nominal 的 0.24 s 只是短时工程排查。

## 6. 交付物与命令

诊断代码：[`e1_d0_probe.py`](../scripts/instinct_rl/e1_d0_probe.py)、[`e1_d0_run.py`](../scripts/diagnostics/e1_d0_run.py)、[`e1_d0_analyze.py`](../scripts/diagnostics/e1_d0_analyze.py)。归档根目录为 [`logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/`](../logs/e1-d0/diagnosis_20260922_171833_odoxrzbd/)。

完整命令、每次 exit/status、配置、runtime mapping、源码快照和 hash 在归档的 `manifest.json`、`commands.sh`、各 run 的 `env.yaml`/`agent.yaml` 与 `runtime.json`。实际运行包含 original zero/initial/model_200、按对象接触过滤、nominal 及逐项恢复条件、以及不超过 2 updates 的 PPO audit；没有覆盖 E0/E1 pilot 产物。
