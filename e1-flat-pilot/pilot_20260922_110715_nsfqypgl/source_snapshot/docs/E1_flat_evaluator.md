# 原始 Flat G1 最小 evaluator

入口：`scripts/instinct_rl/evaluate_flat.py`。必须使用专用 instinctlab Python，在新进程启动真实 Isaac Sim，传入精确 checkpoint 路径。只接受 `Instinct-Locomotion-Flat-G1-v0`，拒绝 `logs/e0` 下的 checkpoint，不自动选择 latest。

```bash
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/evaluate_flat.py \
  --checkpoint /absolute/path/to/E1/run/model_200.pt \
  --output-dir /absolute/path/to/new/evaluation-directory \
  --num-envs 8 --episodes-per-env 4 --max-steps 5000 \
  --seed 12345 --command-mode original --headless --device cuda:0
```

输出目录必须不存在。`episodes.json` 包含元数据、验收结果、逐环境 episode、reset 批次；`episodes.csv` 为同一组 episode 行；`env.yaml` 为实际评估配置。达到每个 env 的 episode 配额就结束，不要求全部 env 同时 done；配额后额外完成的 episode 不纳入统计。若步数上限内配额不足，保存已完成数据并记录 FAIL。避免只取最先结束的短 episode。

验收必须检查 JSON 的 `status == "PASS"`、完整 episode 配额和各核验字段，不能只检查退出码。当前 Isaac Sim 的 fast shutdown 可能在 `app.close()` 内以 0 结束进程，即使此前 evaluator 已捕获异常。JSON 在关闭前落盘；第一次 YAML 解析失败的记录也予以保留。导出的 YAML 使用仅增加 `builtins.slice` 数据解析的 FullLoader，不启用任意 Python 对象执行。

原始 command 分布：10 s 重采样；x∈[-0.5,1.0] m/s，y∈[-0.5,0.5] m/s，yaw∈[-1.5,1.5] rad/s，standing 比例 0.2，heading 比例 0.5。保留原任务的观测噪声、动力学随机化和 push。未使用 Flat-Play，不提供 fixed-command 模式。

模型与 policy/critic normalizer 均通过 runner 加载，逐项对比 checkpoint；全部置 eval 模式。动作使用 `act_inference` 的 Gaussian mean，与原 actor MLP 直接输出核对。整个 rollout 使用 `torch.inference_mode()`，不调用学习或 optimizer update；结束后逐张量验证 model 和两个 normalizer 完全未变。

每个物理 step 后，reward compute hook 在 command 更新和 auto-reset 前累计原始 reward、yaw-frame XY 速度误差和实际速度；reset hook 在 reset 前记录终止 flags、位置及 episode 统计。终止步计入 return/length；return 额外与环境每个 reward term 的 episode sum 之和交叉核对，length 与 raw episode buffer 核对。

| 字段 | 定义 |
| --- | --- |
| episode_id / env_id / env_episode_id | 本评估全局 ID、向量环境 ID、该 env 内 episode 序号（均从 0 开始） |
| episode_return / episode_length | 包含终止步的原始 dt-scaled reward 总和 / 控制步数；秒数=步数×0.02 |
| velocity_tracking_error | episode 平均 XY command 与实际速度的 L2 误差（m/s），使用 yaw-only 重力对齐坐标系 |
| velocity_tracking | episode 平均 exp(-XY 平方误差/0.5²)，与原 tracking reward 核一致 |
| base_contact | 原 `base_contact` illegal_contact term；包含躯干和非足部肢体，不能解释为仅躯干接地 |
| timeout | 原 `time_out` term；单独记录 |
| terminated / truncated | termination manager 的两个独立 flags，保留同一步同时为 true 的情况 |
| fall | 原 Flat 的 base_contact 失败；即使同一步 timeout 也保留 fall=true |
| actual_forward_distance | terminal root XY 净位移在 episode 初始 yaw 前向上的投影（m）；允许负值，不等于路径长度 |
| actual_velocity | episode 平均 yaw-frame root 前向速度（m/s），不使用 command 代替实际速度 |
| mean_speed_xy | episode 平均世界系水平速率（m/s） |
| command_mode / seed | 固定为 original / evaluator seed，默认 12345 |

随机侧向/转向/倒退 command 下，初始朝向投影距离不能作为单独成功判据。固定 seed 不保证跨 GPU/软件版本的物理逐 bit 一致。evaluator 通过表示统计和独立加载流程通过，不表示被评估策略已经学会行走。
