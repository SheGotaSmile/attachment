# InstinctLab 仓库逆向审计

审计日期：2026-09-16。目标仓库：/home/xiexuhui/InstinctLab；提交：ba28d3d2655b15a19b729476a630937a19610a3b。此次只新增本文，没有修改源码、配置、算法或现有实验产物，没有安装依赖或启动训练／仿真。

证据分为“源码确认”“CPU 局部验证”“待仿真验证”。链接指向本机文件及行号；同名文件以链接中的完整路径为准。目录和缺失项没有源码行号，使用 A.1 的现场清单及 F.1 的检索记录作为证据，不为不存在的文件编造行号。外部依赖明确标注，不能视作本仓库自带实现。

## A. 仓库结构摘要

本仓库提供 Isaac Lab 环境、任务配置、机器人资源、传感器与地形扩展。训练脚本把环境交给外部 Instinct-RL；真实机器人执行由 README 指向另一仓库 instinct_onboard。这个边界由实际导入和调用确定，见 [README.md:12](/home/xiexuhui/InstinctLab/README.md:12)、[train.py:73](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:73)、[train.py:204](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:204)、[README.md:20](/home/xiexuhui/InstinctLab/README.md:20)。

### A.1 顶层现场清单

以下为审计开始时 ls -la、git ls-files 和 rg --files -uu 的结果；docs/ 是本次新增的报告目录。

~~~text
InstinctLab/
├── .agents/                      本次现场为空
├── .codex/                       本次现场为空
├── .cursor/rules/                编辑器项目说明
├── .git/                         Git 元数据
├── .vscode/                      IDE 配置与工具
├── docker/                       Dockerfile、compose、.env.base、启动/附着/停止脚本
├── logs/instinct_rl/              本地实验配置、TensorBoard 事件和 Git diff
├── outputs/<日期>/<时间>/         Hydra 配置和日志
├── scripts/                      训练、播放、列任务、动作数据转换/可视化脚本
├── source/instinctlab/            可编辑安装的 Python 扩展
├── .flake8
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTORS.md
├── CONTRIBUTOR_AGREEMENT.md
├── DOCS.md
├── LICENSE
├── README.md
├── pyproject.toml
└── docs/repo_audit.md             本次唯一新增文件
~~~

主要源码目录及用途如下；每项给出代表性实现作为行号证据。

| 目录／主要文件 | 用途与证据 |
| --- | --- |
| source/instinctlab/setup.py、pyproject.toml、config/extension.toml | 包安装及扩展元数据：[setup.py:14](/home/xiexuhui/InstinctLab/source/instinctlab/setup.py:14)、[extension.toml:17](/home/xiexuhui/InstinctLab/source/instinctlab/config/extension.toml:17) |
| source/instinctlab/instinctlab/assets/ | G1 articulation、URDF、网格资源：[unitree_g1.py:536](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:536) |
| source/instinctlab/instinctlab/tasks/ | locomotion、parkour、shadowing；递归导入完成注册：[tasks/__init__.py:16](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/__init__.py:16) |
| source/instinctlab/instinctlab/envs/ | 环境子类与 MDP observations/actions/events/rewards/terminations/commands/curriculums：[manager_based_rl_env.py:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/manager_based_rl_env.py:12) |
| source/instinctlab/instinctlab/managers/、monitors/ | 多奖励组、指标监控：[reward_manager.py:18](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/managers/reward_manager.py:18)、[monitor_manager.py:115](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/monitors/monitor_manager.py:115) |
| source/instinctlab/instinctlab/terrains/ | height_field、trimesh、虚拟障碍、导入器：[terrain_importer.py:17](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_importer.py:17) |
| source/instinctlab/instinctlab/sensors/ | grouped ray caster、带噪深度相机、volume points：[grouped_ray_caster.py:26](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster.py:26)、[volume_points.py:1](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/volume_points/volume_points.py:1) |
| source/instinctlab/instinctlab/motion_reference/ | 动作文件、缓存、参考轨迹管理；Manager 本身继承 SensorBase：[motion_reference_manager.py:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/motion_reference/motion_reference_manager.py:34) |
| source/instinctlab/instinctlab/utils/ | RL wrapper、网络配置接口、噪声、历史缓冲、Warp raycast、运动学：[vecenv_wrapper.py:15](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:15) |
| source/instinctlab/instinctlab/actuators/、sim/ | actuators/ 目前只有导入和注释占位，本地 PD 实现仓库中未找到；资产实际使用 Isaac Lab 的 actuator：[actuator_pd.py:1](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/actuators/actuator_pd.py:1)、[actuator_cfg.py:1](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/actuators/actuator_cfg.py:1)、[unitree_g1.py:8](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:8)。sim/ 实现 mesh 转 USD 后生成资源：[from_files.py:23](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sim/spawners/from_files/from_files.py:23)。 |
| scripts/instinct_rl/ | train.py、play.py、cli_args.py、plotter.py；[train.py:21](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:21) |
| scripts/list_envs.py、multi_play.py、amass_visualize.py 等 | 列任务、批量播放和动作可视化：[list_envs.py:26](/home/xiexuhui/InstinctLab/scripts/list_envs.py:26)、[multi_play.py:6](/home/xiexuhui/InstinctLab/scripts/multi_play.py:6)、[amass_visualize.py:1](/home/xiexuhui/InstinctLab/scripts/amass_visualize.py:1) |
| README.md、DOCS.md、各任务 README | 安装说明、组件设计和任务用法：[README.md:34](/home/xiexuhui/InstinctLab/README.md:34)、[DOCS.md:88](/home/xiexuhui/InstinctLab/DOCS.md:88) |
| logs/、outputs/ | 运行产物，不属于核心源码且被忽略：[.gitignore:20](/home/xiexuhui/InstinctLab/.gitignore:20) |

### A.2 注册任务与现成环境

AST 静态枚举得到 14 个任务 ID，即下列 7 组训练／Play 配对；这不是运行时成功创建环境的证明。

| 训练任务 ID（对应 Play 变体在 G1 后加 -Play） | 已确认配置 | 注册证据 |
| --- | --- | --- |
| Instinct-Locomotion-Flat-G1-v0 | 平地速度跟踪；policy 无深度／高度扫描；训练默认 4096 环境 | [locomotion/config/g1/__init__.py:6](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py:6)、[flat_env_cfg.py:35](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:35)、[flat_env_cfg.py:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:97) |
| Instinct-Parkour-Target-Amp-G1-v0 | 非平地、深度历史、AMP/WasabiPPO、4 专家 | [parkour/config/g1/__init__.py:13](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py:13)、[parkour_env_cfg.py:290](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:290)、[instinct_rl_amp_cfg.py:32](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:32) |
| Instinct-BeyondMimic-Plane-G1-v0 | 平地动作模仿 | [beyondmimic/config/g1/__init__.py:14](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/config/g1/__init__.py:14)、[beyondmimic_env_cfg.py:1](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/beyondmimic_env_cfg.py:1) |
| Instinct-Shadowing-WholeBody-Plane-G1-v0 | 全身参考轨迹、默认 8 专家 MoE | [whole_body/config/g1/__init__.py:7](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/__init__.py:7)、[whole_body/agents/instinct_rl_ppo_cfg.py:107](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py:107)、[同文件:150](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py:150) |
| Instinct-Perceptive-Shadowing-G1-v0 | 动作匹配 mesh 地形、深度 actor、高度扫描 critic | [perceptive/config/g1/__init__.py:7](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/__init__.py:7)、[perceptive_env_cfg.py:69](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:69)、[同文件:242](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:242)、[同文件:301](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:301) |
| Instinct-Perceptive-Vae-G1-v0 | VAE 学生、教师策略蒸馏；配置 using_ppo=False | [perceptive/config/g1/__init__.py:29](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/__init__.py:29)、[instinct_rl_vae_cfg.py:80](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:80) |
| Instinct-Perceptive-HOI-Shadowing-G1-v0 | 感知与人／物交互参考任务配置、动态对象 raycast 目标 | [perceptive_hoi/config/g1/__init__.py:7](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive_hoi/config/g1/__init__.py:7)、[perceptive_shadowing_cfg.py:136](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive_hoi/config/g1/perceptive_shadowing_cfg.py:136) |

非平地、坡度和扰动不是待从零开发的功能：

| 检查项 | 结论与精确位置 |
| --- | --- |
| 随机起伏、台阶、沟隙、箱体 | Parkour 的 ROUGH_TERRAINS_CFG 已实例化这些子地形：[parkour_env_cfg.py:55](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:55)、[同文件:91](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:91)、[同文件:110](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:110)、[同文件:218](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:218) |
| 斜坡 | 已启用反向金字塔坡，proportion=0.10、slope_range=(0.0,0.7)。坡度是升高／水平距离比值，不是角度；生成公式为端点随 difficulty 线性插值：[parkour_env_cfg.py:261](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:261)、[hf_terrains.py:52](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:52)、[同文件:78](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:78)。它不是整张地形具有恒定法向的无限平面坡。 |
| 其他倾斜地形组件 | PerlinTiltedRampTerrainCfg 的 tilt_angle 明确以度表示；只有生成组件的存在证据，不代表另有注册的坡地 locomotion 任务：[hf_terrains_cfg.py:160](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains_cfg.py:160)。独立 Rough/Slope locomotion 注册任务：仓库中未找到，见上述注册表。 |
| 外部扰动 | Flat 训练每 10–15 秒把 x/y 速度扰动到 ±0.5，属于速度注入；reset 外力／力矩项虽然存在，当前范围全为零，不能报告成启用非零外力。Play 关闭这些项：[flat_env_cfg.py:281](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:281)、[同文件:313](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:313)、[同文件:397](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:397)。Perceptive 的 push 配置在基类中被注释：[perceptive_env_cfg.py:549](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:549)。 |
| 高度扫描 | Perceptive torso 上有 1.6×1.0 m、0.1 m 分辨率网格，进入 critic；Parkour 左右脚各有局部扫描器：[perceptive_env_cfg.py:109](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:109)、[同文件:301](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:301)、[parkour_env_cfg.py:318](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:318)。不要把脚扫描器自动计入 policy 输入。 |
| 动作模仿任务的推扰 | WholeBody 与 BeyondMimic 训练基类还配置了每 1–3 秒一次的线速度／角速度注入，包含 x/y、z、roll/pitch/yaw：[shadowing_env_cfg.py:367](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/shadowing_env_cfg.py:367)、[beyondmimic_env_cfg.py:350](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/beyondmimic_env_cfg.py:350)。同样属于速度扰动配置，不是非零力／力矩施加的证据。 |

## B. 关键文件表格

### B.1 功能定位

| 目标 | 定义／调用位置 | 结论 |
| --- | --- | --- |
| 机器人模型 | [assets/unitree_g1.py:536](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:536)；[g1_29dof_torsobase_popsicle.urdf:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf:65) | Flat 实际选择 Popsicle G1，URDF 解析得到 29 revolute + 10 fixed joints；默认位姿、碰撞、PD 在 asset 配置；不是根据“29DOF”名称估算。 |
| 观测空间 | [flat_env_cfg.py:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:97)、[同文件:114](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:114)；[vecenv_wrapper.py:194](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:194) | policy／critic 各自定义 ObsTerm；运行时 obs_format 保存分量名与形状，wrapper 将所有分量展平拼接。 |
| 动作空间 | [flat_env_cfg.py:64](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:64)、[同文件:364](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:364)、[unitree_g1.py:523](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:523) | 全关节位置动作；声明中的 scale=0.5 会被逐关节 beyondmimic_action_scale 覆盖；不是直接输出力矩。 |
| 动作处理实现 | [actions/action_cfg.py:3](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/actions/action_cfg.py:3)；外部 [joint_actions.py:169](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:169)、[同文件:197](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:197) | 基础 JointPositionAction 来自 Isaac Lab；仿射变换后写入 joint position target，再由 actuator／物理引擎执行。本地另有覆盖部分动作的扩展：[joint_actions.py:19](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/actions/joint_actions.py:19)。 |
| 奖励函数 | [flat_env_cfg.py:142](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:142)、[locomotion/mdp/rewards.py:25](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/mdp/rewards.py:25)、[同文件:47](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/mdp/rewards.py:47) | 速度跟踪、足端腾空／打滑、姿态、关节偏离、动作变化率、加速度和力矩等；通用项导入 Isaac Lab，局部项在本仓库。 |
| 多奖励及模仿奖励 | [managers/reward_manager.py:120](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/managers/reward_manager.py:120)、[perceptive_env_cfg.py:352](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:352)、[envs/mdp/rewards/motion_reference.py:1](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/rewards/motion_reference.py:1) | 支持按组 sum/prod 合成，多组奖励交给外部多价值网络。Parkour 目前一个 rewards 组：[parkour_env_cfg.py:799](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:799)。 |
| episode 终止／reset | [flat_env_cfg.py:232](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:232)、[同文件:290](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:290)、[manager_based_rl_env.py:44](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/manager_based_rl_env.py:44) | Flat 因超时／非法接触重置；局部 env 只加 monitor reset，实际调度在父类。轨迹任务额外使用 [events/motion_reference.py:62](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/events/motion_reference.py:62)。 |
| 地形生成 | [terrain_importer.py:72](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_importer.py:72)、[terrain_generator.py:21](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_generator.py:21)、[hf_terrains.py:18](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:18)、[mesh_terrains.py:21](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/trimesh/mesh_terrains.py:21) | Perlin 高度场转 mesh、预制 mesh、动作匹配地形；FiledTerrainGenerator 保存子地形配置。规则网格生成和环境原点分配部分继承 Isaac Lab。 |
| 训练入口 | [scripts/instinct_rl/train.py:121](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:121)、[同文件:230](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:230) | Hydra 绑定任务与 agent 配置，创建 Gym env／wrapper／runner，调用 learn。 |
| PPO／其他算法 | [locomotion/agents/instinct_rl_ppo_cfg.py:20](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:20)、[instinct_rl_amp_cfg.py:43](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:43)、[instinct_rl_vae_cfg.py:80](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:80) | 本仓库有 PPO／WasabiPPO／VaeDistill 配置；优化器及损失实现：仓库中未找到，实际在外部依赖，见 B.2。 |
| 策略／价值网络 | [rl_cfg.py:10](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:10)、[同文件:94](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:94)、[module_cfg.py:39](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/module_cfg.py:39) | 本地定义 MLP／recurrent／encoder／MoE／VAE 的配置协议；网络 forward 实现：仓库中未找到，实际在 instinct_rl.modules。 |
| 随机化配置 | [flat_env_cfg.py:260](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:260)、[perceptive_env_cfg.py:430](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:430)、[events/randomization.py:22](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/events/randomization.py:22) | 启动时材质、质量、质心、PD gains、关节零位、相机安装误差；reset 状态随机；interval 推扰；观测 noise、相机 noise_pipeline、actuator delay 分属不同层。 |
| evaluation／checkpoint | [play.py:101](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:101)、[同文件:160](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:160)、[train.py:173](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:173) | 通用 play 加载并播放，train 可恢复；shadowing 专用脚本有成功率统计但存在适用性限制，见 E.2。 |
| 传感器模拟 | [grouped_ray_caster_camera.py:140](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster_camera.py:140)、[noisy_grouped_raycaster_camera.py:44](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/noisy_grouped_raycaster_camera.py:44)、[noisy_camera.py:102](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/noisy_camera.py:102) | 动态多 mesh 分组射线、深度投影、噪声流水线、独立环境历史；接触传感器来自 Isaac Lab。 |
| LiDAR | [grouped_ray_caster.py:26](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster.py:26) 是可复用底层 | 专门的激光雷达配置、观测项、点云策略和真实驱动：仓库中未找到。通用 ray caster／高度扫描不等于已完成 LiDAR 接入，检索范围见 F.1。 |
| 真实机器人部署接口 | [README.md:20](/home/xiexuhui/InstinctLab/README.md:20)、[play.py:171](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:171)、[parkour/scripts/onnxer.py:14](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/scripts/onnxer.py:14) | 有导出 ONNX、在仿真中用 ONNX Runtime 回放的接口；硬件通信、实时状态读取、关节下发、部署状态机：仓库中未找到。README 明确交给外部 instinct_onboard。 |

### B.2 外部依赖实际源码边界

本机 /home/xiexuhui/miniconda3/envs/instinctlab/bin/python 的包定位结果是 /home/xiexuhui/IsaacLab-Instinct 和 /home/xiexuhui/instinct_rl；不是另一份 /home/xiexuhui/IsaacLab。前者提交 f73c33173801f5f8afea4142482e47b7710c2b75，与 [README.md:36](/home/xiexuhui/InstinctLab/README.md:36) 一致；后者提交 ba45ed231ebbf0a4099cd31d607e2886814fd165。版本现场记录见 F.1。

| 外部文件 | 实际职责 |
| --- | --- |
| [instinct_rl/runners/on_policy_runner.py:62](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:62) | 用 obs_format、num_actions、num_rewards 构建网络和算法；:111 开始 learn，:207 单次 rollout，:433 load，:474 inference policy，:487 ONNX 导出。 |
| [instinct_rl/algorithms/ppo.py:137](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/ppo.py:137) | act 采样并计算 value／log probability；:151 保存 transition、超时 bootstrap、重置 recurrent 状态；:183 算 returns；:187 update；:268 PPO clipped surrogate；:273 value loss。 |
| [instinct_rl/storage/rollout_storage.py:79](/home/xiexuhui/instinct_rl/instinct_rl/storage/rollout_storage.py:79) | rollout 张量存储；:153 returns/GAE；:179 minibatch。 |
| [instinct_rl/algorithms/wasabi.py:312](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/wasabi.py:312) | WasabiPPO 组合判别器逻辑与 PPO。 |
| [instinct_rl/algorithms/vae_distill.py:53](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/vae_distill.py:53)、[tppo.py:78](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/tppo.py:78) | VaeDistill／TPPO 的教师构建、加载、蒸馏；教师加载入口在 tppo.py:100。 |
| [instinct_rl/modules/actor_critic.py:43](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:43) | actor、critic MLP，Gaussian action distribution；:170 确定性推理，:174 value。 |
| [instinct_rl/modules/encoder_actor_critic.py:29](/home/xiexuhui/instinct_rl/instinct_rl/modules/encoder_actor_critic.py:29) | 依据 obs segments 构建 encoder；:88/:92 编码后送入 actor。 |
| [instinct_rl/modules/moe_actor_critic.py:38](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:38)、[moe.py:48](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:48) | actor 和 critic 各有专家与门控；softmax gate 后对全部专家输出加权求和。当前是 dense output mixture，不是只计算 top-k 专家的实现。 |
| [Isaac Lab manager_based_rl_env.py:173](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:173)、[同文件:349](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:349) | 环境 step 主循环与按 env_ids 重置。 |

## C. 训练与推理调用链

### C.1 从启动到环境 step

图中 I/R/L 是下表的路径缩写；这样可直接区分仓库内代码和外部实现。

~~~mermaid
flowchart TD
    A[train.py:63 AppLauncher] --> B[train.py:103 导入任务注册]
    B --> C[train.py:121 Hydra 解析 env 和 agent]
    C --> D[train.py:183 gym.make]
    D --> E[I/envs/manager_based_rl_env.py:17 load_managers]
    E --> F[I/utils/wrappers/instinct_rl/vecenv_wrapper.py:21 wrapper + reset]
    F --> G[R/runners/on_policy_runner.py:62 构建网络/算法/storage]
    G --> H[train.py:211 可选恢复 checkpoint]
    H --> J[R/runners/on_policy_runner.py:111 learn]
    J --> K[R/runners/on_policy_runner.py:207 rollout_step]
    K --> M[R/algorithms/ppo.py:137 act + value]
    M --> N[I/utils/wrappers/instinct_rl/vecenv_wrapper.py:162 step]
    N --> P[I/envs/manager_based_rl_env.py:37 step]
    P --> Q[L/envs/manager_based_rl_env.py:173 父类 step]
    Q --> S[动作处理 → decimation 次物理步 → 终止/奖励 → 局部 reset → command/event → observation]
    S --> T[I/envs/manager_based_rl_env.py:39 monitor]
    T --> U[wrapper 展平观测、奖励变 N×G、生成 dones]
    U --> V[R/runners/on_policy_runner.py:217 normalize + process_env_step]
    V --> K
    V --> W[收满 rollout 后 returns/GAE → PPO update → checkpoint]
    W --> J
~~~

| 图中代号／步骤 | 路径与行号证据 |
| --- | --- |
| I | /home/xiexuhui/InstinctLab/source/instinctlab/instinctlab；环境入口：[manager_based_rl_env.py:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/manager_based_rl_env.py:12) |
| R | 外部 /home/xiexuhui/instinct_rl/instinct_rl；[on_policy_runner.py:50](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:50) |
| L | 外部 /home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab；[manager_based_rl_env.py:173](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:173) |
| 应用、注册、配置、创建环境 | [train.py:63](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:63)、[同文件:103](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:103)、[同文件:121](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:121)、[同文件:183](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:183) |
| wrapper／runner／训练启动 | [train.py:201](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:201)、[同文件:230](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:230) |
| rollout 返回后的处理 | [on_policy_runner.py:207](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:207)、[ppo.py:151](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/ppo.py:151) |
| returns、优化和保存 | [on_policy_runner.py:188](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:188)、[同文件:195](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:195)、[同文件:205](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:205) |

父类 step 的确切顺序为：处理动作（:173）；重复 decimation 次 apply_action → scene.write_data_to_sim → sim.step → 按需 render → scene.update（:182）；计算 terminated/truncated（:204）和 reward（:208）；对 reset_env_ids 调用 _reset_idx（:216）；更新 command（:232）、interval events（:235）；最后计算带 history 更新的 observation（:238）。依据：[外部 manager_based_rl_env.py:173](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:173)。

reset 的顺序为本地 monitor.reset → 父类 curriculum.compute → scene.reset → reset events → 各 manager.reset → episode_length_buf 清零。Flat reset events 写随机根状态和关节状态；reference reset 则从运动参考读取姿态、速度和关节状态写回 simulator。依据：[本地 manager_based_rl_env.py:44](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/manager_based_rl_env.py:44)、[父类:349](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:349)、[flat_env_cfg.py:290](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:290)、[events/motion_reference.py:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/events/motion_reference.py:97)。

因此 step 返回的已结束环境的观测是 reset 后的下一观测；dones 对应刚结束的 transition，wrapper 另传 time_outs。不能把这一 observation 当作终止前的状态。依据：[父类:215](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:215)、[wrapper:167](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:167)。

### C.2 推理、评估与导出

~~~text
scripts/instinct_rl/play.py
  AppLauncher(:51)
  → parse_env_cfg + parse_instinct_rl_cfg(:92)
  → --load_run / --checkpoint → get_checkpoint_path(:101)
  → gym.make(:128) → InstinctRlVecEnvWrapper(:157)
  → OnPolicyRunner(:160) → runner.load(:163)
  → get_inference_policy(:169)
      外部 runner:474：eval → policy normalizer → actor_critic.act_inference
  → [--exportonnx] runner.export_as_onnx(:179)，要求 N=1(:175)
  → while simulation_app.is_running(:185)
      actions = policy(obs)(:189)
      obs, rewards, dones, infos = env.step(actions)(:193)
~~~

证据：[play.py:89](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:89)、[play.py:159](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:159)、[外部 runner:474](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:474)。通用 play 主要是回放，仓库中未找到适用于全部任务、按独立 episode 汇总并输出标准评估表的统一 evaluator。

Shadowing 专用 play 另有参考位置误差、XY≤1.0 m／Z≤0.1 m 成功判据和累计 100 条停止逻辑，但只在同一步全部环境 done 时统计，并依赖特定 monitor 名称。见 [shadowing/play.py:261](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/play.py:261)、[同文件:298](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/play.py:298)。这段不能直接作为通用并行评估的依据。

导出路径是实验目录/exported；外部 runner 另将 normalizer 导出为 policy_normalizer.npz，actor 网络独立导出。Parkour ONNX loader 加载 0-depth_encoder.onnx 与 actor.onnx，并把 torch Tensor 转成 NumPy 再转回。见 [runner:487](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:487)、[actor_critic.py:202](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:202)、[onnxer.py:19](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/scripts/onnxer.py:19)。

## D. 关键张量形状

### D.1 具体示例：Instinct-Locomotion-Flat-G1-v0

令 N 为并行环境数，J=29。训练配置默认 N=4096，Play 默认 N=1，命令行可覆盖。URDF 实测 29 个 revolute joints，观测和动作使用全部关节。依据：[flat_env_cfg.py:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:65)、[同文件:344](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:344)、[同文件:391](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:391)、[URDF:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf:65)。

| 顺序／分量 | policy | critic | 证据 |
| --- | --- | --- | --- |
| base_lin_vel | 无 | (N,3)，critic 第一项 | [flat_env_cfg.py:115](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:115) |
| base_ang_vel | (N,3) | (N,3) | [同文件:98](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:98) |
| projected_gravity | (N,3) | (N,3) | [同文件:99](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:99) |
| velocity_commands | (N,3)：vx、vy、yaw rate | (N,3) | [同文件:77](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:77)、[同文件:103](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:103) |
| joint_pos | (N,29)，相对默认角度 | (N,29) | [同文件:104](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:104) |
| joint_vel | (N,29) | (N,29) | [同文件:105](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:105) |
| actions | (N,29)，上一步原始动作 | (N,29) | [同文件:106](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:106)、[同文件:123](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:123) |
| wrapper 输出 | (N,96) | infos.observations.critic 为 (N,99) | [vecenv_wrapper.py:121](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:121)、[同文件:222](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:222) |

policy 展平总维数为 3+3+3+29+29+29=96；critic 多 3 维线速度，总维数 99。heading command 的内部目标不增加第四个 policy command 维度。该任务没有配置 history_length，不能套用感知任务的 8 帧历史。配置依据为上表及 [flat_env_cfg.py:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:97)。

网络为 actor：96→256→128→128→29，critic：99→256→128→128→1；中间激活 ELU。训练时从逐动作 Gaussian 采样，std 参数形状 (29,)；确定性推理输出均值 (N,29)，critic 返回 (N,1)。已使用外部实际 ActorCritic 类在 CPU 上执行 N=4 的前向，得到 (4,29)／(4,1)，但没有创建 Isaac Sim 环境。依据：[instinct_rl_ppo_cfg.py:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:12)、[外部 actor_critic.py:112](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:112)、[同文件:159](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:159)。

动作语义为 q_target = default_joint_pos + scale[j] × action[j]；scale[j] 来自 0.25 × effort_limit_sim[j] / stiffness[j]。网络输出没有因此天然限制在 [-1,1]，当前 Flat 配置也没有显式动作 clip。最终动作范围还受关节／actuator／仿真约束影响。见 [unitree_g1.py:523](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:523)、[外部 joint_actions.py:169](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:169)、[actor_critic.py:119](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:119)。

Flat 物理步长 0.005 s、decimation=4，因此控制周期为 0.02 s（50 Hz），20 s episode 对应最多 1000 个环境 step。PPO 每次采集 T=24 步，因此 storage 中 policy=(24,N,96)、critic=(24,N,99)、actions=(24,N,29)、reward/value/return=(24,N,1)，而 wrapper dones=(N,)；storage dones 才是 (24,N,1)。依据：[flat_env_cfg.py:358](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:358)、[instinct_rl_ppo_cfg.py:48](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py:48)、[wrapper:162](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:162)、[外部 storage:90](/home/xiexuhui/instinct_rl/instinct_rl/storage/rollout_storage.py:90)。

### D.2 感知、历史与专家张量

| 对象 | 形状／推导 | 证据与边界 |
| --- | --- | --- |
| 原始相机输出 | (N,H,W,1)；visualizable_image 转为 (N,1,H,W) | [exteroception.py:87](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:87) |
| 感知 Shadowing 深度 | 原始 (N,27,48,1)，归一化、裁剪并 resize 为 (N,18,32,1)，policy term=(N,1,18,32)，576 个标量 | [perceptive_env_cfg.py:137](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:137)、[同文件:164](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:164) |
| 高度扫描 | (N,187)，由 17×11 个网格采样点得到；ray hits=(N,187,3)。这是相对高度扫描，不是全局地图文件 | [perceptive_env_cfg.py:109](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:109)、[外部 patterns.py:45](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns.py:45)、[外部 observations.py:292](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/observations.py:292) |
| 感知 Shadowing 本体历史 | gravity／angular velocity 各 (N,24)；joint_pos／joint_vel／last_action 各 (N,232)，即 8 帧 | [perceptive_env_cfg.py:50](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:50)、[同文件:249](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:249) |
| 感知 Shadowing 默认整体输入 | 10 帧 reference：290+290+30+60=670；本体历史 744；深度 576；actor 展平 1990，深度编码 32 后 actor 主干输入 1446。critic：610 reference + 126 link pose +187 scan+744 history=1667 | [perceptive_shadowing_cfg.py:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py:65)、[同文件:86](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py:86)、[perceptive_env_cfg.py:224](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:224)、[同文件:283](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:283)、[encoder 配置:19](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_ppo_cfg.py:19)。为静态推导；动作数据和配置覆盖可能改变结果，应以运行时 get_obs_format 为准。 |
| Parkour 深度历史 | 原始 36×64，裁掉上 18／左右各16后为18×32；保留37帧，抽取8帧、间隔5、延迟0或1，term=(N,8,18,32) | [parkour_env_cfg.py:352](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:352)、[同文件:387](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:387)、[同文件:448](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:448)、[exteroception.py:139](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:139) |
| Parkour policy／critic | 本体与command为8×96=768／8×99=792；加4608深度得5376／5400；深度 encoder 输出128后为896／920 | [parkour_env_cfg.py:417](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:417)、[同文件:470](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:470)、[instinct_rl_amp_cfg.py:12](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:12)。静态推导，未运行相机。 |
| Parkour 4 专家 | gate=(N,4)，actor expert outputs=(N,4,29)，加权 action=(N,29)；单奖励组 critic expert outputs=(N,4,1)，value=(N,1) | [instinct_rl_amp_cfg.py:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:34)、[外部 moe.py:48](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:48)、[moe_actor_critic.py:51](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:51) |

## E. 推荐修改点（本次不实施）

### E.1 功能接入位置

下表覆盖配置、数据获取、网络、训练／导出接口；优先复用已存在的实现。这里的“推荐修改”是后续工作位置，不表示本次已改动。

| 功能 | 所有主要接入层及文件行号 | 建议与约束 |
| --- | --- | --- |
| 坡度参数 | 地形参数：[parkour_env_cfg.py:261](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:261)；生成公式：[hf_terrains.py:78](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:78)；难度课程：[parkour/mdp/curriculums.py:15](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/mdp/curriculums.py:15)；子地形元数据：[terrain_generator.py:21](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_generator.py:21)；观测挂点：[flat_env_cfg.py:97](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:97) | 区分地形几何坡度、critic 特权坡度和部署可测局部坡度；现有 policy 无显式坡度标量。Flat 若切换坡地还应审查 flat_orientation_l2，见 [flat_env_cfg.py:172](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:172)。 |
| 地形高度图 | 生成高度场：[hf_terrains.py:18](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/height_field/hf_terrains.py:18)；scene scanner：[perceptive_env_cfg.py:109](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:109)；观测：[同文件:301](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:301)；CNN 参数：[module_cfg.py:39](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/module_cfg.py:39)；张量格式：[wrapper:208](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:208) | 已有局部高度扫描。若要 CNN 输入，保留网格形状并与射线 ordering 对齐；若需要全局在线 elevation map／融合真实点云的地图模块，仓库中未找到。 |
| 深度相机 | scene 内外参／刷新：[perceptive_env_cfg.py:117](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:117)；动态遮挡 mesh：[perceptive_shadowing_cfg.py:111](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py:111)；噪声：[noisy_camera.py:37](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/noisy_camera.py:37)；观测：[exteroception.py:47](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:47)；编码：[perceptive/agents/instinct_rl_ppo_cfg.py:19](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_ppo_cfg.py:19)；ONNX 接口：[onnxer.py:14](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/scripts/onnxer.py:14) | 完整仿真路径已存在；迁移到 Flat 时需成套增加 scene sensor、ObsTerm、encoder config，同时核对单位、裁剪、归一化、坐标和 history。 |
| 激光雷达 | raycast 底层：[grouped_ray_caster.py:77](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster.py:77)；分组：[同文件:82](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster.py:82)；scene 示例：[perceptive_env_cfg.py:109](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:109)；观测扩展：[exteroception.py:47](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:47)；encoder 协议：[module_cfg.py:8](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/module_cfg.py:8) | 需定义扫描 pattern、range／point／valid mask、时间与坐标约定，再决定固定维度编码；可复用 RayCaster，但真实驱动与专用 policy 尚不存在。 |
| 多专家策略 | 通用配置：[rl_cfg.py:94](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:94)、[同文件:148](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:148)；4专家模板：[instinct_rl_amp_cfg.py:32](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:32)；8专家模板：[whole_body/agents/instinct_rl_ppo_cfg.py:107](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py:107)；实际模型：[外部 moe_actor_critic.py:9](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe_actor_critic.py:9) | 可以复用当前 MoE。若改变专家计算或权重共享，修改点属于外部 instinct_rl，不应假设本仓库有隐藏网络实现。 |
| 专家门控网络 | gate 层数配置：[rl_cfg.py:100](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:100)；现成128/64门控：[whole_body/agents/instinct_rl_ppo_cfg.py:113](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py:113)；gate forward：[外部 moe.py:27](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:27)、[同文件:48](/home/xiexuhui/instinct_rl/instinct_rl/modules/moe.py:48) | 门控已存在；当前 gate 与专家读取同一输入，actor／critic 分开。只让 gate 读坡度／视觉、添加监督或门控损失，需要明确扩展外部网络／算法；本仓库没有已实现的专门“地形类别门控”。 |
| 教师与学生 | student obs：[perceptive_vae_cfg.py:102](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_vae_cfg.py:102)；student 模型：[instinct_rl_vae_cfg.py:49](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:49)；教师／损失／路径：[同文件:80](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:80)、[同文件:98](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:98)、[同文件:150](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:150)；实现：[外部 tppo.py:78](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/tppo.py:78) | 已有蒸馏路径。教师 obs_format 写死，需与教师 checkpoint、输入排列、normalizer 一起校验；不能仅改 num_actions 或换 checkpoint 路径。 |
| 观测历史 | 本体 history_length：[parkour_env_cfg.py:417](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:417)；相机历史：[同文件:395](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:395)；采样与延迟：[exteroception.py:110](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:110)；缓冲：[async_circular_buffer.py:8](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/buffers/async_circular_buffer.py:8)；recurrent 备选：[rl_cfg.py:30](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py:30) | 已有时间历史；expanded_joint_pos 等只是沿新维度复制当前值，不能替代真实历史，见 [expanded.py:56](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/expanded.py:56)。需要同步核对 reset、采样间隔、网络维数和部署填充方式。 |
| 随机化参数 | Events：[flat_env_cfg.py:260](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:260)、[perceptive_env_cfg.py:430](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:430)；实现：[randomization.py:22](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/events/randomization.py:22)；图像噪声：[noisy_camera.py:37](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/noisy_camera.py:37)；执行器：[unitree_g1.py:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:65)；种子：[train.py:131](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:131) | 先确定 startup/reset/interval 生命周期和按环境采样范围；若把采样参数输入 critic，需新增明确 ObsTerm。现有随机化配置不等于已把随机参数作为网络输入。 |

### E.2 并行训练、传感器与部署问题

基础设计已经支持批量训练：Flat／Parkour 默认 4096 环境，wrapper 所有主要输出保留 N 维，train 也提供分布式分支。不能得出“仓库不支持多环境”的结论。依据：[flat_env_cfg.py:344](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:344)、[wrapper:162](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:162)、[train.py:136](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:136)。下列问题有明确触发条件，并非所有默认任务都会触发。

| 优先级／验证级别 | 问题、触发条件与影响 | 证据和后续修改位置 |
| --- | --- | --- |
| 高／CPU 已复现 | SensorDeadNoiseModel 用全局 buffer mask 写入只属于 env_ids 子集的 data。N=4、env_ids=[1,3]、dead_probability=0 时左侧形状(4,2,2,1)，右侧(2,2,2,1)，抛 RuntimeError。异步传感器刷新或局部 reset 后更新会触发；默认相机未启用该噪声，不能归咎所有训练。 | [noise_model.py:681](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/noise/noise_model.py:681)、[同文件:706](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/noise/noise_model.py:706)。应按选中全局 env_ids 写回。 |
| 中／CPU 已复现 | AsyncCircularBuffer.append(data) 不传 batch_ids 时走父类，让 _pointer 保持 int；随后访问 buffer 调 get_by_batch_ids 并执行 shifts[batch_ids]，抛 TypeError。显式传 torch.arange(N) 的路径通过，默认 noisy camera 使用的正是显式路径。 | [async_circular_buffer.py:18](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/buffers/async_circular_buffer.py:18)、[同文件:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/buffers/async_circular_buffer.py:34)、[noisy_camera.py:121](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/noisy_camera.py:121)。新增历史调用方需避免踩中该接口分歧。 |
| 高／源码确认 | Shadowing 成功率仅在所有环境同一步结束时累加；不同步终止时会漏记 episode，甚至无法达到100条停止条件。误差在 env.step 前读取 monitor；也不是统一的 terminal-state 误差采集。若 teacher_policy 存在，则实际 step 使用 teacher_actions，不能把结果算作 student 性能。 | [shadowing/play.py:261](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/play.py:261)、[同文件:274](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/play.py:274)、[同文件:298](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/play.py:298)。需按每个 done 的 env_id 和被评估策略独立统计。 |
| 高／源码确认 | 感知任务的 MOTION_FOLDER 是占位符；Parkour 要求个人目录下的动作 YAML；VAE 教师目录写死。直接照命令不能保证完成初始化。 | [perceptive_shadowing_cfg.py:28](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py:28)、[g1_parkour_target_amp_cfg.py:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py:34)、[instinct_rl_vae_cfg.py:150](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:150)。路径应参数化，并提前验证资源。 |
| 高／源码确认的容量风险 | 默认 Parkour N=4096、T=24；仅 actor/critic 观测 rollout 就约 24×4096×(5376+5400)×4=3.95 GiB，尚未包括相机历史、AMP、梯度、物理和 encoder 激活。单份37帧18×32 float32历史约333 MiB，代码还保留环形与输出历史并产生 clone。是否 OOM 取决于显存，未实测。 | [parkour_env_cfg.py:395](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:395)、[同文件:902](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:902)、[instinct_rl_amp_cfg.py:77](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:77)、[外部 storage:90](/home/xiexuhui/instinct_rl/instinct_rl/storage/rollout_storage.py:90)、[noisy_camera.py:102](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/noisy_camera.py:102)、[async_circular_buffer.py:23](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/buffers/async_circular_buffer.py:23)。先用小 N 测量。 |
| 中／源码确认 | GroupedRayCaster 每次 _update_mesh_transforms(env_ids) 实际读取全部 transforms，增加局部更新成本；camera 将子集 raycast 结果整体赋给 self.ray_hits_w，可能使该公开缓冲的首维由 N 变成子集大小。depth output 本身按 env_ids 写回，不能据此宣称默认深度输出必坏。 | [grouped_ray_caster.py:122](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster.py:122)、[grouped_ray_caster_camera.py:104](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster_camera.py:104)、[同文件:148](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster_camera.py:148)。需仿真验证子集更新及 ray_hits 使用方。 |
| 中／源码确认 | NoisyCameraMixin.__str__ 直接遍历 noise_pipeline 字典并解包成二元组，键为 normalize 等字符串时会 ValueError；然后访问 config 实例的 __name__ 也不可靠。打印传感器即可触发，属于诊断接口缺陷。 | [noisy_camera.py:19](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/noisy_camera.py:19)、[perceptive_env_cfg.py:164](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py:164)。 |
| 中／源码确认 | visualizable_image 文档说 skip=2 取0,3,6，代码实际用 ::2；VAE 最终覆盖为3时取0,3,6,9。history_skip_frames 名称、文档与实现不一致，会导致训练与部署采样不同。函数文档还声称处理 RGB／inf，实际此函数仅 clone/转置，归一化在相机 noise_pipeline。 | [exteroception.py:73](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:73)、[同文件:94](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:94)、[perceptive_vae_cfg.py:213](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_vae_cfg.py:213)。 |
| 中／源码确认 | randomize_ray_offsets 接受 distribution，但始终调用 sample_uniform；设置 gaussian/log_uniform 不会得到声明分布。 | [events/randomization.py:62](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/events/randomization.py:62)、[同文件:87](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/events/randomization.py:87)。 |
| 中／源码确认 | FiledTerrainGenerator.get_subterrain_cfg 使用 torch.Tensor，但模块未 import torch；该方法一旦调用会 NameError，妨碍按环境查询坡度元数据。不是现有生成入口必然失败。 | [terrain_generator.py:1](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_generator.py:1)、[同文件:42](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/terrain_generator.py:42)。 |
| 中／源码确认的并发风险 | task 包全量递归导入；whole_body／beyondmimic 模块导入时写固定 /tmp/<MOTION_NAME>.yaml。同主机多任务／多 rank 可能互相覆盖，且“列环境”也有写文件副作用。 | [tasks/__init__.py:16](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/__init__.py:16)、[plane_shadowing_cfg.py:170](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/plane_shadowing_cfg.py:170)、[beyondmimic_plane_cfg.py:39](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/config/g1/beyondmimic_plane_cfg.py:39)。 |
| 中／源码确认 | DDP 初始化把 app_launcher.local_rank 传作 rank，多节点时 local rank 不能代替全局 RANK；auto_affinity 又用全局 RANK 乘单机核数分块。单机多 GPU 与多节点支持应分开验证。非分布式 --device 只显式覆盖 env_cfg.sim.device，未同步 agent_cfg.device；CLI --experiment_name 被声明但未应用。 | [train.py:112](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:112)、[同文件:134](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:134)、[同文件:138](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:138)、[cli_args.py:21](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:21)、[同文件:66](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:66)。 |
| 高／部署边界确认 | 本仓库没有硬件闭环；ONNX 导出不足以保证关节顺序、传感器坐标／时间同步、normalizer、action scale／offset、历史填充与真实设备一致。现有 actor 导出也不包含 q_target 仿射处理。 | [README.md:22](/home/xiexuhui/InstinctLab/README.md:22)、[外部 actor_critic.py:202](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:202)、[unitree_g1.py:17](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py:17)、[wrapper:199](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:199)。部署协议需在 instinct_onboard 核实。 |
| 中／源码确认 | ONNX 导出要求 N=1；基础 actor exporter 没声明 dynamic axes；Parkour loader 每步 CPU/NumPy 往返且未读取 normalizer。当前 Parkour runner 没配置 normalizers，因此不能声称其默认输出必错；若复用到带 normalizer 的策略则不完整。通用 --sample 路径也绕过 get_inference_policy 的 normalizer。 | [play.py:166](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:166)、[同文件:175](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:175)、[onnxer.py:23](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/scripts/onnxer.py:23)、[外部 runner:479](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:479)。 |
| 中／源码确认 | 专用 Parkour play 无条件取得 app_window、keyboard，即使未开启 keyboard_control；无窗口运行存在失败风险。通用 play 录完视频又调用外部 code 命令，缺少 VS Code CLI 时会在收尾报错；video_start_step>0 时主循环仍按 video_length 提前退出。 | [parkour/scripts/play.py:223](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/scripts/play.py:223)、[通用 play.py:133](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:133)、[同文件:205](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:205)、[同文件:214](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:214)。 |
| 能力缺口／源码确认 | GroupedRayCasterCamera 明确不支持 RGB／语义分割；NoisyTiledCamera 文件仅有 import，导出也被注释。不能依据文件名认定已有 RGB tiled camera 加噪实现。 | [grouped_ray_caster_camera.py:35](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster_camera.py:35)、[noisy_tiled_camera.py:1](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/noisy_tiled_camera.py:1)、[noisy_camera/__init__.py:8](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/noisy_camera/__init__.py:8)。 |

### E.3 安装、训练、评估和可视化命令完整性

结论是“基本入口齐全，首次复现与通用评估说明不完整”，不是所有任务都能直接复制 README 命令成功。

| 环节 | 完整程度与问题 | 依据 |
| --- | --- | --- |
| 安装 | 给出 Isaac Sim 5.1、Isaac Lab 提交、外部 instinct_rl、pip editable 安装顺序；没有固定 instinct_rl 提交或完整环境锁。setup.py 未列出 Isaac Lab／instinct_rl，依赖 README 手动安装；普通 wheel 打包只声明顶层包，不能把 editable 成功等同于 wheel 完整。 | [README.md:36](/home/xiexuhui/InstinctLab/README.md:36)、[同文件:38](/home/xiexuhui/InstinctLab/README.md:38)、[同文件:58](/home/xiexuhui/InstinctLab/README.md:58)、[setup.py:14](/home/xiexuhui/InstinctLab/source/instinctlab/setup.py:14)、[同文件:30](/home/xiexuhui/InstinctLab/source/instinctlab/setup.py:30) |
| 训练 | train 支持 num_envs、max_iterations、seed、logroot、video、Hydra 覆盖。顶层 README 示例却使用 WholeBody 的 Play 配置且依赖外部动作数据，不适合作为最小入门训练。建议先用无动作数据依赖的 Flat 训练任务。 | [train.py:23](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:23)、[README.md:64](/home/xiexuhui/InstinctLab/README.md:64)、[plane_shadowing_cfg.py:186](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/plane_shadowing_cfg.py:186) |
| 恢复训练 | 入口完整，但 --load_run 不会自动把 resume 改成 True；应明确同时传 --resume。 | [cli_args.py:69](/home/xiexuhui/InstinctLab/scripts/instinct_rl/cli_args.py:69)、[train.py:173](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:173) |
| 通用播放 | 必须提供 --load_run，或显式 --no_resume；--checkpoint 是 run 内的文件匹配，不是替代 --load_run 的独立加载入口。--env_cfg 会覆盖前面已应用 num_envs/device 的对象，需注意保存配置优先级。 | [play.py:101](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:101)、[同文件:120](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:120) |
| 评估 | 有专用 Shadowing 成功率和 grid_search.sh；脚本 run 名称写死、统计有同步结束假设；统一 benchmark 命令和输出格式仓库中未找到。 | [shadowing/play.py:298](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/play.py:298)、[grid_search.sh:7](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/grid_search.sh:7) |
| 文档过时 | BeyondMimic README 指向不存在的 beyondmimic/play.py；multi_play.py 默认 ID 不在14个注册任务中，必须显式传 --task。 | [beyondmimic/README.md:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/README.md:65)、[multi_play.py:43](/home/xiexuhui/InstinctLab/scripts/multi_play.py:43)，并对照 A.2 与 F.1 的文件检索。 |
| 可视化 | GUI play、视频、相机 debug_vis、motion visualization 都有入口；缺少统一无窗口有限步数 smoke 命令，视频模式有 code 依赖，debug image 使用 OpenCV 窗口。 | [play.py:16](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:16)、[play.py:185](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:185)、[exteroception.py:39](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py:39)、[amass_visualize.py:1](/home/xiexuhui/InstinctLab/scripts/amass_visualize.py:1) |
| TensorBoard | 外部 runner 实际创建 SummaryWriter，DOCS 提到指标会进 TensorBoard；顶层 README 未给完整启动命令，G 中补充。 | [外部 runner:121](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:121)、[DOCS.md:3](/home/xiexuhui/InstinctLab/DOCS.md:3) |
| Docker | .env.base 存在，但依赖预先准备名为 isaac-lab-base 的镜像；Dockerfile 只安装系统工具，没有安装本项目和 instinct_rl。个人服务有绝对挂载路径，不能作为通用一键安装。 | [docker/.env.base:6](/home/xiexuhui/InstinctLab/docker/.env.base:6)、[Dockerfile:1](/home/xiexuhui/InstinctLab/docker/Dockerfile:1)、[docker-compose.yaml:49](/home/xiexuhui/InstinctLab/docker/docker-compose.yaml:49) |

## F. 当前无法确认的内容

### F.1 已执行检查与缺失项证据

以下是本次命令结果记录，没有启动 simulator，也没有依据历史日志声称本次训练成功。

| 检查 | 结果 | 方法／对应源码 |
| --- | --- | --- |
| 工作树与版本 | 初始 git status --short 为空；343 个 Git 跟踪文件；目标提交见文首 | git status --short、git ls-files、git rev-parse HEAD |
| Python 语法 | source/ 与 scripts/ 下216个 .py 全部 ast.parse 通过 | 纯 AST 检查，不证明导入或运行成功 |
| 任务注册 | 14 个静态 gym.register ID | 对 tasks/**/__init__.py 进行 AST 枚举，证据位置见 A.2 |
| URDF | 29 revolute、10 fixed | xml.etree.ElementTree 解析 [popsicle URDF:65](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf:65) 所属文件 |
| Python 环境 | 默认 shell python 为3.13.11；显式 instinctlab 环境为3.11.15 | command -v python、python --version、显式环境 python -B 查询；README 要求参见 [README.md:5](/home/xiexuhui/InstinctLab/README.md:5) |
| 已安装包 metadata | torch 2.7.0+cu128、isaacsim 5.1.0.0、isaaclab 0.54.4、instinctlab 0.1.0、instinct-rl 1.0.2 | importlib.metadata；isaaclab 的包版本号不等同于 README 的 release 标签 |
| 包实际位置 | isaaclab→/home/xiexuhui/IsaacLab-Instinct；instinct_rl→/home/xiexuhui/instinct_rl | importlib.util.find_spec，在不导入 Isaac Sim 的情况下定位；对应源码见 B.2 |
| 网络前向 | N=4 时 actor=(4,29)、critic=(4,1) | 使用外部真实 ActorCritic；人工按 D.1 构造零输入，未验证真实 observation manager 输出 |
| 两个局部缺陷 | SensorDead subset 更新 shape mismatch；AsyncCircularBuffer 无 batch_ids 的 append 后读 buffer 报 TypeError；显式 env_ids 的历史读取返回(4,3,2) | 从原文件 AST 提取现有类，在 CPU 上调用；未创建测试文件、未执行源模块顶层副作用。位置见 E.2 |
| 当前 checkpoint | 整个目标仓库含被忽略目录内 *.pt 搜索结果为0；没有可直接回放的现成 .pt | pathlib.Path.cwd().rglob('*.pt')；checkpoint 使用位置：[play.py:104](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:104) |
| 历史产物 | Flat 有保存的配置、TensorBoard 事件和 Git diff；其中一个 env.yaml 记录 num_envs=256，但不能据此证明训练收敛或 checkpoint 完整 | [历史 env.yaml:86](/home/xiexuhui/InstinctLab/logs/instinct_rl/g1_locomotion_flat/20260805_220512_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05/params/env.yaml:86)、[历史 agent.yaml:6](/home/xiexuhui/InstinctLab/logs/instinct_rl/g1_locomotion_flat/20260805_220512_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05/params/agent.yaml:6) |
| 未找到的实现 | 本仓库内无 PPO/ActorCritic 实现目录；无专门 LiDAR 任务、真实机器人硬件驱动或完整部署闭环；无全任务统一 evaluator | 检索 source/、scripts/、README.md、DOCS.md 的类定义、导入及 lidar/ros2/rclpy/unitree_sdk/LowCmd/LowState/onnxruntime/deploy/instinct_onboard；唯一 ONNX Runtime loader 在 Parkour。外部导入证据：[train.py:73](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:73)、[README.md:20](/home/xiexuhui/InstinctLab/README.md:20) |

本次未确认以下事项：

1. GPU／PhysX／USD 转换／Nucleus 资源访问是否能在当前会话成功完成，也未实测每秒采样量、显存上限、reward 数值稳定性或训练收敛；这些路径要到 [train.py:183](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:183) 创建环境及 [同文件:230](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:230) 学习时才能验证。
2. 外部动作数据、terrain metadata／mesh 与教师 checkpoint 是否真实可用；目前配置包含占位符和个人路径，不能从配置存在推断数据已备齐。见 [perceptive_shadowing_cfg.py:28](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py:28)、[g1_parkour_target_amp_cfg.py:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py:34)、[instinct_rl_vae_cfg.py:150](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:150)。
3. 关节运行时排序、完整 sensor output、真实 obs_format 是否与 D 的静态推导完全一致；不同资产、history、reference 选择会改变格式，应通过 [wrapper:208](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:208) 核对。
4. 多 GPU／多节点、异步 raycast 子环境更新和高分辨率相机扩容的实际表现；相关风险见 [train.py:136](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:136)、[grouped_ray_caster_camera.py:140](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster_camera.py:140)。
5. ONNX 与 torch 的数值一致性、端到端时延、normalizer／history 在板载端的处理；仓库无 checkpoint，本次未导出。导出与比对入口：[play.py:171](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:171)、[parkour/scripts/play.py:239](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/scripts/play.py:239)。
6. 真实机器人 SDK、通信频率、控制状态机、急停、传感器校准和部署运行记录：仓库中未找到；需要另审 README 指向的 instinct_onboard，本报告不作成功部署判断。见 [README.md:22](/home/xiexuhui/InstinctLab/README.md:22)。

## G. 不修改代码的最小复现计划

此处是可执行计划，不表示本次已运行这些仿真命令。训练、播放、导出只使用现有入口与命令行覆盖；运行会写日志／模型／仿真缓存，但不需要修改仓库源码。优先 Flat，因为它的 scene 不需要动作数据和相机；证据：[flat_env_cfg.py:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py:34)。

### G.1 环境与入口预检

在仓库根目录运行；本机直接使用已定位的 Python 3.11 环境，避免默认 Python 3.13。新机器先按 README 安装 Isaac Lab、instinct_rl 和本包；不要在当前已安装环境里盲目重复安装。依据：[README.md:36](/home/xiexuhui/InstinctLab/README.md:36)、[README.md:58](/home/xiexuhui/InstinctLab/README.md:58)。

~~~bash
cd /home/xiexuhui/InstinctLab
audit_python=/home/xiexuhui/miniconda3/envs/instinctlab/bin/python
"$audit_python" --version
"$audit_python" -m pip show isaaclab isaacsim instinctlab instinct-rl torch
"$audit_python" scripts/instinct_rl/train.py --help
"$audit_python" scripts/instinct_rl/play.py --help
"$audit_python" scripts/list_envs.py
~~~

通过标准：Python 与包位置符合 B.2，list_envs 输出 A.2 的14个任务。list_envs 自己以 headless 启动 AppLauncher，不需要再传 --headless；任务导入会写固定 /tmp YAML，见 [list_envs.py:13](/home/xiexuhui/InstinctLab/scripts/list_envs.py:13)、[list_envs.py:23](/home/xiexuhui/InstinctLab/scripts/list_envs.py:23)、E.2。若模拟器／资源加载失败，应记录首个异常并停止扩大测试；不能把 AST 通过当作此步通过。

### G.2 小批量训练与 checkpoint 生成

~~~bash
"$audit_python" scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-v0 \
  --headless --device cuda:0 --num_envs 16 --seed 42 \
  --max_iterations 2 --logroot /tmp/instinctlab-repo-audit/runs \
  agent.device=cuda:0 hydra.run.dir=/tmp/instinctlab-repo-audit/hydra
~~~

通过标准：完成两次更新，无 NaN／shape／reset 错误；生成 params/env.yaml、params/agent.yaml、TensorBoard events 和 model_2.pt。CLI 来源：[train.py:26](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:26)、[同文件:29](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:29)、[同文件:121](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:121)；配置保存：[同文件:216](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:216)；两次更新后最后保存的行为由 [外部 runner:202](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:202) 确认。短跑只验证管线，不验证策略学会走路。

验收张量应为 policy(16,96)、critic(16,99)、action(16,29)、reward(16,1)、done(16,)；预期来自 D.1。现有 wrapper 无独立张量打印 CLI，若需逐项确认，可在现有 --debug 入口连接调试器后只读检查 env.get_obs_format()，不更改源码，见 [train.py:90](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:90)、[wrapper:208](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:208)。

### G.3 加载、回放与恢复训练

通用 play 没有非视频模式的有限 step 参数，所以以下用 timeout 为 smoke 回放设定外部上限；它包含模拟器启动时间。若因达到上限返回124，应结合启动及 step 日志判断，不能单看退出码判为环境失败。依据：[play.py:185](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:185)、[play.py:205](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:205)。

~~~bash
timeout --signal=INT 180s "$audit_python" scripts/instinct_rl/play.py \
  --task Instinct-Locomotion-Flat-G1-Play-v0 \
  --headless --device cuda:0 --num_envs 1 \
  --load_run '/tmp/instinctlab-repo-audit/runs/.*' --checkpoint model_2.pt

"$audit_python" scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-v0 \
  --headless --device cuda:0 --num_envs 16 --max_iterations 1 \
  --resume --load_run '/tmp/instinctlab-repo-audit/runs/.*' --checkpoint model_2.pt \
  --logroot /tmp/instinctlab-repo-audit/resume \
  agent.device=cuda:0 hydra.run.dir=/tmp/instinctlab-repo-audit/hydra-resume
~~~

这里绝对 load_run 的父目录为日志根，末段 .* 交给 get_checkpoint_path 匹配实验；首次测试根目录应只包含本次 smoke run，之后可把 .* 替换为精确实验目录名。加载逻辑见 [play.py:103](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:103)、[train.py:174](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:174)。验证重点是权重／optimizer／normalizer 加载和继续更新，不是短训模型的运动质量。

### G.4 可视化和 ONNX

~~~bash
# 有显示环境时：交互回放，手动关闭窗口结束
"$audit_python" scripts/instinct_rl/play.py \
  --task Instinct-Locomotion-Flat-G1-Play-v0 --num_envs 1 \
  --load_run '/tmp/instinctlab-repo-audit/runs/.*' --checkpoint model_2.pt

# N=1 的 ONNX 导出；导出后脚本仍进入回放循环
timeout --signal=INT 180s "$audit_python" scripts/instinct_rl/play.py \
  --task Instinct-Locomotion-Flat-G1-Play-v0 --num_envs 1 --headless \
  --load_run '/tmp/instinctlab-repo-audit/runs/.*' --checkpoint model_2.pt \
  --exportonnx

"$audit_python" -m tensorboard.main --logdir /tmp/instinctlab-repo-audit/runs
~~~

ONNX 通过标准：exported/actor.onnx 及 policy_normalizer.npz 存在；输入(1,96)、输出(1,29)符合模型约定，后续应在相同 normalizer 输入下与 torch 比较。脚本入口不会自动完成 Flat 的 ONNX Runtime 一致性测试。证据：[play.py:171](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:171)、[外部 runner:487](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:487)、[actor_critic.py:202](/home/xiexuhui/instinct_rl/instinct_rl/modules/actor_critic.py:202)。

若还需录视频，可给通用 play 加 --video --video_length 200 --video_start_step 0；该入口会自动 enable_cameras。执行前确认 code 命令存在，或接受视频保存后收尾可能报错；本计划不为规避该问题修改源码。见 [play.py:47](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:47)、[play.py:214](/home/xiexuhui/InstinctLab/scripts/instinct_rl/play.py:214)。

### G.5 可选扩展验收

在 Flat smoke 通过后，再以相同配置把 num_envs 从16扩大到64／256，记录峰值显存、采样速度和不同 env_id 的独立重置；不直接跳到4096。批量维度和 CLI 已有支持，见 [train.py:126](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:126)、[wrapper:162](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:162)。

非平地与传感器优先用现成 Parkour 任务，先准备 [g1_parkour_target_amp_cfg.py:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py:34) 要求的动作集和筛选 YAML，再用已有 train.py 的 Hydra 覆盖路径，不改配置文件。以下数据路径是需要替换的占位参数，不是已存在资源：

~~~bash
"$audit_python" scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Target-Amp-G1-v0 --headless \
  --device cuda:0 --num_envs 4 --max_iterations 2 \
  --logroot /tmp/instinctlab-repo-audit/parkour \
  agent.device=cuda:0 hydra.run.dir=/tmp/instinctlab-repo-audit/hydra-parkour \
  env.scene.motion_reference.motion_buffers.run_walk.path=/absolute/path/to/motions \
  env.scene.motion_reference.motion_buffers.run_walk.filtered_motion_selection_filepath=/absolute/path/to/selection.yaml
~~~

检查深度(4,8,18,32)、不同子环境相机 history/reset、4专家输出、坡地／台阶的生成和接触。若只调整已存在坡的范围，可以额外传 env.scene.terrain.terrain_generator.sub_terrains.hf_pyramid_slope_inv.slope_range='[0.2,0.2]'；这只固定该子地形坡度，不会让所有环境都落在坡上。删除其他地形还会牵涉 CommandsCfg.velocity_ranges 的名字映射和 target flat patches，不应盲删。依据：[parkour_env_cfg.py:261](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:261)、[同文件:630](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:630)、[pose_velocity_command.py:74](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/mdp/commands/pose_velocity_command.py:74)、[同文件:117](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/mdp/commands/pose_velocity_command.py:117)。

Perceptive／VAE 应在动作匹配 mesh、metadata 与教师 checkpoint 备齐后另测；真实机器人不在最小复现步骤中，因为本仓库没有对应硬件执行入口。依据：[mesh_terrains.py:34](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/terrains/trimesh/mesh_terrains.py:34)、[instinct_rl_vae_cfg.py:150](/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py:150)、[README.md:22](/home/xiexuhui/InstinctLab/README.md:22)。
