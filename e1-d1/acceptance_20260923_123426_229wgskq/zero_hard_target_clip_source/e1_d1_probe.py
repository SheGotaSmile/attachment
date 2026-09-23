"""E1-D1: first-episode, no-training Flat G1 semantics and action diagnostics.

Run under the Isaac Python environment. Every invocation requires a new output
directory. The research task's action definition is never changed: diagnostic
transforms are applied to its input and fully recorded. Matrix runs disable
observation noise for reproducible pairing; all other Flat reset/physics settings
are retained. Nominal is a separate deterministic test.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

TASK = "Instinct-Locomotion-Flat-G1-Research-v0"
ORIGINAL_TASK = "Instinct-Locomotion-Flat-G1-v0"
ROOT = Path(__file__).resolve().parents[2]
D0 = ROOT / "logs/e1-d0/diagnosis_20260922_171833_odoxrzbd"


def plain(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    return value


def write_json(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run(args, out, report):
    import gymnasium as gym
    import torch
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdPhysics
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.math import quat_apply
    from isaaclab_tasks.utils import parse_env_cfg
    from instinct_rl.runners import OnPolicyRunner
    import instinctlab.tasks  # noqa: F401
    from instinctlab.tasks.locomotion.config.g1.research_env_cfg import primary_fall
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    sys.path.insert(0, str(ROOT / "scripts/instinct_rl"))
    from evaluate_flat import load_exported_yaml
    from e1_d0_probe import geometry

    train = Path(args.train_dir).resolve()
    task = ORIGINAL_TASK if args.legacy_task else TASK
    count = 8 if args.mode == "matrix" else 1
    cfg = parse_env_cfg(task, device=args.device, num_envs=count)
    cfg.seed = args.seed
    cfg.observations.policy.enable_corruption = False
    if args.mode != "matrix":
        original_events = copy.deepcopy(cfg.events)
        for name in list(vars(cfg.events)):
            if not name.startswith("_"):
                setattr(cfg.events, name, None)
        cfg.events.reset_base = original_events.reset_base
        cfg.events.reset_base.params.update(pose_range={}, velocity_range={})
        cfg.events.reset_robot_joints = original_events.reset_robot_joints
        cfg.events.reset_robot_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
        command_cfg = cfg.commands.base_velocity
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)
        command_cfg.heading_command = False
        command_cfg.rel_standing_envs = 1.0
    dump_yaml(str(out / "env.yaml"), cfg)
    agent = load_exported_yaml(train / "params/agent.yaml")
    dump_yaml(str(out / "agent.yaml"), agent)
    report.update(task=task, controller=args.controller, action_mode=args.action_mode,
                  mode=args.mode, cases=count, seed=args.seed, max_duration_s=2.0,
                  observation_noise=False, training=False,
                  action_transform={
                      "legacy_raw": "q_default + scale * raw",
                      "raw_clip": "q_default + scale * clamp(raw, -1, 1)",
                      "hard_target_clip": "clamp(q_default + scale * raw, hard_lower, hard_upper); no raw clip",
                  }[args.action_mode])
    env = InstinctRlVecEnvWrapper(gym.make(task, cfg=cfg))
    raw = env.unwrapped
    robot, sensor = raw.scene["robot"], raw.scene["contact_forces"]
    action = raw.action_manager.get_term("joint_pos")
    cmd = raw.command_manager.get_term("base_velocity")
    legacy_cfg = parse_env_cfg(ORIGINAL_TASK, device=args.device, num_envs=count).terminations.base_contact
    legacy_cfg.params["sensor_cfg"].resolve(raw.scene)
    ids = legacy_cfg.params["sensor_cfg"].body_ids
    d0_runtime = json.loads((D0 / "final_pairs/runtime.json").read_text())
    monitored_names = [sensor.body_names[i] for i in ids]
    assert monitored_names == d0_runtime["monitored_names"]
    assert action._joint_names == robot.joint_names == d0_runtime["action_joint_names"]
    assert action.cfg.clip is None
    scale = torch.as_tensor(action._scale, device=raw.device).expand(count, env.num_actions)
    offset = torch.as_tensor(action._offset, device=raw.device).expand(count, env.num_actions)
    assert torch.equal(offset, robot.data.default_joint_pos)
    assert torch.allclose(scale[0], torch.tensor(d0_runtime["scale"][0], device=raw.device))
    assert bool((scale != 0).all())
    obs, extras = env.get_observations()
    shapes = {k: list(v.shape) for k, v in extras["observations"].items()}
    assert shapes == {"policy": [count, 96], "critic": [count, 99]}, shapes
    assert env.num_actions == 29
    modules, before = {}, {}
    policy = None
    if args.controller != "zero":
        runner = OnPolicyRunner(env, copy.deepcopy(agent), log_dir=None, device=args.device)
        checkpoint = train / (args.controller + ".pt")
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        saved = torch.load(checkpoint, weights_only=True, map_location=raw.device)
        modules = {"model": runner.alg.actor_critic, **runner.normalizers}
        for name, module in modules.items():
            state = saved["model_state_dict" if name == "model" else name + "_normalizer_state_dict"]
            assert all(torch.equal(v, state[k]) for k, v in module.state_dict().items())
            before[name] = {k: v.clone() for k, v in module.state_dict().items()}
            assert not module.training
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      loaded_states_equal=True, frozen_normalizers=True, policy="deterministic Gaussian mean")
    else:
        report.update(checkpoint=None, policy="zero raw action; no model loaded")

    stage = omni.usd.get_context().get_stage()
    collision_shapes = geometry(stage, robot)
    assert collision_shapes
    body_names_with_collision = sorted({s["body"] for s in collision_shapes})
    assert {"torso_link", "pelvis"}.issubset(body_names_with_collision)
    ground_paths = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/ground"))
                    if p.HasAPI(UsdPhysics.CollisionAPI)]
    self_collision = [[str(p.GetPath()), PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get()]
                      for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
                      if p.HasAPI(PhysxSchema.PhysxArticulationAPI)]
    assert self_collision and all(row[1] for row in self_collision)
    runtime = dict(policy_critic_shapes=shapes, action_shape=[count, env.num_actions],
                   joint_names=robot.joint_names, action_joint_names=action._joint_names,
                   action_joint_ids=action._joint_ids, action_scale=scale, action_offset=offset,
                   q_default=robot.data.default_joint_pos, hard_limits=robot.data.joint_pos_limits,
                   soft_limits=robot.data.soft_joint_pos_limits, sensor_body_names=sensor.body_names,
                   robot_body_names=robot.body_names, legacy_monitored_ids=ids,
                   legacy_monitored_names=monitored_names, legacy_threshold_N=legacy_cfg.params["threshold"],
                   contact_sensor_history_length=sensor.cfg.history_length,
                   contact_sensor_update_period=sensor.cfg.update_period, physics_dt=raw.physics_dt,
                   control_dt=raw.step_dt, enabled_self_collisions=self_collision,
                   primary_fall={"minimum_height_m": 0.45, "max_tilt_rad": 1.20},
                   collision_shapes=collision_shapes, ground_collision_paths=ground_paths,
                   joint_effort_limits=robot.data.joint_effort_limits,
                   stiffness=robot.data.joint_stiffness, damping=robot.data.joint_damping,
                   armature=robot.data.joint_armature,
                   actuators={n: type(a).__name__ for n, a in robot.actuators.items()})
    write_json(out / "runtime.json", runtime)
    initial = dict(root_state=robot.data.root_state_w, joint_pos=robot.data.joint_pos,
                   joint_vel=robot.data.joint_vel, command=cmd.command,
                   command_state={k: v for k, v in vars(cmd).items() if isinstance(v, torch.Tensor)},
                   materials=robot.root_physx_view.get_material_properties(), masses=robot.root_physx_view.get_masses(),
                   inertias=robot.root_physx_view.get_inertias(), coms=robot.root_physx_view.get_coms(),
                   interval_time_left=raw.event_manager._interval_term_time_left,
                   observations=extras["observations"], default_joint_pos=robot.data.default_joint_pos,
                   origins=raw.scene.env_origins)
    write_json(out / "initial_cases.json", initial)
    if args.mode != "matrix":
        checks = dict(default_joint_pose=torch.equal(robot.data.joint_pos, robot.data.default_joint_pos),
                      zero_joint_velocity=bool((robot.data.joint_vel == 0).all()),
                      zero_root_velocity=bool((robot.data.root_state_w[:, 7:] == 0).all()),
                      default_root_height=torch.equal(robot.data.root_pos_w[:, 2], robot.data.default_root_state[:, 2]),
                      zero_command=bool((cmd.command == 0).all()), observation_noise_off=not cfg.observations.policy.enable_corruption,
                      no_dynamics_randomization=set(raw.event_manager.active_terms.get("startup", [])) == set())
        assert all(checks.values()), checks
        report["nominal_initial_checks"] = checks

    # Inspect reset buffers directly: the public .data accessor does a lazy
    # PhysX refresh and can replace zeroed history with stale pre-reset forces.
    reset_checks = []
    original_reset = raw._reset_idx

    def reset_hook(env_ids):
        before_history = sensor._data.net_forces_w_history[env_ids].clone()
        result = original_reset(env_ids)
        check = dict(case_ids=plain(env_ids), before_history_max_N=float(before_history.abs().max()),
                     after_history_max_N=float(sensor._data.net_forces_w_history[env_ids].abs().max()),
                     after_current_max_N=float(sensor._data.net_forces_w[env_ids].abs().max()))
        reset_checks.append(check)
        assert check["after_history_max_N"] == check["after_current_max_N"] == 0.0
        return result

    raw._reset_idx = reset_hook
    # Initial reset check without consuming any RNG or altering concrete cases.
    sensor.reset()
    assert float(sensor._data.net_forces_w_history.abs().max()) == 0.0
    report["initial_contact_history_cleared"] = True

    pair_views, pair_errors = [], []
    for i in range(count):
        filters = [f"/World/envs/env_{i}/Robot/{n}" for n in robot.body_names] + ["/World/ground/*"]
        for name in body_names_with_collision:
            try:
                view = sensor._physics_sim_view.create_rigid_contact_view(
                    f"/World/envs/env_{i}/Robot/{name}", filter_patterns=filters, max_contact_data_count=512)
                assert view.sensor_count == 1 and view.filter_count == len(filters)
                pair_views.append((i, name, filters, view))
            except Exception as exc:
                pair_errors.append(dict(case_id=i, link=name, error=repr(exc)))

    active = [True] * count
    current_step, physics_step = 0, 0
    rows, contacts, episodes = [], [], []
    first_self, first_ground, first_raw = {}, {}, {}
    predicted_target = offset.clone()
    raw_action = torch.zeros_like(offset)
    applied_action = raw_action.clone()
    trace = (out / "physics_trace.jsonl").open("w")
    callbacks = (out / "contact_callbacks.jsonl").open("w")
    geom_cache = [(s, torch.tensor(s["points_local"], dtype=torch.float32, device=raw.device)) for s in collision_shapes]
    max_affine_error, max_application_error, max_pair_net_error = 0.0, 0.0, 0.0
    callback_count = 0
    force_epsilon = 1e-6  # numerical zero only; legacy threshold remains exactly 1 N

    def legacy():
        return legacy_cfg.func(raw, **legacy_cfg.params)

    def state(i):
        root = robot.data.root_state_w[i]
        gravity = robot.data.projected_gravity_b[i]
        tilt = math.acos(max(-1.0, min(1.0, -float(gravity[2]))))
        return dict(root_height_m=float(root[2]), root_orientation_quat=plain(root[3:7]),
                    root_tilt_rad=tilt, root_state=plain(root),
                    joint_position=plain(robot.data.joint_pos[i]), joint_velocity=plain(robot.data.joint_vel[i]))

    def add_contact(i, a, b, force, points, classification, evidence, step=None, extra=None):
        pstep = physics_step if step is None else step
        primary = bool(primary_fall(raw)[i])
        row = dict(controller=args.controller, action_mode=args.action_mode, mode=args.mode,
                   case_id=i, control_step=current_step, physics_step=pstep,
                   contact_time_s=pstep * raw.physics_dt, link_a=a, link_b=b,
                   contact_force_N=float(force), contact_points=plain(points),
                   self_collision=classification == "robot_robot_self_contact" if classification != "unconfirmed" else None,
                   primary_fall_at_sample=primary, primary_fall_same_control_step=None,
                   episode_primary_fall=None, causes_primary_fall="not_established",
                   classification=classification, evidence=evidence, **state(i))
        if extra:
            row.update(extra)
        contacts.append(row)
        first = dict(control_step=current_step, physics_step=pstep, time_s=pstep * raw.physics_dt,
                     link_a=a, link_b=b, force_N=float(force), primary_fall_at_sample=primary, evidence=evidence)
        if classification == "robot_robot_self_contact":
            first_self.setdefault(i, first)
        elif classification == "robot_ground_contact":
            first_ground.setdefault(i, first)

    def callback(headers, data):
        nonlocal callback_count
        for h in headers:
            paths = {k: str(PhysicsSchemaTools.intToSdfPath(getattr(h, k)))
                     for k in ("actor0", "actor1", "collider0", "collider1")}
            points = [{k: plain(list(getattr(data[j], k))) if k in ("position", "normal", "impulse")
                       else float(getattr(data[j], k)) for k in ("position", "normal", "impulse", "separation")}
                      for j in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data)]
            callbacks.write(json.dumps(dict(control_step=current_step, physics_step=physics_step + 1,
                                             type=int(h.type), **paths, points=points), allow_nan=False) + "\n")
            callback_count += 1
            if not points:
                continue
            for i in range(count):
                prefix = f"/World/envs/env_{i}/Robot/"
                if not active[i] or not any(prefix in p for p in paths.values()):
                    continue
                a, b = paths["actor0"], paths["actor1"]
                if prefix not in a:
                    a, b = b, a
                category = ("robot_robot_self_contact" if prefix in b else
                            "robot_ground_contact" if any(p.startswith("/World/ground/") for p in paths.values())
                            else "unconfirmed")
                force = math.sqrt(sum(sum(p["impulse"][j] for p in points) ** 2 for j in range(3))) / raw.physics_dt
                add_contact(i, a.rsplit("/", 1)[-1], b, force, [p["position"] for p in points],
                            category, "PhysX contact callback impulse/dt", step=physics_step + 1)

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(callback)
    scene_update, reward_compute = raw.scene.update, raw.reward_manager.compute

    def update_hook(dt):
        nonlocal physics_step, max_pair_net_error, max_application_error
        scene_update(dt)
        physics_step += 1
        forces = sensor.data.net_forces_w
        current_primary = primary_fall(raw)
        for tensor in (forces, sensor.data.net_forces_w_history, robot.data.joint_pos, robot.data.joint_vel,
                       robot.data.root_state_w, robot.data.applied_torque, action.processed_actions):
            assert torch.isfinite(tensor).all(), "nonfinite physics tensor"
        max_application_error = max(max_application_error, float((action.processed_actions - robot.data.joint_pos_target).abs().max()))
        known = set()
        for i, name, filters, view in pair_views:
            if not active[i]:
                continue
            try:
                matrix = view.get_contact_force_matrix(dt=raw.physics_dt)[0]
                net = view.get_net_contact_forces(dt=raw.physics_dt)[0]
                assert torch.isfinite(matrix).all() and torch.isfinite(net).all()
                max_pair_net_error = max(max_pair_net_error, float((net - forces[i, sensor.body_names.index(name)]).abs().max()))
                hits = (matrix.norm(dim=-1) > force_epsilon).nonzero().flatten().tolist()
                if not hits:
                    continue
                _, points, _, _, counts, starts = view.get_contact_data(dt=raw.physics_dt)
                for j in hits:
                    n, start = int(counts[0, j]), int(starts[0, j])
                    category = "robot_ground_contact" if j == len(filters) - 1 else "robot_robot_self_contact"
                    add_contact(i, name, filters[j], float(matrix[j].norm()), points[start:start + n],
                                category, "PhysX filtered contact matrix (normal force)")
                    known.add((i, name))
            except Exception as exc:
                error = dict(case_id=i, link=name, physics_step=physics_step, error=repr(exc))
                if len(pair_errors) < 100:
                    pair_errors.append(error)
        for i in range(count):
            if not active[i]:
                continue
            # Keep every nonzero sensor event, even with absent/partial pair coverage.
            for j in (forces[i].norm(dim=-1) > force_epsilon).nonzero().flatten().tolist():
                name = sensor.body_names[j]
                add_contact(i, name, "unresolved_sensor_net", float(forces[i, j].norm()), [], "unconfirmed",
                            "net force; pair rows are separate, not assumed exhaustive")
            geo = []
            states = robot.data.body_link_state_w[i]
            for shape, pts in geom_cache:
                pose = states[shape["body_id"]]
                world = quat_apply(pose[3:7].expand(len(pts), 4), pts) + pose[:3]
                geo.append(dict(body=shape["body"], points=plain(world), radius=shape["radius"],
                                min_z=float(world[:, 2].min()) - shape["radius"], method=shape["method"]))
            trace.write(json.dumps(dict(case_id=i, control_step=current_step, physics_step=physics_step,
                                        time_s=physics_step * raw.physics_dt, **state(i),
                                        primary_fall_predicate=bool(current_primary[i]), geometry=geo,
                                        net_forces_w=plain(forces[i]), q_target=plain(action.processed_actions[i]),
                                        actual_position_target=plain(robot.data.joint_pos_target[i]),
                                        computed_torque=plain(robot.data.computed_torque[i]),
                                        applied_torque=plain(robot.data.applied_torque[i])), allow_nan=False) + "\n")

    def compute_hook(*a, **kw):
        nonlocal max_affine_error
        reward = reward_compute(*a, **kw)
        assert torch.isfinite(reward).all()
        primary, raw_contact = primary_fall(raw), legacy()
        q = action.processed_actions
        max_affine_error = max(max_affine_error, float((predicted_target - q).abs().max()))
        assert torch.allclose(predicted_target, q, atol=2e-6, rtol=1e-6)
        if args.controller == "zero":
            assert torch.equal(q, offset)
        if not args.legacy_task:
            assert torch.equal(primary, raw.termination_manager.get_term("primary_fall"))
        for i in range(count):
            if not active[i]:
                continue
            hard, soft = robot.data.joint_pos_limits[i], robot.data.soft_joint_pos_limits[i]
            violations = lambda values, limits: plain((values < limits[:, 0] - 1e-6) | (values > limits[:, 1] + 1e-6))
            if raw_contact[i]:
                first_raw.setdefault(i, dict(control_step=current_step, physics_step=physics_step,
                                            time_s=physics_step * raw.physics_dt, **state(i)))
            terminated = bool(raw.termination_manager.terminated[i])
            truncated = bool(raw.termination_manager.time_outs[i])
            row = dict(controller=args.controller, action_mode=args.action_mode, mode=args.mode,
                       case_id=i, control_step=current_step, physics_step=physics_step,
                       time_s=physics_step * raw.physics_dt, raw_action=plain(raw_action[i]),
                       raw_action_abs_max=float(raw_action[i].abs().max()),
                       clipped_action=plain(applied_action[i]), raw_clip_candidate=plain(raw_action[i].clamp(-1, 1)),
                       q_default=plain(robot.data.default_joint_pos[i]), action_scale=plain(scale[i]),
                       action_offset=plain(offset[i]), q_target=plain(q[i]),
                       actual_position_target=plain(robot.data.joint_pos_target[i]),
                       soft_limit_violation=violations(q[i], soft), hard_limit_violation=violations(q[i], hard),
                       actual_q_soft_violation=violations(robot.data.joint_pos[i], soft),
                       actual_q_hard_violation=violations(robot.data.joint_pos[i], hard),
                       primary_fall=bool(primary[i]), raw_contact=bool(raw_contact[i]),
                       legacy_would_have_terminated=i in first_raw,
                       terminated=terminated, truncated=truncated, finite=True,
                       command=plain(cmd.command[i]), **state(i))
            rows.append(row)
            if terminated or truncated or current_step == 100:
                episodes.append(dict(case_id=i, length=current_step, seconds=current_step * raw.step_dt,
                                     primary_fall=bool(primary[i]), terminated=terminated, truncated=truncated,
                                     horizon_reached=current_step == 100 and not terminated,
                                     raw_contact_at_end=bool(raw_contact[i]), first_raw_contact=first_raw.get(i),
                                     first_self_collision=first_self.get(i), first_robot_ground_contact=first_ground.get(i),
                                     **state(i)))
                active[i] = False
        return reward

    raw.scene.update, raw.reward_manager.compute = update_hook, compute_hook
    try:
        with torch.inference_mode():
            if args.mode == "ground_positive":
                # Positive sensor test only: a full robot lying on its back.
                # No change to task reset height or nominal initialization.
                root = robot.data.default_root_state.clone()
                root[:, :3] += raw.scene.env_origins
                root[:, 2] = 0.10
                root[:, 3:7] = torch.tensor([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0], device=raw.device)
                root[:, 7:] = 0
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.joint_vel))
                write_json(out / "injected_ground_test_state.json", dict(root=root, q=robot.data.default_joint_pos,
                           diagnostic_only=True, bypass_auto_reset=True, reason="observe real torso/pelvis ground pairs"))
                raw.action_manager.process_action(torch.zeros_like(offset))
                for step in range(200):
                    current_step = step // cfg.decimation + 1
                    raw.action_manager.apply_action()
                    raw.scene.write_data_to_sim()
                    raw.sim.step(render=False)
                    raw.scene.update(raw.physics_dt)
                report["ground_positive_links"] = sorted({r["link_a"] for r in contacts if r["classification"] == "robot_ground_contact"})
                report["ground_positive_torso_pelvis_verified"] = {n: n in report["ground_positive_links"] for n in ("torso_link", "pelvis")}
            else:
                for current_step in range(1, 101):
                    raw_action = torch.zeros_like(offset) if policy is None else policy(obs)
                    assert torch.isfinite(raw_action).all()
                    if args.action_mode == "raw_clip":
                        applied_action = raw_action.clamp(-1, 1)
                        predicted_target = offset + scale * applied_action
                    elif args.action_mode == "hard_target_clip":
                        limits = robot.data.joint_pos_limits
                        predicted_target = torch.clamp(offset + scale * raw_action, limits[:, :, 0], limits[:, :, 1])
                        applied_action = (predicted_target - offset) / scale
                    else:
                        applied_action = raw_action.clone()
                        predicted_target = offset + scale * applied_action
                    obs, reward, _, extras = env.step(applied_action)
                    for value in (obs, reward, *extras["observations"].values()):
                        assert torch.isfinite(value).all(), "nonfinite observation or reward"
                    if not any(active):
                        break
                assert len(episodes) == count
            # Explicit reset after a nonzero-contact horizon also verifies clearing.
            raw._reset_idx(torch.arange(count, device=raw.device))
            report["states_unchanged"] = {name: all(torch.equal(v, before[name][k]) for k, v in module.state_dict().items())
                                          for name, module in modules.items()}
            assert all(report["states_unchanged"].values())
            control_primary = {(r["case_id"], r["control_step"]): r["primary_fall"] for r in rows}
            ep_primary = {e["case_id"]: e["primary_fall"] for e in episodes}
            for row in contacts:
                row["primary_fall_same_control_step"] = control_primary.get((row["case_id"], row["control_step"]))
                row["episode_primary_fall"] = ep_primary.get(row["case_id"])
            report.update(episodes=episodes, action_rows=len(rows), contact_rows=len(contacts), finite=True,
                          max_affine_error=max_affine_error, max_target_application_error=max_application_error,
                          max_pair_net_error=max_pair_net_error, reset_history_checks=reset_checks,
                          policy_critic_shapes=shapes, action_shape=[count, 29], pair_view_count=len(pair_views),
                          pair_view_errors=pair_errors, callback_count=callback_count,
                          raw_action_abs_max=max((r["raw_action_abs_max"] for r in rows), default=0),
                          applied_action_abs_max=max((max(abs(x) for x in r["clipped_action"]) for r in rows), default=0),
                          hard_target_violation_rows=sum(any(r["hard_limit_violation"]) for r in rows),
                          soft_target_violation_rows=sum(any(r["soft_limit_violation"]) for r in rows),
                          actual_q_hard_violation_rows=sum(any(r["actual_q_hard_violation"]) for r in rows),
                          primary_fall_count=sum(e["primary_fall"] for e in episodes),
                          ground_pair_rows=sum(r["classification"] == "robot_ground_contact" for r in contacts),
                          self_pair_rows=sum(r["classification"] == "robot_robot_self_contact" for r in contacts),
                          unknown_sensor_rows=sum(r["classification"] == "unconfirmed" for r in contacts))
            report["status"] = "PASS"
    finally:
        raw.scene.update, raw.reward_manager.compute, raw._reset_idx = scene_update, reward_compute, original_reset
        subscription = None
        trace.close()
        callbacks.close()
        write_csv(out / "action_safety.csv", rows)
        write_csv(out / "contact_taxonomy.csv", contacts)
        write_json(out / "result.json", report)
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--controller", choices=["zero", "model_initial", "model_200"], default="zero")
    parser.add_argument("--action-mode", choices=["legacy_raw", "raw_clip", "hard_target_clip"], default="legacy_raw")
    parser.add_argument("--mode", choices=["matrix", "nominal", "ground_positive"], default="matrix")
    parser.add_argument("--legacy-task", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", argv=sys.argv, pid=os.getpid(), probe_version=2)
    app = None
    try:
        app = AppLauncher(args).app
        run(args, out, report)
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(out / "result.json", report)
        if app:
            app.close()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
