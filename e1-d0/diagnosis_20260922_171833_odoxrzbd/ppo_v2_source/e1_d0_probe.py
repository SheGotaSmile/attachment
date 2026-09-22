"""E1-D0: bounded, separate-process Flat contact and PPO diagnostics.

No task source is patched. Outputs must be new. Original runs retain auto reset;
only the first episode of each predeclared environment is included in traces.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import traceback

from evaluate_flat import TASK, load_exported_yaml


def plain(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, slice):
        return {'slice': [x.start, x.stop, x.step]}
    return x


def write(path, obj):
    path.write_text(json.dumps(plain(obj), indent=2, allow_nan=False) + '\n')


def geometry(stage, robot):
    """Read actual spawned USD collision shapes, relative to each rigid body.

    Capsules/spheres use analytic endpoints + radius. Other shapes retain their
    actual points or conservative local bounding-box corners, labelled as such.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    cache = UsdGeom.XformCache()
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    shapes = []
    root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False:
            continue
        body = p
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body or body.GetName() not in robot.body_names:
            continue
        transform = cache.GetLocalToWorldTransform(p) * cache.GetLocalToWorldTransform(body).GetInverse()
        typ = p.GetTypeName()
        radius = 0.0
        method = 'exact_mesh_vertices'
        if typ in ('Capsule', 'Sphere'):
            shape = getattr(UsdGeom, typ)(p)
            radius = float(shape.GetRadiusAttr().Get())
            points = np.zeros((2, 3))
            if typ == 'Capsule':
                axis = {'X': 0, 'Y': 1, 'Z': 2}[str(shape.GetAxisAttr().Get())]
                points[:, axis] = [-float(shape.GetHeightAttr().Get())/2, float(shape.GetHeightAttr().Get())/2]
            method = 'analytic_endpoints_radius'
        elif typ == 'Mesh':
            points = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=float)
        else:
            box = bounds.ComputeLocalBound(p).ComputeAlignedRange()
            lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
            points = np.array([[lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1],
                                lo[2] if i & 4 else hi[2]] for i in range(8)])
            method = 'conservative_bbox'
        local = [list(transform.Transform(Gf.Vec3d(*v))) for v in points]
        shapes.append(dict(path=str(p.GetPath()), body=body.GetName(), body_id=robot.body_names.index(body.GetName()),
                           type=typ, method=method, points_local=local, radius=radius,
                           transform_scale=list(transform.ExtractRotationMatrix().GetRow(0))))
    return shapes


def run(args, out, report):
    import gymnasium as gym
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import parse_env_cfg
    from instinct_rl.runners import OnPolicyRunner
    import instinctlab.tasks  # noqa
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
    from e0_training_audit import equal_state
    import omni.usd
    from pxr import PhysxSchema, PhysicsSchemaTools
    from omni.physx import get_physx_simulation_interface

    train = Path(args.train_dir).resolve()
    agent = load_exported_yaml(train/'params/agent.yaml')
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.mode == 'nominal':
        cfg.observations.policy.enable_corruption = False
        for name in list(vars(cfg.events)):
            if not name.startswith('_'):
                setattr(cfg.events, name, None)
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.rel_standing_envs = 1.0
    env = InstinctRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
    raw = env.unwrapped
    runner = OnPolicyRunner(env, copy.deepcopy(agent), log_dir=str(out/'training') if args.mode == 'ppo' else None,
                            device=args.device)
    checkpoint = train/('model_200.pt' if args.controller == 'model_200' else 'model_initial.pt')
    runner.load(str(checkpoint))
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    report['checkpoint'] = str(checkpoint)
    report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report['loaded_states_equal'] = {name: equal_state(mod.state_dict(), saved[key]) for name, mod, key in
        [('model', runner.alg.actor_critic, 'model_state_dict'),
         *[(name, mod, name+'_normalizer_state_dict') for name, mod in runner.normalizers.items()]]}
    assert all(report['loaded_states_equal'].values())
    dump_yaml(str(out/'env.yaml'), cfg)
    dump_yaml(str(out/'agent.yaml'), agent)
    if args.mode == 'original' or args.mode == 'ppo':
        actual = load_exported_yaml(out/'env.yaml')
        original = load_exported_yaml(train/'params/env.yaml')
        sections = ['actions', 'observations', 'commands', 'rewards', 'terminations', 'events', 'curriculum', 'decimation']
        report['task_section_mismatches'] = [s for s in sections if actual[s] != original[s]]
        assert not report['task_section_mismatches']
        assert actual['scene']['robot'] == original['scene']['robot']
    if args.mode == 'ppo':
        audit_ppo(runner, env, out, report)
        env.close()
        return
    policy = runner.get_inference_policy(device=args.device)
    modules = {'model': runner.alg.actor_critic, **runner.normalizers}
    before = {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in modules.items()}
    robot, sensor = raw.scene['robot'], raw.scene['contact_forces']
    term = raw.termination_manager.get_term_cfg('base_contact')
    ids = term.params['sensor_cfg'].body_ids
    action = raw.action_manager.get_term('joint_pos')
    stage = omni.usd.get_context().get_stage()
    shapes = geometry(stage, robot)
    assert shapes, 'No runtime collision shapes found'
    self_collision = []
    from pxr import Usd
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot')):
        if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
            self_collision.append([str(p.GetPath()), PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get()])
    runtime = dict(function=term.func.__module__+'.'+term.func.__qualname__, file=inspect.getfile(term.func),
        threshold=term.params['threshold'], history_length=sensor.cfg.history_length,
        update_period=sensor.cfg.update_period, physics_dt=raw.physics_dt, step_dt=raw.step_dt,
        sensor_body_names=sensor.body_names, robot_body_names=robot.body_names,
        sensor_to_robot=[robot.body_names.index(n) for n in sensor.body_names],
        monitored_ids=ids, monitored_names=[sensor.body_names[i] for i in ids],
        sensor_prim_paths=sensor.body_physx_view.prim_paths,
        joint_names=robot.joint_names, action_joint_names=action._joint_names, action_joint_ids=action._joint_ids,
        scale=action._scale, offset=action._offset, action_class=type(action).__module__+'.'+type(action).__name__,
        joint_limits=robot.data.joint_pos_limits, soft_joint_limits=robot.data.soft_joint_pos_limits,
        joint_effort_limits=robot.data.joint_effort_limits, joint_stiffness=robot.data.joint_stiffness,
        joint_damping=robot.data.joint_damping, joint_armature=robot.data.joint_armature,
        actuators={n:type(a).__module__+'.'+type(a).__name__ for n,a in robot.actuators.items()},
        enabled_self_collisions=self_collision, collision_shapes=shapes,
        force_matrix_available=sensor.data.force_matrix_w is not None)
    write(out/'runtime.json', runtime)
    report['runtime'] = 'runtime.json'
    obs, _ = env.get_observations()
    cmd = raw.command_manager.get_term('base_velocity')
    initial = dict(case_ids=list(range(raw.num_envs)), root_state=robot.data.root_state_w,
        joint_pos=robot.data.joint_pos, joint_vel=robot.data.joint_vel, command=cmd.command,
        command_state={k:v for k,v in vars(cmd).items() if isinstance(v, torch.Tensor)},
        materials=robot.root_physx_view.get_material_properties(), masses=robot.root_physx_view.get_masses(),
        inertias=robot.root_physx_view.get_inertias(), coms=robot.root_physx_view.get_coms(),
        interval_time_left=raw.event_manager._interval_term_time_left,
        external_wrench={k:v for k,v in vars(robot.permanent_wrench_composer).items() if isinstance(v, torch.Tensor)},
        observations=obs, contact_current=sensor._data.net_forces_w,
        contact_history=sensor._data.net_forces_w_history, default_joint_pos=robot.data.default_joint_pos,
        origins=raw.scene.env_origins)
    write(out/'initial_cases.json', initial)
    # Saving concrete states allows comparison across processes, beyond merely a seed assertion.
    active = [True]*raw.num_envs
    first = {}
    report.update(episodes=[], reset_checks=[], contact_callback_count=0)
    control_step = 0
    physics_step = 0
    contacts = (out/'contact_pairs.jsonl').open('w')
    traces = (out/'physics_trace.jsonl').open('w')
    control = (out/'control_trace.jsonl').open('w')

    def pair_callback(headers, data):
        for h in headers:
            paths = {k: str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ['actor0','actor1','collider0','collider1']}
            env_ids = [i for i in range(raw.num_envs) if active[i] and any(f'/env_{i}/' in p for p in paths.values())]
            if not env_ids:
                continue
            points = [{k:plain(list(getattr(data[j], k))) if k in ['position','normal','impulse'] else float(getattr(data[j], k))
                       for k in ['position','normal','impulse','separation']}
                      for j in range(h.contact_data_offset, h.contact_data_offset+h.num_contact_data)]
            contacts.write(json.dumps(dict(control_step=control_step, physics_step=physics_step+1,
                type=int(h.type), **paths, points=points))+'\n')
            report['contact_callback_count'] += 1

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(pair_callback)
    # Runtime body poses + actual USD local collision shape positions; Fabric
    # keeps dynamic transforms out of USD, so do not read stale USD world poses.
    from isaaclab.utils.math import quat_apply
    geometry_cache = [(s, torch.tensor(s['points_local'], device=raw.device, dtype=torch.float32)) for s in shapes]

    def geom_at(i):
        states = robot.data.body_link_state_w[i]
        result = []
        for s, pts in geometry_cache:
            state = states[s['body_id']]
            world = quat_apply(state[3:7].expand(len(pts),4), pts)+state[:3]
            result.append(dict(body=s['body'], path=s['path'], points=plain(world), radius=s['radius'],
                min_z=float(world[:,2].min())-s['radius'], method=s['method']))
        return result

    def state_row(i):
        hard = robot.data.joint_pos_limits[i]
        q = action.processed_actions[i]
        return dict(episode_id=i, case_id=i, control_step=control_step, physics_step=physics_step,
            sim_time=physics_step*raw.physics_dt, root_state=plain(robot.data.root_state_w[i]),
            projected_gravity=plain(robot.data.projected_gravity_b[i]),
            joint_pos=plain(robot.data.joint_pos[i]), joint_vel=plain(robot.data.joint_vel[i]),
            raw_action=plain(action.raw_actions[i]), q_target=plain(q),
            actual_position_target=plain(robot.data.joint_pos_target[i]),
            hard_limit_exceeded=plain((q<hard[:,0])|(q>hard[:,1])),
            computed_torque_estimate=plain(robot.data.computed_torque[i]),
            applied_torque_estimate=plain(robot.data.applied_torque[i]),
            estimated_saturation=plain(robot.data.computed_torque[i].abs()>=robot.data.joint_effort_limits[i]),
            command=plain(cmd.command[i]), monitored_force=plain(sensor.data.net_forces_w[i,ids]),
            monitored_history_max=plain(sensor.data.net_forces_w_history[i,:,ids].norm(dim=-1).max(dim=0).values),
            all_body_forces=plain(sensor.data.net_forces_w[i]), geometry=geom_at(i))

    scene_update = raw.scene.update
    reward_compute = raw.reward_manager.compute
    reset_idx = raw._reset_idx

    def update_hook(dt):
        nonlocal physics_step
        scene_update(dt)
        physics_step += 1
        for i in range(raw.num_envs):
            if not active[i]:
                continue
            row = state_row(i)
            forces = sensor.data.net_forces_w[i,ids].norm(dim=-1)
            crossing = (forces>term.params['threshold']).nonzero().flatten().tolist()
            if crossing and i not in first:
                first[i] = dict(physics_step=physics_step, control_step=control_step, time=physics_step*raw.physics_dt,
                               links=[sensor.body_names[ids[j]] for j in crossing], forces=plain(forces[crossing]))
            row.update(terminated=None, truncated=None, base_contact=None,
                       flags_phase='termination evaluated at control boundary only')
            traces.write(json.dumps(row, allow_nan=False)+'\n')

    def compute_hook(*a, **kw):
        reward = reward_compute(*a, **kw)
        for i in range(raw.num_envs):
            if not active[i]:
                continue
            row = state_row(i)
            row.update(terminated=bool(raw.termination_manager.terminated[i]),
                truncated=bool(raw.termination_manager.time_outs[i]),
                base_contact=bool(raw.termination_manager.get_term('base_contact')[i]), reward=float(reward[i]))
            control.write(json.dumps(row, allow_nan=False)+'\n')
            if row['terminated'] or row['truncated'] or control_step>=100:
                report['episodes'].append(dict(case_id=i, length=control_step, seconds=control_step*raw.step_dt,
                    terminated=row['terminated'], truncated=row['truncated'], base_contact=row['base_contact'],
                    root_state=row['root_state'], first_crossing=first.get(i),
                    result='base_contact' if row['base_contact'] else 'short_time_no_termination'))
                active[i] = False
        return reward

    def reset_hook(env_ids):
        pre = plain(sensor._data.net_forces_w_history[env_ids].abs().amax())
        result = reset_idx(env_ids)
        report['reset_checks'].append(dict(ids=plain(env_ids), before_history_abs_max=pre,
            after_history_abs_max=float(sensor._data.net_forces_w_history[env_ids].abs().max()),
            after_current_abs_max=float(sensor._data.net_forces_w[env_ids].abs().max()),
            after_termination_flags=plain(raw.termination_manager.terminated[env_ids]),
            after_episode_length=plain(raw.episode_length_buf[env_ids])))
        return result

    raw.scene.update, raw.reward_manager.compute, raw._reset_idx = update_hook, compute_hook, reset_hook
    for i in range(raw.num_envs):
        write(out/f'initial_geometry_{i}.json', geom_at(i))
    try:
        with torch.inference_mode():
            for control_step in range(1,101):
                actions = torch.zeros((raw.num_envs,env.num_actions),device=raw.device) if args.controller=='zero' else policy(obs)
                assert torch.isfinite(actions).all()
                obs, _, _, _ = env.step(actions)
                if not any(active):
                    break
        report['states_unchanged'] = {n:equal_state(before[n],m.state_dict()) for n,m in modules.items()}
        assert all(report['states_unchanged'].values())
        assert len(report['episodes']) == raw.num_envs
    finally:
        raw.scene.update, raw.reward_manager.compute, raw._reset_idx = scene_update, reward_compute, reset_idx
        subscription = None
        traces.close(); control.close(); contacts.close()
        env.close()


def audit_ppo(runner, env, out, report):
    """Exactly two diagnostic updates, original hyperparameters, read-only hooks."""
    import torch
    alg = runner.alg
    original_losses, original_update = alg.compute_losses, alg.update
    entries, checks = [], []
    iteration = 0

    def losses(batch):
        lr_before = alg.learning_rate
        result = original_losses(batch)
        mu, sig = alg.actor_critic.action_mean, alg.actor_critic.action_std
        exact = (torch.log(sig/batch.old_sigma)+(batch.old_sigma.square()+(batch.old_mu-mu).square())/(2*sig.square())-.5).sum(-1)
        ratio = result[1]['ratio']
        entries.append(dict(update=iteration, minibatch=len(entries)%20, learning_rate_before=lr_before,
            learning_rate_after=alg.learning_rate, optimizer_lr=[g['lr'] for g in alg.optimizer.param_groups],
            kl=float(result[1]['kl'].mean()), exact_kl=float(exact.mean()),
            clip_fraction=float(((ratio-1).abs()>.2).float().mean()), ratio_min=float(ratio.min()),ratio_max=float(ratio.max())))
        assert all(g['lr']==alg.learning_rate for g in alg.optimizer.param_groups)
        return result

    def update(it):
        nonlocal iteration
        iteration = it
        st = alg.storage
        with torch.no_grad():
            x = st.observations.flatten(0,1)
            mean = alg.actor_critic.actor(x)
            sigma = alg.actor_critic.std.expand_as(mean)
            logp = torch.distributions.Normal(mean,sigma).log_prob(st.actions.flatten(0,1)).sum(-1)
            checks.append(dict(update=it, old_mu_max_error=float((mean-st.mu.flatten(0,1)).abs().max()),
                old_sigma_max_error=float((sigma-st.sigma.flatten(0,1)).abs().max()),
                old_log_prob_max_error=float((logp-st.actions_log_prob.flatten()).abs().max()),
                normalizer_counts={n:int(m.count) for n,m in runner.normalizers.items()}))
        return original_update(it)

    alg.compute_losses, alg.update = losses, update
    # Keep original randomized episode-length initialization used by train.py.
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    report.update(updates=2, transitions=2*env.num_envs*runner.num_steps_per_env,
                  ppo_minibatches=entries, rollout_consistency=checks,
                  limitation='Diagnostic replay from E1 initial checkpoint; not a bit-exact reproduction of pilot RNG.')
    write(out/'ppo_audit.json', report)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['original','nominal','ppo'], default='original')
    parser.add_argument('--controller', choices=['zero','model_initial','model_200'], default='zero')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    report = dict(status='RUNNING', argv=sys.argv, pid=os.getpid(), mode=args.mode,
                  controller=args.controller, seed=args.seed, num_envs=args.num_envs)
    app = None
    try:
        app = AppLauncher(args).app
        run(args,out,report)
        report['status'] = 'PASS'
    except Exception:
        report.update(status='FAIL', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        write(out/'result.json',report)
        print('[E1-D0]',report['status'],str(out),flush=True)
        if app:
            app.close()


if __name__=='__main__':
    main()
