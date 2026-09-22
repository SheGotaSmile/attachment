"""Independent Gaussian-mean evaluator for the unmodified Flat G1 training task."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import MethodType

TASK = 'Instinct-Locomotion-Flat-G1-v0'
FIELDS = ['episode_id', 'env_id', 'env_episode_id', 'episode_return', 'episode_length',
          'velocity_tracking_error', 'velocity_tracking', 'base_contact', 'timeout', 'fall',
          'terminated', 'truncated', 'actual_forward_distance', 'actual_velocity',
          'mean_speed_xy', 'command_mode', 'seed']


def load_exported_yaml(path):
    """Read Isaac's tuples/slices for comparison, without arbitrary object loading."""
    import yaml

    class ExportLoader(yaml.FullLoader):
        pass

    ExportLoader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice',
        lambda loader, node: {'slice': loader.construct_sequence(node)})
    return yaml.load(Path(path).read_text(), Loader=ExportLoader)


def write_results(output, report):
    (output/'episodes.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (output/'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(report['episodes'])


def evaluate(args, report, output):
    # Isaac modules must be imported after AppLauncher has started Kit.
    import gymnasium as gym
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.math import quat_apply, yaw_quat
    from isaaclab_tasks.utils import parse_env_cfg
    from instinct_rl.runners import OnPolicyRunner
    import instinctlab.tasks  # noqa: F401
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
    from e0_training_audit import finite, equal_state
    from flat_metrics import measurements

    checkpoint = Path(args.checkpoint).resolve()
    if 'e0' in checkpoint.parts:
        raise ValueError('E0 checkpoints are excluded from this pilot evaluator.')
    train_dir = checkpoint.parent
    train_env = load_exported_yaml(train_dir/'params/env.yaml')
    train_agent = load_exported_yaml(train_dir/'params/agent.yaml')
    if train_agent['policy']['class_name'] != 'ActorCritic' or train_agent.get('ckpt_manipulator'):
        raise ValueError('Only the original Flat ActorCritic checkpoint is supported.')
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # Original registry configuration, including noise/randomization/push/commands.
    # Check semantic equivalence after manager resolution, rather than using Play.
    env = InstinctRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg))
    raw = env.unwrapped
    original_compute = raw.reward_manager.compute
    original_reset = raw._reset_idx
    try:
        dump_yaml(str(output/'env.yaml'), env_cfg)
        eval_env = load_exported_yaml(output/'env.yaml')
        checked_sections = ['actions', 'observations', 'commands', 'rewards', 'terminations',
                            'events', 'curriculum', 'decimation', 'episode_length_s']
        mismatches = [key for key in checked_sections if train_env[key] != eval_env[key]]
        if train_env['scene']['robot']['actuators'] != eval_env['scene']['robot']['actuators']:
            mismatches.append('actuators')
        assert not mismatches, f'Task definition drift: {mismatches}'
        assert raw.cfg.scene.terrain.terrain_type == 'plane'
        runner = OnPolicyRunner(env, train_agent, log_dir=None, device=args.device)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        finite(saved, 'evaluation.checkpoint')
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        assert set(runner.normalizers) == {'policy', 'critic'}
        modules = {'model': runner.alg.actor_critic, **runner.normalizers}
        before = {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                  for name, module in modules.items()}
        restored = {'model': equal_state(before['model'], saved['model_state_dict'])}
        restored.update({name: equal_state(before[name], saved[f'{name}_normalizer_state_dict'])
                         for name in runner.normalizers})
        assert all(restored.values()), restored
        assert all(not module.training for module in modules.values())
        report.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      loaded_iteration=runner.current_learning_iteration, loaded_states_equal=restored,
                      configuration_sections_equal=checked_sections+['actuators'], step_dt=raw.step_dt)
        robot = raw.scene['robot']
        counts = [0] * raw.num_envs
        returns = torch.zeros(raw.num_envs, device=raw.device)
        lengths = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.long)
        sums = {key: torch.zeros_like(returns) for key in
                ['base_velocity_error', 'velocity_tracking', 'forward_velocity', 'speed_xy']}
        start_pos = robot.data.root_pos_w[:, :2].clone()

        def heading():
            basis = torch.zeros_like(robot.data.root_pos_w)
            basis[:, 0] = 1
            return quat_apply(yaw_quat(robot.data.root_quat_w), basis)[:, :2]

        start_heading = heading().clone()
        latest = {}
        report['reset_batches'] = []
        report['episode_return_max_error'] = 0.0

        def compute(*pos, **kw):
            reward = original_compute(*pos, **kw)
            latest.clear()
            latest.update(measurements(raw))
            finite(latest, 'evaluation.pre_reset_metrics')
            finite(reward, 'evaluation.reward')
            # Includes the terminal transition; executes before any auto-reset.
            returns.add_(reward.reshape(raw.num_envs, -1).sum(-1))
            lengths.add_(1)
            for key in sums:
                sums[key].add_(latest[key])
            return reward

        def reset_hook(self, env_ids):
            ids = env_ids.tolist()
            report['reset_batches'].append({'step': report['vector_steps']+1, 'env_ids': ids})
            for env_id in ids:
                assert lengths[env_id] == self.episode_length_buf[env_id]
                if counts[env_id] >= args.episodes_per_env:
                    continue
                length = int(lengths[env_id])
                assert length > 0
                # Independent cross-check against the environment's per-term
                # episode sums, also still intact before reset.
                manager_return = sum(value[env_id] for value in self.reward_manager._episode_sums.values())
                error = float((manager_return - returns[env_id]).abs())
                report['episode_return_max_error'] = max(report['episode_return_max_error'], error)
                assert torch.isclose(manager_return, returns[env_id], atol=1e-4, rtol=1e-5), error
                row = dict(episode_id=len(report['episodes']), env_id=env_id,
                    env_episode_id=counts[env_id], episode_return=float(returns[env_id]), episode_length=length,
                    velocity_tracking_error=float(sums['base_velocity_error'][env_id]/length),
                    velocity_tracking=float(sums['velocity_tracking'][env_id]/length),
                    actual_velocity=float(sums['forward_velocity'][env_id]/length),
                    mean_speed_xy=float(sums['speed_xy'][env_id]/length),
                    actual_forward_distance=float(torch.dot(
                        robot.data.root_pos_w[env_id, :2]-start_pos[env_id], start_heading[env_id])),
                    command_mode='original', seed=args.seed)
                for key in ['base_contact', 'timeout', 'terminated', 'truncated']:
                    row[key] = bool(latest[key][env_id])
                # A physical fall and timeout may both occur in the same step.
                row['fall'] = row['base_contact']
                assert row['terminated'] or row['truncated']
                finite(row, 'evaluation.episode')
                report['episodes'].append(row)
                counts[env_id] += 1
            result = original_reset(env_ids)
            returns[env_ids] = 0
            lengths[env_ids] = 0
            for value in sums.values():
                value[env_ids] = 0
            start_pos[env_ids] = robot.data.root_pos_w[env_ids, :2]
            start_heading[env_ids] = heading()[env_ids]
            return result

        raw.reward_manager.compute = compute
        raw._reset_idx = MethodType(reset_hook, raw)
        obs, extras = env.get_observations()
        with torch.inference_mode():
            first_actions = policy(obs)
            direct_mean = runner.alg.actor_critic.actor(runner.normalizers['policy'](obs))
            report['mean_action_max_error'] = float((first_actions-direct_mean).abs().max())
            assert torch.equal(first_actions, direct_mean)
            for _ in range(args.max_steps):
                finite(obs, 'evaluation.obs')
                finite(extras['observations'], 'evaluation.all_obs')
                actions = policy(obs)
                finite(actions, 'evaluation.mean_actions')
                obs, reward, done, extras = env.step(actions)
                assert torch.equal(done.bool(), latest['terminated'] | latest['truncated'])
                report['vector_steps'] += 1
                if all(count >= args.episodes_per_env for count in counts):
                    break
        finite(obs, 'evaluation.final_obs')
        report['states_unchanged'] = {name: equal_state(before[name], module.state_dict())
                                      for name, module in modules.items()}
        assert all(report['states_unchanged'].values()), report['states_unchanged']
        assert all(not module.training for module in modules.values())
        report['completed_per_env'] = counts
        report['unfinished_episode_lengths'] = lengths.cpu().tolist()
        assert all(count == args.episodes_per_env for count in counts), 'Episode quota not met within max_steps'
        report['asynchronous_reset_observed'] = any(len(b['env_ids']) < raw.num_envs for b in report['reset_batches'])
        report['status'] = 'PASS'
    finally:
        raw.reward_manager.compute = original_compute
        raw._reset_idx = original_reset
        env.close()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--episodes-per-env', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--task', choices=[TASK], default=TASK)
    parser.add_argument('--command-mode', choices=['original'], default='original')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.episodes_per_env, args.max_steps) < 1:
        parser.error('num-envs, episodes-per-env and max-steps must be positive')
    if args.enable_cameras:
        parser.error('Camera input is excluded from this evaluator')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', argv=sys.argv, pid=os.getpid(), task=TASK,
                  checkpoint=str(Path(args.checkpoint).resolve()), seed=args.seed, num_envs=args.num_envs,
                  command_mode='original', sampled=False, vector_steps=0, episodes=[],
                  definitions={
                      'velocity_tracking_error': 'episode mean XY command L2 error in yaw-only root frame (m/s)',
                      'velocity_tracking': 'episode mean exp(-XY squared error/0.5^2)',
                      'actual_velocity': 'episode mean root forward velocity in yaw-only frame (m/s)',
                      'actual_forward_distance': 'root XY net displacement projected on initial yaw heading (m)',
                      'base_contact': 'original base_contact illegal_contact term (includes non-foot body parts)',
                      'fall': 'base_contact, including simultaneous timeout',
                      'timeout': 'time_out term; truncated is separately read from termination manager',
                      'episode_return': 'sum of original dt-scaled rewards, including terminal transition'})
    app = None
    try:
        app = AppLauncher(args).app
        evaluate(args, report, output)
    except Exception:
        report.update(status='FAIL', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_results(output, report)
        # Kit fast shutdown can terminate the interpreter inside close().
        # Persist and announce the authoritative status before that call.
        print(f"[FLAT EVALUATOR] {report['status']}: {output}", flush=True)
        if app is not None:
            app.close()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
