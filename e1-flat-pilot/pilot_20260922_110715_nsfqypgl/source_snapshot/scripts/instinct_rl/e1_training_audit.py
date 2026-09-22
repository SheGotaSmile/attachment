"""Opt-in Flat pilot diagnostics; does not change task or optimizer operations."""

import hashlib
import json
import time
from collections import deque
from pathlib import Path

import torch

from e0_training_audit import finite, plain
from flat_metrics import measurements


class E1TrainingAudit:
    def __init__(self, runner, env, log_dir):
        self.runner, self.env = runner, env
        self.path = Path(log_dir) / 'e1_training_audit.json'
        raw = env.unwrapped
        assert raw.spec.id == 'Instinct-Locomotion-Flat-G1-v0'
        assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
        assert not runner.cfg.get('resume') and not runner.cfg.get('load_run')
        self.started = self.last_update = time.perf_counter()
        self.sums = {}
        self.returns = torch.zeros(env.num_envs, device=env.device)
        self.recent_returns = deque(maxlen=100)
        self.data = dict(status='RUNNING', starting_iteration=0, random_initialization=True,
                         checkpoint_loaded=False, steps=0, minibatch_loss_checks=0,
                         gradient_checks=0, updates=[], checkpoints=[], runtime={
                             'num_envs': env.num_envs, 'rollout_length': runner.num_steps_per_env,
                             'observation_terms': env.get_obs_format(), 'physics_dt': raw.physics_dt,
                             'step_dt': raw.step_dt, 'decimation': raw.cfg.decimation})
        # Preserve this run's random model as evidence; never read an E0 checkpoint.
        runner.save(str(Path(log_dir) / 'model_initial.pt'))
        self.data['initial_checkpoint_sha256'] = hashlib.sha256(
            (Path(log_dir) / 'model_initial.pt').read_bytes()).hexdigest()
        torch.cuda.reset_peak_memory_stats()
        self.install()
        self.flush()

    def flush(self):
        self.path.write_text(json.dumps(plain(self.data), indent=2, allow_nan=False) + '\n')

    def install(self):
        raw, runner = self.env.unwrapped, self.runner
        reward_compute, step = raw.reward_manager.compute, self.env.step
        update, losses, save = runner.alg.update, runner.alg.compute_losses, runner.save

        def compute(*args, **kwargs):
            reward = reward_compute(*args, **kwargs)
            values = measurements(raw)
            values['mean_step_reward'] = reward.reshape(raw.num_envs, -1).sum(-1)
            finite(values, 'E1.pre_reset_metrics')
            for name, value in values.items():
                self.sums[name] = self.sums.get(name, 0.0) + float(value.float().mean())
            self.returns += values['mean_step_reward']
            done = values['terminated'] | values['truncated']
            self.recent_returns.extend(self.returns[done].cpu().tolist())
            self.returns[done] = 0
            return reward

        def checked_step(actions):
            finite(actions, 'E1.actions')
            result = step(actions)
            finite(result[0], 'E1.policy_obs')
            finite(result[1], 'E1.rewards')
            finite(result[3]['observations'], 'E1.all_obs')
            self.data['steps'] += 1
            shapes = {'policy': list(result[0].shape),
                      'critic': list(result[3]['observations']['critic'].shape), 'action': list(actions.shape)}
            assert shapes == {'policy': [raw.num_envs, 96], 'critic': [raw.num_envs, 99],
                              'action': [raw.num_envs, 29]}, shapes
            self.data['runtime']['shapes'] = shapes
            return result

        def checked_losses(*args, **kwargs):
            result = losses(*args, **kwargs)
            finite(result, 'E1.minibatch_losses_and_intermediates')
            self.data['minibatch_loss_checks'] += 1
            return result

        def before_optimizer(optimizer, args, kwargs):
            gradients = [p.grad for p in runner.alg.actor_critic.parameters() if p.grad is not None]
            assert gradients
            finite(gradients, 'E1.actual_gradients_before_optimizer')
            finite([g['lr'] for g in optimizer.param_groups], 'E1.optimizer_lr')
            self.data['gradient_checks'] += 1

        self.gradient_hook = runner.alg.optimizer.register_step_pre_hook(before_optimizer)

        def checked_update(iteration):
            result = update(iteration)
            finite(result, 'E1.update')
            finite(runner.alg.state_dict(), 'E1.model_optimizer')
            for name, norm in runner.normalizers.items():
                finite(norm.state_dict(), f'E1.normalizer.{name}')
            now = time.perf_counter()
            metrics = {k: v / runner.num_steps_per_env for k, v in self.sums.items()}
            metrics['mean_episode_return_last100'] = (
                sum(self.recent_returns) / len(self.recent_returns) if self.recent_returns else None)
            metrics['action_standard_deviation'] = float(runner.alg.actor_critic.std.mean())
            metrics['learning_rate'] = float(runner.alg.learning_rate)
            metrics['transitions_per_second'] = raw.num_envs * runner.num_steps_per_env / (now-self.last_update)
            entry = dict(iteration=iteration, metrics=metrics, losses=plain(result[0]), stats=plain(result[1]))
            finite(entry, 'E1.update_entry')
            self.data['updates'].append(entry)
            if runner.writer:
                for key, value in metrics.items():
                    if value is not None:
                        runner.writer.add_scalar(f'E1/{key}', value, iteration)
                for key, value in result[0].items():
                    runner.writer.add_scalar(f'E1/Loss/{key}', float(value), iteration)
                for key, value in result[1].items():
                    runner.writer.add_scalar(f'E1/Stats/{key}', float(value), iteration)
            self.sums.clear()
            self.last_update = now
            self.flush()
            if (iteration + 1) % 10 == 0:
                print(f'[E1 AUDIT] {iteration+1}/200 updates finite; {metrics}', flush=True)
            return result

        def checked_save(path, infos=None):
            result = save(path, infos)
            state = torch.load(path, map_location='cpu', weights_only=True)
            finite(state, 'E1.saved_checkpoint')
            self.data['checkpoints'].append({'path': str(Path(path).resolve()), 'iter': state['iter'],
                'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
            self.flush()
            return result

        raw.reward_manager.compute = compute
        self.env.step = checked_step
        runner.alg.compute_losses = checked_losses
        runner.alg.update = checked_update
        runner.save = checked_save

    def finish(self, success):
        elapsed = time.perf_counter() - self.started
        self.data.update(status='PASS' if success else 'FAIL',
            ending_iteration=self.runner.current_learning_iteration,
            actual_transitions=self.data['steps'] * self.env.num_envs,
            elapsed_training_seconds=elapsed,
            transitions_per_second=self.data['steps'] * self.env.num_envs / elapsed,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        self.flush()
