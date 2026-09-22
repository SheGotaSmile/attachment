"""Opt-in observation of the existing runner; no policy/configuration changes."""

import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import torch


def plain(value):
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def finite(value, label):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"E0 non-finite tensor: {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"E0 non-finite scalar: {label}")


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


class E0TrainingAudit:
    def __init__(self, runner, env, log_dir, reference=None):
        self.runner, self.env = runner, env
        self.path = Path(log_dir) / 'e0_training_audit.json'
        raw = env.unwrapped
        term = raw.action_manager.get_term('joint_pos')
        self.fixed_input = None
        self.data = {'status': 'RUNNING', 'steps': 0, 'updates': [], 'minibatch_loss_checks': 0,
                     'gradient_parameter_checks': 0, 'checkpoints': [], 'starting_iteration': runner.current_learning_iteration,
                     'runtime': {'task': raw.spec.id, 'registered_tasks': sorted(key for key in gym.registry if key.startswith('Instinct-')),
                                 'num_envs': env.num_envs, 'num_actions': env.num_actions,
                                 'observation_terms': env.get_obs_format(), 'resolved_joint_names': term._joint_names,
                                 'action_scale': term._scale, 'action_offset': term._offset,
                                 'physics_dt': raw.physics_dt, 'step_dt': raw.step_dt, 'decimation': raw.cfg.decimation,
                                 'rollout_length': runner.num_steps_per_env,
                                 'actuators': {name: type(actuator).__name__ for name, actuator in raw.scene['robot'].actuators.items()}}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()
        if reference:
            self.verify_loaded(reference)
        self.install()

    def flush(self):
        self.path.write_text(json.dumps(plain(self.data), indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    def deterministic(self, observations):
        actor = self.runner.alg.actor_critic
        modules = [actor, *self.runner.normalizers.values()]
        modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            with torch.inference_mode():
                norm = self.runner.normalizers.get('policy')
                return actor.act_inference(norm(observations) if norm else observations).detach().clone()
        finally:
            for module, mode in zip(modules, modes):
                module.train(mode)

    def verify_loaded(self, reference_path):
        reference = torch.load(reference_path, weights_only=True, map_location='cpu')
        checkpoint_path = Path(reference_path).with_suffix('').with_suffix('')
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        self.fixed_input = reference['observations'].to(self.env.device)
        actions = self.deterministic(self.fixed_input).cpu()
        error = float((actions - reference['actions']).abs().max())
        result = {'checkpoint': str(checkpoint_path), 'reference': str(reference_path), 'max_absolute_error': error,
                  'atol': 1e-6, 'rtol': 1e-5, 'actions_equal': bool(torch.allclose(actions, reference['actions'], atol=1e-6, rtol=1e-5)),
                  'iteration': self.runner.current_learning_iteration,
                  'iteration_equal': self.runner.current_learning_iteration == checkpoint['iter'],
                  'model_equal': equal_state(self.runner.alg.actor_critic.state_dict(), checkpoint['model_state_dict']),
                  'optimizer_equal': equal_state(self.runner.alg.optimizer.state_dict(), checkpoint['optimizer_state_dict']),
                  'normalizers_equal': {name: equal_state(norm.state_dict(), checkpoint[f'{name}_normalizer_state_dict'])
                                        for name, norm in self.runner.normalizers.items()}}
        result['algorithm_learning_rate'] = self.runner.alg.learning_rate
        result['optimizer_learning_rates'] = [group['lr'] for group in self.runner.alg.optimizer.param_groups]
        result['adaptive_learning_rate_equal'] = self.runner.alg.schedule != 'adaptive' or all(
            rate == self.runner.alg.learning_rate for rate in result['optimizer_learning_rates'])
        finite(actions, 'loaded_actions')
        result['status'] = 'PASS' if all([result['actions_equal'], result['iteration_equal'], result['model_equal'],
                                         result['optimizer_equal'], result['adaptive_learning_rate_equal'],
                                         *result['normalizers_equal'].values()]) else 'FAIL'
        self.data['checkpoint_roundtrip'] = result
        self.flush()
        if result['status'] != 'PASS':
            raise AssertionError(f'E0 checkpoint restore mismatch: {result}')
        print('[E0] checkpoint roundtrip:', result, flush=True)

    def install(self):
        original_step = self.env.step
        original_losses = self.runner.alg.compute_losses
        original_gradient = self.runner.alg.gradient_step
        original_update = self.runner.alg.update
        original_save = self.runner.save

        def step(actions):
            finite(actions, 'actions')
            result = original_step(actions)
            obs, rewards, _, infos = result
            finite(infos['observations'], 'observations')
            finite(rewards, 'rewards')
            shapes = {'policy': list(obs.shape), 'critic': list(infos['observations']['critic'].shape), 'action': list(actions.shape)}
            if shapes != {'policy': [4, 96], 'critic': [4, 99], 'action': [4, 29]}:
                raise AssertionError(f'Unexpected runtime shapes: {shapes}')
            self.data['runtime']['shapes'] = shapes
            self.data['steps'] += 1
            if self.fixed_input is None:
                self.fixed_input = obs.detach().clone()
            return result

        def losses(*args, **kwargs):
            result = original_losses(*args, **kwargs)
            finite(result[0], 'minibatch_losses')
            self.data['minibatch_loss_checks'] += 1
            return result

        def gradient(*args, **kwargs):
            result = original_gradient(*args, **kwargs)
            finite(self.runner.alg.actor_critic.state_dict(), 'model_after_gradient')
            self.data['gradient_parameter_checks'] += 1
            return result

        def update(iteration):
            result = original_update(iteration)
            finite(result, 'update_losses_stats')
            finite(self.runner.alg.state_dict(), 'model_optimizer_state')
            for name, norm in self.runner.normalizers.items():
                finite(norm.state_dict(), f'normalizer.{name}')
            self.data['updates'].append({'iteration': iteration, 'losses': plain(result[0]), 'stats': plain(result[1])})
            if self.runner.writer is not None:
                transitions = (iteration + 1) * self.runner.num_steps_per_env * self.env.num_envs
                for name, value in result[0].items():
                    self.runner.writer.add_scalar(f'E0/Loss/{name}', float(value), transitions)
            self.flush()
            print(f'[E0] PPO update {iteration} complete; finite checks passed', flush=True)
            return result

        def save(path, infos=None):
            if self.fixed_input is None:
                raise AssertionError('No real env.step observed before checkpoint save')
            reference = {'observations': self.fixed_input.cpu(), 'actions': self.deterministic(self.fixed_input).cpu()}
            finite(reference, 'pre_save_reference')
            result = original_save(path, infos)
            torch.save(reference, str(path) + '.reference.pt')
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            finite(checkpoint, 'saved_checkpoint')
            expected = ['model_state_dict', 'optimizer_state_dict', 'iter', 'policy_normalizer_state_dict', 'critic_normalizer_state_dict']
            if not all(key in checkpoint for key in expected):
                raise AssertionError(f'Missing checkpoint keys: {set(expected) - checkpoint.keys()}')
            self.data['checkpoints'].append({'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                              'iter': checkpoint['iter'], 'keys': sorted(checkpoint.keys()),
                                              'reference': str(Path(str(path) + '.reference.pt').resolve())})
            self.flush()
            return result

        self.env.step = step
        self.runner.alg.compute_losses = losses
        self.runner.alg.gradient_step = gradient
        self.runner.alg.update = update
        self.runner.save = save

    def finish(self, success):
        self.data.update(status='PASS' if success else 'FAIL', ending_iteration=self.runner.current_learning_iteration,
                         transitions=self.data['steps'] * self.env.num_envs)
        self.flush()
