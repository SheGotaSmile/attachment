"""Validate and archive a completed pilot; never launches or changes training."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def assert_finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    archive = Path(args.archive).resolve()
    runs = list((archive/'train').iterdir())
    assert len(runs) == 1
    run = runs[0]
    audit = json.loads((run/'e1_training_audit.json').read_text())
    manifest = json.loads((archive/'manifest.json').read_text())
    assert manifest['returncode'] == 0 and audit['status'] == 'PASS'
    assert [u['iteration'] for u in audit['updates']] == list(range(200))
    assert audit['actual_transitions'] == 64 * 24 * 200 == 307200
    assert audit['steps'] == 4800
    assert audit['gradient_checks'] == audit['minibatch_loss_checks'] == 4000
    assert_finite(audit)
    protected = manifest['source_hashes']
    assert all(digest(root/p) == h for p, h in protected.items())
    analysis = dict(train_dir=str(run), completed_iterations=200,
        actual_transitions=audit['actual_transitions'], source_files_unchanged=len(protected),
        throughput=audit['transitions_per_second'], training_seconds=audit['elapsed_training_seconds'],
        first_last={}, first20_last20={}, evaluator={}, checkpoint_validation={})
    updates = audit['updates']
    for key in updates[0]['metrics']:
        analysis['first_last'][key] = [updates[0]['metrics'][key], updates[-1]['metrics'][key]]
        analysis['first20_last20'][key] = [statistics.mean(u['metrics'][key] for u in window
            if u['metrics'][key] is not None) for window in [updates[:20], updates[-20:]]]
    for filename in ['model_initial.pt', 'model_200.pt']:
        path = run/filename
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert_finite(state)
        validation = {'path': str(path), 'sha256': digest(path), 'iter': state['iter'], 'keys': sorted(state)}
        validation['normalizer_counts'] = {group: int(state[f'{group}_normalizer_state_dict']['count'])
                                          for group in ['policy','critic']}
        steps = [float(s['step']) for s in state['optimizer_state_dict']['state'].values()]
        validation['optimizer_steps'] = sorted(set(steps))
        analysis['checkpoint_validation'][filename] = validation
    initial = analysis['checkpoint_validation']['model_initial.pt']
    final = analysis['checkpoint_validation']['model_200.pt']
    assert initial['iter'] == 0 and initial['optimizer_steps'] == []
    assert final['iter'] == 200 and final['optimizer_steps'] == [4000.0]
    assert final['sha256'] == audit['checkpoints'][-1]['sha256']
    event = EventAccumulator(str(run), size_guidance={'scalars': 0})
    event.Reload()
    scalars = {tag: [{'step': x.step, 'value': x.value} for x in event.Scalars(tag)]
               for tag in event.Tags()['scalars']}
    assert_finite(scalars)
    assert len(scalars['E1/base_velocity_error']) == 200
    json_write(archive/'tensorboard_scalars.json', scalars)
    analysis['tensorboard'] = {'tags': len(scalars), 'all_finite': True,
                              'E1_points_per_tag': 200, 'runner_last_logged_iteration': 190}
    samples = list(csv.DictReader((archive/'gpu.csv').open()))
    analysis['gpu'] = dict(samples=len(samples), sampling_period_seconds=5,
        sampled_max_device_memory_MiB=max(float(s['memory_used_MiB']) for s in samples),
        total_device_memory_MiB=float(samples[0]['memory_total_MiB']),
        torch_peak_allocated_MiB=audit['torch_peak_allocated_bytes']/1024**2,
        torch_peak_reserved_MiB=audit['torch_peak_reserved_bytes']/1024**2)
    commands = [shlex.join(manifest['argv'])]
    for label in ['evaluation', 'evaluation_v2', 'evaluation_initial']:
        data = json.loads((archive/label/'episodes.json').read_text())
        commands.append(shlex.join([sys.executable, '-B', '-u', *data['argv']]))
        summary = {'status': data['status'], 'pid': data['pid'], 'episodes': len(data['episodes'])}
        if data['status'] == 'PASS':
            rows = list(csv.DictReader((archive/label/'episodes.csv').open()))
            assert len(rows) == len(data['episodes']) == 32
            assert data['completed_per_env'] == [4]*8
            assert all(data['loaded_states_equal'].values()) and all(data['states_unchanged'].values())
            assert data['mean_action_max_error'] == 0.0
            assert data['asynchronous_reset_observed']
            for j, row in enumerate(rows):
                assert int(row['episode_id']) == j
                for k, v in data['episodes'][j].items():
                    assert row[k] == str(v), (j, k)
            assert len({(r['env_id'], r['env_episode_id']) for r in data['episodes']}) == 32
            for key in ['episode_return','episode_length','velocity_tracking_error','velocity_tracking',
                        'base_contact','timeout','fall','actual_forward_distance','actual_velocity','mean_speed_xy']:
                summary[key] = statistics.mean(r[key] for r in data['episodes'])
            summary['episode_return_max_error'] = data['episode_return_max_error']
            summary['states_unchanged'] = data['states_unchanged']
        analysis['evaluator'][label] = summary
    assert analysis['evaluator']['evaluation_v2']['status'] == 'PASS'
    assert analysis['evaluator']['evaluation_initial']['status'] == 'PASS'
    assert analysis['evaluator']['evaluation_v2']['pid'] != manifest['pid']
    (archive/'commands.sh').write_text('# Executed command archive; do not replay over existing outputs.\n\n'+'\n\n'.join(commands)+'\n')
    analysis['criterion4'] = 'NOT_CONFIRMED: velocity scores improve but early-contact termination increases and survival regresses'
    analysis['pilot_status'] = 'NOT_PASSED'
    json_write(archive/'analysis.json', analysis)

    # Copy original simulator logs referenced by the actual process logs.
    for log in [archive/'train.log', *archive.glob('evaluation*.log')]:
        content = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text())
        for name in re.findall(r'Logging to file: (\S+)', content):
            src = Path(name)
            if src.is_file():
                target = archive/'simulator_logs'/log.stem/src.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)

    sources = ['scripts/instinct_rl/train.py', 'scripts/instinct_rl/e1_training_audit.py',
               'scripts/instinct_rl/evaluate_flat.py', 'scripts/instinct_rl/flat_metrics.py',
               'scripts/instinct_rl/e0_training_audit.py', 'scripts/diagnostics/e1_pilot_run.py',
               'scripts/diagnostics/e1_pilot_artifacts.py', 'docs/E0_report.md', 'docs/RESEARCH_STATE.md',
               'docs/E1_flat_pilot_report.md', 'docs/E1_flat_evaluator.md']
    for name in sources:
        source = root/name
        if source.exists():
            target = archive/'source_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    diff = subprocess.run(['git','diff','HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout
    for name in sources:
        tracked = subprocess.run(['git','ls-files','--error-unmatch',name],cwd=root,capture_output=True)
        if tracked.returncode and (root/name).exists():
            result = subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=root,capture_output=True,text=True)
            assert result.returncode in (0,1)
            diff += result.stdout
    (archive/'git_diff.patch').write_text(diff)
    json_write(archive/'repositories_final.json', {str(repo): {
        'commit': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'status': subprocess.check_output(['git','-C',str(repo),'status','--short','--untracked-files=all'],text=True)}
        for repo in [root, root.parent/'instinct_rl', root.parent/'IsaacLab-Instinct']})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(10,6), constrained_layout=True)
    for ax, key, label in zip(axes.flat,
            ['mean_episode_return_last100','base_velocity_error','velocity_tracking','terminated'],
            ['Episode return (last 100)','XY velocity error (m/s)','Velocity tracking score','Terminations per transition']):
        values = [u['metrics'][key] for u in updates]
        ax.plot(range(200),values,alpha=.35,lw=.7)
        ax.plot(range(19,200),[statistics.mean(values[i-19:i+1]) for i in range(19,200)],lw=1.8)
        ax.set(xlabel='PPO update index',ylabel=label)
        ax.grid(alpha=.2)
    fig.suptitle('E1-flat pilot: 64 envs, seed 42, 307,200 transitions\nVelocity scores improve while early termination worsens')
    fig.savefig(archive/'learning_trends.png',dpi=160)
    plt.close(fig)
    hashes = {str(p.relative_to(archive)): digest(p) for p in sorted(archive.rglob('*'))
              if p.is_file() and p.name not in ['SHA256SUMS','artifact_hashes.json']}
    json_write(archive/'artifact_hashes.json',hashes)
    (archive/'SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for p,h in hashes.items()))
    print(json.dumps(analysis,indent=2))


if __name__ == '__main__':
    main()
