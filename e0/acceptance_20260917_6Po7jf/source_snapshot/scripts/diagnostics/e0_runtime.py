"""Record E0 environment metadata and bounded commands in a persistent directory."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time


def capture(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    return {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def collect(root):
    import torch

    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run_directory": str(root),
                "cwd": os.getcwd(), "python": {"executable": sys.executable, "version": sys.version},
                "platform": platform.platform(), "repositories": {}, "packages": {}, "commands": [],
                "initial_user_file_sha256": {}}
    for repo in (Path.cwd(), Path('/home/xiexuhui/instinct_rl'), Path('/home/xiexuhui/IsaacLab-Instinct')):
        manifest["repositories"][str(repo)] = {
            "commit": capture(["git", "-C", str(repo), "rev-parse", "HEAD"]),
            "status": capture(["git", "-C", str(repo), "status", "--short", "--untracked-files=all"])}
    for name in ("repo_audit.md", "Research Decision Record v0.1.md", "repo_fact_check.md", "experiment_protocol_v0.1.md"):
        path = Path('docs') / name
        manifest["initial_user_file_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for distribution, module in (("torch", "torch"), ("isaacsim", "isaacsim"), ("isaaclab", "isaaclab"),
                                 ("instinct-rl", "instinct_rl"), ("instinctlab", "instinctlab")):
        dist = importlib.metadata.distribution(distribution)
        spec = importlib.util.find_spec(module)
        manifest["packages"][distribution] = {"version": dist.version, "metadata_path": str(dist.locate_file('')),
                                               "module_origin": spec.origin if spec else None}
    manifest["gpu_query"] = capture(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version", "--format=csv"])
    manifest["torch_cuda"] = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                              "available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        torch.ones(1, device='cuda:0').add_(1)
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        manifest["torch_cuda"].update(device=torch.cuda.get_device_name(0), free_bytes=free, total_bytes=total,
                                      allocation_test="PASS")
    manifest["collection_command"] = shlex.join(sys.orig_argv)
    save_manifest(root / 'runtime_manifest.json', manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def run(root, label, timeout, argv):
    if argv and argv[0] == '--':
        argv = argv[1:]
    if not argv or Path(label).name != label:
        raise ValueError('Expected command and simple unique label')
    path = root / 'runtime_manifest.json'
    manifest = json.loads(path.read_text())
    record = {"label": label, "argv": argv, "shell_command": shlex.join(argv), "cwd": os.getcwd(),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "timeout_seconds": timeout,
              "log": str(root / f'{label}.log'), "status": "RUNNING"}
    with open(record['log'], 'x', encoding='utf-8') as log:
        log.write(f"COMMAND: {record['shell_command']}\nCWD: {os.getcwd()}\n")
        log.flush()
        manifest['commands'].append(record)
        save_manifest(path, manifest)
        print(f"Starting {label}; full output: {record['log']}", flush=True)
        start = time.monotonic()
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
            record['status'] = 'PASS' if returncode == 0 else 'FAIL'
        except subprocess.TimeoutExpired:
            record['status'] = 'TIMEOUT'
            os.killpg(proc.pid, signal.SIGINT)
            try:
                returncode = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait()
        record.update(returncode=returncode, elapsed_seconds=time.monotonic() - start,
                      finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        log.write(f"\nE0_COMMAND_RESULT: {json.dumps(record)}\n")
    save_manifest(path, manifest)
    print(json.dumps(record, indent=2), flush=True)
    return 0 if record['status'] == 'PASS' else 1


def finalize(root, training_dir, resume_dir, reset_result):
    """Archive exact, explicitly selected results; never search for a latest run."""
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    manifest_path = root / 'runtime_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    training_path = training_dir / 'e0_training_audit.json'
    resume_path = resume_dir / 'e0_training_audit.json'
    training = json.loads(training_path.read_text())
    resume = json.loads(resume_path.read_text())
    reset = json.loads(reset_result.read_text())
    validation = {'tensorboard': {}, 'configuration_files': {}, 'intermediate_tensorboard': {}}
    for label, directory, audit in (('training', training_dir, training), ('resume', resume_dir, resume)):
        events = EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()
        scalars = {tag: [{'step': item.step, 'value': item.value} for item in events.Scalars(tag)]
                   for tag in events.Tags()['scalars']}
        assert scalars and all(math.isfinite(item['value']) for values in scalars.values() for item in values)
        expected_steps = [(update['iteration'] + 1) * audit['runtime']['rollout_length'] * audit['runtime']['num_envs']
                          for update in audit['updates']]
        for loss in audit['updates'][0]['losses']:
            assert [item['step'] for item in scalars[f'E0/Loss/{loss}']] == expected_steps
        validation['tensorboard'][label] = {
            'status': 'PASS', 'directory': str(directory), 'scalars': scalars,
            'event_files': [str(path) for path in directory.glob('events.out.tfevents.*')]}
        validation['configuration_files'][label] = [
            {'path': str(path), 'size_bytes': path.stat().st_size,
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (directory / 'params' / 'env.yaml', directory / 'params' / 'agent.yaml')]
    # Preserve evidence of the short-run logging problem in earlier attempts.
    for record in manifest['commands']:
        if record['label'] not in ('train_2_updates', 'checkpoint_restore_and_update_fixed'):
            continue
        log_text = Path(record['log']).read_text()
        match = re.search(r"Storing git diff for 'InstinctLab' in: (.+)/git/[^\n]+", log_text)
        if match:
            directory = Path(match.group(1))
            events = EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()
            validation['intermediate_tensorboard'][record['label']] = {
                'directory': str(directory), 'scalar_tags': events.Tags()['scalars']}
    before = torch.load(training['checkpoints'][-1]['path'], weights_only=True, map_location='cpu')
    after = torch.load(resume['checkpoints'][-1]['path'], weights_only=True, map_location='cpu')
    differences = {key: float((value - after['model_state_dict'][key]).abs().max())
                   for key, value in before['model_state_dict'].items() if value.is_floating_point() and value.numel()}
    validation['continued_model_update'] = {'status': 'PASS' if any(differences.values()) else 'FAIL',
                                             'parameter_max_absolute_change': differences}
    assert validation['continued_model_update']['status'] == 'PASS'
    save_manifest(root / 'artifact_validation.json', validation)
    manifest['acceptance'] = {
        'status': 'PASS' if all(data['status'] == 'PASS' for data in (training, resume, reset)) else 'FAIL',
        'training_audit': str(training_path), 'resume_audit': str(resume_path), 'reset_results': str(reset_result),
        'training_transitions': training['transitions'], 'resume_transitions': resume['transitions'],
        'runtime': reset['runtime'], 'reset_checks': {k: v['status'] for k, v in reset['checks'].items()},
        'checkpoint_roundtrip': resume['checkpoint_roundtrip'],
        'training_updates': training['updates'], 'resume_updates': resume['updates'],
        'artifact_validation': str(root / 'artifact_validation.json'),
        'not_run': ['E1-flat formal training', 'slope tasks', 'MoE', 'sensor additions',
                    'GUI/video', 'DDP', 'bitwise physics/RNG continuation', '256-env capacity test']}
    manifest['checkpoint_artifacts'] = []
    for audit in (training, resume):
        for saved in audit['checkpoints']:
            checkpoint_path = Path(saved['path'])
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            assert digest == saved['sha256'], f'Checkpoint hash changed: {checkpoint_path}'
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
            manifest['checkpoint_artifacts'].append({**saved, 'size_bytes': checkpoint_path.stat().st_size,
                'optimizer_parameter_states': len(checkpoint['optimizer_state_dict']['state']),
                'optimizer_steps': sorted({float(state['step']) for state in checkpoint['optimizer_state_dict']['state'].values()}),
                'optimizer_learning_rates': [group['lr'] for group in checkpoint['optimizer_state_dict']['param_groups']],
                'normalizer_counts': {group: int(checkpoint[f'{group}_normalizer_state_dict']['count']) for group in ('policy', 'critic')}})
    manifest['user_files_unchanged'] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
        for name, digest in manifest['initial_user_file_sha256'].items()}
    assert all(manifest['user_files_unchanged'].values()), 'An existing user document changed'
    manifest['repositories_final'] = {
        name: capture(['git', '-C', name, 'status', '--short', '--untracked-files=all'])
        for name in manifest['repositories']}
    manifest['source_snapshot'] = []
    for name in ('scripts/diagnostics/e0_runtime.py', 'scripts/instinct_rl/e0_training_audit.py',
                 'scripts/instinct_rl/e0_reset_probe.py', 'scripts/instinct_rl/train.py', 'docs/E0_report.md'):
        target = root / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
        manifest['source_snapshot'].append({'source': name, 'copy': str(target),
                                            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest['simulator_log_copies'] = []
    for record in manifest['commands']:
        log_text = Path(record['log']).read_text()
        for source in re.findall(r'Logging to file: (\S+?\.log)', log_text):
            source_path = Path(source)
            item = {'label': record['label'], 'source': source, 'available': source_path.is_file()}
            if source_path.is_file():
                target = root / 'simulator_logs' / record['label'] / source_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
                item.update(copy=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            manifest['simulator_log_copies'].append(item)
    manifest['finalized_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    manifest['finalize_command'] = shlex.join(sys.orig_argv)
    manifest['initial_worktree_before_diagnostics'] = {
        'tracked_changes': [],
        'untracked_user_files': list(manifest['initial_user_file_sha256']),
        'note': 'Recorded by initial git status before adding E0 scripts; repositories entry records collection-time state.'}
    save_manifest(manifest_path, manifest)
    commands = ['# E0 executed commands (archive, not a batch replay script)',
                '# cwd: ' + os.getcwd(), manifest['collection_command']]
    commands.extend(record['shell_command'] for record in manifest['commands'])
    commands.append(manifest['finalize_command'])
    (root / 'commands.sh').write_text('\n\n'.join(commands) + '\n')
    print(json.dumps({'status': manifest['acceptance']['status'],
                      'checkpoint_artifacts': manifest['checkpoint_artifacts'],
                      'user_files_unchanged': manifest['user_files_unchanged']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('collect')
    finalizer = sub.add_parser('finalize')
    finalizer.add_argument('--training-dir', type=Path, required=True)
    finalizer.add_argument('--resume-dir', type=Path, required=True)
    finalizer.add_argument('--reset-result', type=Path, required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--label', required=True)
    runner.add_argument('--timeout', type=int, default=300)
    runner.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.operation == 'collect':
        if (root / 'runtime_manifest.json').exists():
            raise FileExistsError('Do not overwrite an existing E0 manifest')
        collect(root)
    elif args.operation == 'run':
        sys.exit(run(root, args.label, args.timeout, args.command))
    else:
        finalize(root, args.training_dir.resolve(), args.resume_dir.resolve(), args.reset_result.resolve())
