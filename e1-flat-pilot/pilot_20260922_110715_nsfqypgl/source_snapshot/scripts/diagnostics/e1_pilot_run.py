"""Run exactly one fresh E1 Flat pilot, preserving command, git state and GPU samples."""

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time


def capture(argv):
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'argv': argv, 'returncode': result.returncode, 'output': result.stdout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, choices=[64, 32], default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    logroot = root / 'logs/e1-flat-pilot'
    logroot.mkdir(parents=True, exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix='pilot_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=logroot))
    command = [sys.executable, '-B', '-u', 'scripts/instinct_rl/train.py',
               '--task', 'Instinct-Locomotion-Flat-G1-v0', '--num_envs', str(args.num_envs),
               '--seed', '42', '--max_iterations', '200', '--headless', '--device', 'cuda:0',
               '--logroot', str(archive / 'train'), '--run_name', 'E1_flat_pilot_s42', '--e1_audit']
    manifest = {'status': 'RUNNING', 'cwd': str(root), 'argv': command,
                'started': datetime.datetime.now().isoformat(), 'repositories': {}}
    for repo in [root, root.parent/'instinct_rl', root.parent/'IsaacLab-Instinct']:
        manifest['repositories'][str(repo)] = {
            'commit': capture(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
            'status': capture(['git', '-C', str(repo), 'status', '--short', '--untracked-files=all'])}
    manifest['source_hashes'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/'source').rglob('*.py'))}
    (archive/'command.sh').write_text(shlex.join(command)+'\n')
    (archive/'git_before.diff').write_text(capture(['git', '-C', str(root), 'diff', 'HEAD'])['output'])
    (archive/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'E1_ARCHIVE={archive}', flush=True)
    manifest['cuda_preflight'] = capture([sys.executable, '-c',
        'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.ones(1,device="cuda:0"))'])
    if manifest['cuda_preflight']['returncode']:
        manifest['status'] = 'BLOCKED_CUDA'
        (archive/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return 1
    with (archive/'train.log').open('w') as log, (archive/'gpu.csv').open('w', newline='') as gpu:
        log.write(shlex.join(command)+'\n')
        log.flush()
        writer = csv.writer(gpu)
        writer.writerow(['wall_time', 'gpu_index', 'memory_used_MiB', 'memory_total_MiB', 'utilization_percent'])
        start = time.monotonic()
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        manifest['pid'] = child.pid
        while child.poll() is None:
            sample = capture(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'])
            if sample['returncode'] == 0:
                for line in sample['output'].strip().splitlines():
                    writer.writerow([datetime.datetime.now().isoformat(), *[s.strip() for s in line.split(',')]])
                gpu.flush()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        manifest.update(returncode=child.returncode, elapsed_seconds=time.monotonic()-start,
                        status='COMPLETED' if child.returncode == 0 else 'FAILED',
                        ended=datetime.datetime.now().isoformat())
    manifest['source_unchanged'] = all(hashlib.sha256((root/p).read_bytes()).hexdigest() == digest
                                      for p, digest in manifest['source_hashes'].items())
    (archive/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ['status', 'returncode', 'elapsed_seconds', 'source_unchanged']}), flush=True)
    return child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
