"""Collect simulator recordings, then score independent teacher-forced predictions.

python examples/analysis/grade_teacher_forced.py collect --out /tmp/tf-records
python examples/analysis/grade_teacher_forced.py score --recordings /tmp/tf-records \
    --models marlenv/demodata/flex_ego/model_step24000.pt --out /tmp/tf-scores.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from marlenv.flex_wm.model import load_flex_model
from marlenv.grading.teacher_forced import Recording, Scores, collect_recording, evaluate_recording


def digest(path):
    checksum = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def provenance():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return {'git_revision': result.stdout.strip() or None,
            'python': platform.python_version(), 'numpy': np.__version__,
            'torch': str(torch.__version__)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    collect = sub.add_parser('collect', help='record simulator truth once, without loading a world model')
    collect.add_argument('--out', type=Path, required=True)
    collect.add_argument('--policy', type=Path, default=Path('marlenv/demodata/az_policy.pt'))
    collect.add_argument('--episodes', type=int, default=12)
    collect.add_argument('--steps', type=int, default=60)
    collect.add_argument('--seed', type=int, default=2100)
    collect.add_argument('--side', type=int, default=15)
    collect.add_argument('--num-agents', type=int, default=3)
    collect.add_argument('--num-fruits', type=int, default=4)
    collect.add_argument('--view-radius', type=int, default=4)
    collect.add_argument('--obstacle-density', type=float, default=0.12)
    collect.add_argument('--observation-noise', type=float, default=2.0)
    collect.add_argument('--snake-noise', type=float, default=8.0)
    collect.add_argument('--noise-period', type=int, default=3)
    collect.add_argument('--background-gradient', type=float, default=0.0)
    collect.add_argument('--gradient-period', type=int, default=6)
    collect.add_argument('--simulations', type=int, default=16)
    collect.add_argument('--max-depth', type=int, default=24)
    collect.add_argument('--max-joint-actions', type=int, default=4)
    collect.add_argument('--threads', type=int, default=2)
    score = sub.add_parser('score', help='read saved recordings and independently predict each next step')
    score.add_argument('--recordings', type=Path, required=True)
    score.add_argument('--models', type=Path, nargs='+', required=True)
    score.add_argument('--names', nargs='+')
    score.add_argument('--out', type=Path, required=True)
    score.add_argument('--window', type=int, default=48,
                       help='total frames per query, including the target (default: 47 past + 1 target)')
    score.add_argument('--denoise-steps', type=int, default=12)
    score.add_argument('--first-target', type=int, default=1,
                       help='first scored frame index; earlier frames remain real conditioning')
    score.add_argument('--seed', type=int, default=0)
    score.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    score.add_argument('--threads', type=int, default=2)
    return parser.parse_args()


def collect(args):
    import gymnasium as gym
    import marlenv  # noqa: F401
    from marlenv.policies import AlphaZeroSolver, NetworkEvaluator, SnakeNet

    if args.episodes < 1 or args.steps < 1 or not 1 <= args.num_agents <= 6:
        raise ValueError('need positive episodes/steps and 1 to 6 distinct snake colours')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('recording directory must be empty; recordings are not overwritten')
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(height=args.side, width=args.side, num_snakes=args.num_agents,
                       num_fruits=args.num_fruits, view_radius=args.view_radius,
                       obstacle_density=args.obstacle_density,
                       observation_noise=args.observation_noise,
                       snake_noise_sigma=args.snake_noise, noise_period=args.noise_period,
                       background_gradient=args.background_gradient,
                       gradient_period=args.gradient_period,
                       reward_dict={'fruit': 1., 'kill': 0., 'lose': -5., 'win': 0., 'time': .01})
    search = dict(objective='sum', num_simulations=args.simulations, max_depth=args.max_depth,
                  max_joint_actions=args.max_joint_actions, exploration_fraction=0.)
    state = torch.load(args.policy, map_location='cpu', weights_only=False)
    net = SnakeNet(channels=state.get('channels', 32), blocks=state.get('blocks', 2))
    net.load_state_dict(state['model'])
    evaluator = NetworkEvaluator(net, device='cpu')
    metadata = {'environment': environment, 'search': search,
                'policy': str(args.policy.resolve()), 'policy_sha256': digest(args.policy),
                'requested_steps': args.steps, 'runtime': provenance()}
    files = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        solver = AlphaZeroSolver(evaluator, seed=seed, **search)
        env = gym.make('Snake-v1', **environment, disable_env_checker=True)
        try:
            recording = collect_recording(env, solver.solve, seed=seed,
                                           steps=args.steps, metadata=metadata)
        finally:
            env.close()
        path = args.out / f'episode_{seed}.npz'
        recording.save(path)
        files.append({'file': path.name, 'sha256': digest(path), 'seed': seed,
                      'steps': recording.steps})
        print(f'recorded {path.name}: {recording.steps} transitions', flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(
        {'format_version': 1, 'metadata': metadata, 'recordings': files}, indent=2) + '\n')


def score(args):
    names = args.names or [str(path) for path in args.models]
    if len(names) != len(args.models) or args.window < 2 or args.denoise_steps < 1:
        raise ValueError('invalid model names, window, or denoise step count')
    manifest_path = args.recordings / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    recordings = []
    for entry in manifest['recordings']:
        path = args.recordings / entry['file']
        if digest(path) != entry['sha256']:
            raise ValueError(f'recording checksum mismatch: {path}')
        recordings.append(Recording.load(path))
    if not recordings:
        raise ValueError('manifest contains no recordings')
    output = {'instrument': 'teacher_forced_v1', 'runtime': provenance(),
              'recording_manifest': manifest, 'recording_manifest_sha256': digest(manifest_path),
              'settings': {'window': args.window, 'denoise_steps': args.denoise_steps,
                           'first_target': args.first_target, 'seed': args.seed, 'device': args.device,
                           'conditioning': 'all available real agent views before target',
                           'targets': 'all agents alive before the action, including fatal transitions',
                           'snake_metric': 'exact class matches / cells where either view is snake'},
              'models': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, path in zip(names, args.models):
        model, state = load_flex_model(path, args.device)
        total, episodes = Scores(), []
        started = time.monotonic()
        for recording in recordings:
            scores = evaluate_recording(model, recording, window=args.window,
                                         denoise_steps=args.denoise_steps, seed=args.seed,
                                         first_target=args.first_target)
            total.merge(scores)
            episodes.append({'seed': recording.metadata['seed'], **scores.report()})
            print(f'{name}: seed {recording.metadata["seed"]}, '
                  f'snake consistency={scores.consistency.report()["snake_exact_rate"]}, '
                  f'{time.monotonic() - started:.1f}s', flush=True)
        output['models'].append({'name': name, 'checkpoint': str(path.resolve()),
                                 'checkpoint_sha256': digest(path),
                                 'schedule': ''.join(model.schedule),
                                 'training_context': state.get('context'),
                                 'training_window': state.get('window'),
                                 'scores': total.report(), 'episodes': episodes,
                                 'elapsed_seconds': time.monotonic() - started})
        args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
        del model
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    args = arguments()
    torch.set_num_threads(args.threads)
    {'collect': collect, 'score': score}[args.command](args)
