"""Collect multi-agent episodes into a HuggingFace dataset.

    python examples/collect_dataset.py --episodes 200 --out marlenv/demodata

Actions come from a random policy by default; pass --checkpoint to drive
the snakes with a trained AlphaZero network instead, which produces longer
episodes and far more interesting data.
"""
import argparse
import os

import numpy as np

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.data import (build_dataset, collect_episode, decode_episode,
                          random_policy)

REWARD_DICT = {'fruit': 1.0, 'kill': 0.0, 'lose': -5.0, 'win': 0.0,
               'time': 0.01}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='marlenv/demodata/episodes')
    p.add_argument('--episodes', type=int, default=64)
    p.add_argument('--max-steps', type=int, default=80)
    p.add_argument('--num-snakes', type=int, default=3)
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--side', type=int, default=15)
    p.add_argument('--view-radius', type=int, default=4)
    p.add_argument('--observation-noise', type=float, default=2.0)
    p.add_argument('--snake-noise', type=float, default=8.0)
    p.add_argument('--background-gradient', type=float, default=16.0)
    p.add_argument('--checkpoint', default=None,
                   help='drive the snakes with a trained network')
    p.add_argument('--num-simulations', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def make_env(args):
    return gym.make(
        'Snake-v1', height=args.side, width=args.side,
        num_snakes=args.num_snakes, num_fruits=args.num_fruits,
        reward_dict=REWARD_DICT, view_radius=args.view_radius,
        observation_noise=args.observation_noise,
        snake_noise_sigma=args.snake_noise,
        background_gradient=args.background_gradient,
        disable_env_checker=True)


def search_policy(args):
    """A trained solver, wrapped to look like a policy function."""
    import torch
    from marlenv.policies import (AlphaZeroSolver, NetworkEvaluator, SnakeNet)

    state = torch.load(args.checkpoint, map_location='cpu',
                       weights_only=False)
    net = SnakeNet(channels=state.get('channels', 32),
                   blocks=state.get('blocks', 2))
    net.load_state_dict(state['model'])
    solver = AlphaZeroSolver(NetworkEvaluator(net),
                             num_simulations=args.num_simulations,
                             exploration_fraction=0.0, seed=args.seed)

    def policy(env):
        return solver.solve(env)
    return policy


def main():
    args = parse_args()
    env = make_env(args)
    rng = np.random.default_rng(args.seed)
    policy = search_policy(args) if args.checkpoint else random_policy(rng)

    rows = []
    for index in range(args.episodes):
        rows.append(collect_episode(env, policy, seed=args.seed + index,
                                    max_steps=args.max_steps))

    dataset = build_dataset(rows)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    dataset.save_to_disk(args.out)

    steps = np.array([row['steps'] for row in rows])
    alive = np.array([row['alive_mask'][-1].sum() for row in rows])
    stored = sum(os.path.getsize(os.path.join(root, name))
                 for root, _, names in os.walk(args.out) for name in names)
    print(f'\n{len(rows)} episodes -> {args.out}')
    print(f'  transitions   {steps.sum()} '
          f'(mean {steps.mean():.1f}, max {steps.max()})')
    print(f'  agents alive at end   mean {alive.mean():.2f} '
          f'of {args.num_snakes}')
    print(f'  on disk       {stored / 1024:.0f} KiB '
          f'({stored / max(steps.sum(), 1):.0f} B per transition)')

    sample = decode_episode(dataset[0])
    print('  columns:')
    for key, value in sorted(sample.items()):
        if isinstance(value, np.ndarray):
            print(f'    {key:18s} {str(value.shape):24s} {value.dtype}')


if __name__ == '__main__':
    main()
