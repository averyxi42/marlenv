"""Collect multi-agent episodes into a HuggingFace dataset.

    python examples/collect_dataset.py --preset expert --episodes 1200 \
        --checkpoint az_obs_latest.pt --workers 20

Components are collected *separately*, one dataset per behaviour policy, so
that training recipes can mix them afterwards in whatever proportion. Mixing
at collection time bakes in a ratio that cannot be undone.

    expert   the trained search policy, no exploration noise
    explore  the same policy with epsilon-random actions, so the data covers
             states a competent policy never reaches -- a world model still
             has to predict driving into a wall
    random   uniform actions; short, death-heavy episodes
"""
import argparse
import os
import time

import numpy as np

from marlenv.data import CollectConfig, collect_dataset, decode_episode

PRESETS = {
    'expert': dict(epsilon=0.0, use_checkpoint=True, first_seed=0),
    'explore': dict(epsilon=0.15, use_checkpoint=True, first_seed=100_000),
    'random': dict(epsilon=0.0, use_checkpoint=False, first_seed=200_000),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=sorted(PRESETS), default='expert')
    p.add_argument('--out', default=None,
                   help='defaults to marlenv/demodata/<preset>')
    p.add_argument('--episodes', type=int, default=64)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=80)
    p.add_argument('--num-snakes', type=int, default=3)
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--side', type=int, default=15)
    p.add_argument('--view-radius', type=int, default=4)
    p.add_argument('--obstacle-density', type=float, default=0.12)
    p.add_argument('--observation-noise', type=float, default=2.0)
    p.add_argument('--snake-noise', type=float, default=8.0)
    p.add_argument('--background-gradient', type=float, default=16.0)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--num-simulations', type=int, default=16)
    p.add_argument('--max-joint-actions', type=int, default=4)
    p.add_argument('--keep-shards', action='store_true')
    return p.parse_args()


def build_config(args, preset):
    return CollectConfig(
        height=args.side, width=args.side, num_snakes=args.num_snakes,
        num_fruits=args.num_fruits, view_radius=args.view_radius,
        obstacle_density=args.obstacle_density,
        observation_noise=args.observation_noise,
        snake_noise_sigma=args.snake_noise,
        background_gradient=args.background_gradient,
        max_steps=args.max_steps,
        checkpoint=args.checkpoint if preset['use_checkpoint'] else None,
        num_simulations=args.num_simulations,
        max_joint_actions=args.max_joint_actions,
        epsilon=preset['epsilon'])


def summarise(dataset, out, seconds):
    steps = np.array(dataset['steps'])
    agents = int(dataset['num_agents'][0])
    stored = sum(os.path.getsize(os.path.join(root, name))
                 for root, _, names in os.walk(out) for name in names)
    transitions = int(steps.sum())
    alive = np.array([decode_episode(row)['alive_mask'][-1].sum()
                      for row in dataset.select(range(min(64,
                                                          len(dataset))))])
    print(f'  episodes      {len(dataset)}')
    print(f'  transitions   {transitions} '
          f'(mean {steps.mean():.1f}, max {steps.max()})')
    print(f'  agent frames  {transitions * agents}')
    print(f'  alive at end  {alive.mean():.2f} of {agents} (first 64 eps)')
    print(f'  on disk       {stored / 1e6:.0f} MB '
          f'({stored / max(transitions, 1):.0f} B per transition)')
    print(f'  collected in  {seconds:.0f}s '
          f'({transitions / seconds:.0f} transitions/s)')


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    if preset['use_checkpoint'] and not args.checkpoint:
        raise SystemExit(f'--preset {args.preset} needs --checkpoint')

    out = args.out or os.path.join('marlenv/demodata', args.preset)
    shards = os.path.join(out, '_shards')
    config = build_config(args, preset)

    print(f'collecting {args.episodes} {args.preset} episodes '
          f'on {args.workers} workers -> {out}')
    start = time.time()
    dataset = collect_dataset(config, args.episodes, shards,
                              workers=args.workers,
                              first_seed=preset['first_seed'])
    dataset.save_to_disk(out)
    seconds = time.time() - start

    if not args.keep_shards:
        import shutil
        shutil.rmtree(shards, ignore_errors=True)
    summarise(dataset, out, seconds)


if __name__ == '__main__':
    main()
