"""Record a rollout and grade predictors against it.

    python examples/grade_rollout.py --out marlenv/demodata

Runs the simulator with noise off over a fixed action sequence, saves the
rollout, then scores two stand-in predictors so the harness can be checked
before a world model exists:

* the recorded observations themselves, which must score a perfect match;
* a deliberately corrupted copy, whose errors should land in the class
  pairs that were corrupted and nowhere else.
"""
import argparse
import os

import numpy as np

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.palette import cell_color
from marlenv.core.snake import Cell
from marlenv.grading import ConfusionMatrix, grade, record_rollout
from marlenv.grading.rollout import save_rollout

REWARD_DICT = {'fruit': 1.0, 'kill': 0.0, 'lose': -5.0, 'win': 0.0,
               'time': 0.01}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='marlenv/demodata')
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--num-snakes', type=int, default=3)
    p.add_argument('--side', type=int, default=15)
    p.add_argument('--view-radius', type=int, default=4)
    p.add_argument('--agent', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--corrupt', type=float, default=0.02,
                   help='fraction of cells the broken predictor gets wrong')
    return p.parse_args()


def make_env(args, noise):
    """Noise off for grading; on to see what a model would be trained from."""
    return gym.make(
        'Snake-v1', height=args.side, width=args.side,
        num_snakes=args.num_snakes, num_fruits=4, reward_dict=REWARD_DICT,
        view_radius=args.view_radius,
        observation_noise=2.0 if noise else 0.0,
        snake_noise_sigma=8.0 if noise else 0.0,
        background_gradient=16.0 if noise else 0.0,
        disable_env_checker=True)


def corrupt(views, fraction, rng):
    """Repaint a fraction of cells as a snake that is not on the board."""
    broken = views.copy()
    intruder = cell_color(Cell.BODY.value, 5).astype(np.uint8)
    count = max(1, int(round(broken[0].size / 3 * fraction)))
    for frame in broken:
        rows = rng.integers(0, frame.shape[0], count)
        cols = rng.integers(0, frame.shape[1], count)
        frame[rows, cols] = intruder
    return broken


def summarise(name, matrix):
    total = matrix.matrix.sum()
    print(f'\n{name}: {matrix.errors} wrong of {total} cells '
          f'({matrix.errors / max(total, 1):.3%})')
    for expected, observed, count in matrix.top_confusions(5):
        print(f'    {expected:>7s} -> {observed:<7s} {count}')


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env = make_env(args, noise=False)
    env.reset(seed=args.seed)
    actions = rng.integers(0, 3, size=(args.steps, args.num_snakes))
    rollout = record_rollout(env, actions, agent=args.agent)

    print(f'recorded {rollout.steps} steps for agent {rollout.agent}  '
          f'(alive at end: {bool(rollout.alive[-1])})')
    drift = rollout.pose_drift()
    if drift:
        print(f'  dead reckoning left the simulator at step {drift[0]} '
              f'(the agent stopped moving)')

    path = os.path.join(args.out, f'rollout_seed{args.seed}.npz')
    save_rollout(path, rollout)
    print(f'  saved {path}')

    # a second copy with noise on, which is what a model would train from
    noisy_env = make_env(args, noise=True)
    noisy_env.reset(seed=args.seed)
    noisy = record_rollout(noisy_env, actions, agent=args.agent)
    noisy_path = os.path.join(args.out, f'rollout_seed{args.seed}_noisy.npz')
    save_rollout(noisy_path, noisy)
    print(f'  saved {noisy_path}')

    for reference in ('local', 'global'):
        summarise(f'perfect predictor vs {reference}',
                  grade(rollout, rollout.local_obs, reference=reference))

    broken = corrupt(rollout.local_obs, args.corrupt, rng)
    summarise('corrupted predictor vs local',
              grade(rollout, broken, reference='local'))

    # the noisy recording must grade identically: snapping removes the noise
    matrix = grade(rollout, noisy.local_obs, reference='local')
    summarise('noisy recording vs clean rollout', matrix)


if __name__ == '__main__':
    main()
