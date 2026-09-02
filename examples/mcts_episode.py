"""Render a GIF of one episode driven by the MCTS joint-action solver.

    python examples/mcts_episode.py --num-snakes 3 --out episode.gif

Every snake is controlled by a single search that maximises the communal
reward, so ``--communal`` is what changes the group's behaviour: ``sum``
maximises total return, ``min`` favours the worst-off snake.

A short rollout mostly learns to stay alive, since a random rollout rarely
stumbles onto fruit; raise ``--rollout-depth`` to trade runtime for
fruit-seeking.
"""
import argparse

import numpy as np
from PIL import Image

import gymnasium as gym
import marlenv  # noqa: F401  (registers the Snake envs)
from marlenv.policies import MCTSSolver

COMMUNAL_FNS = {
    'sum': sum,
    'min': min,
    'max': max,
    'mean': lambda rewards: float(np.mean(rewards)),
}

REWARD_DICT = {
    'fruit': 1.0,
    'kill': 0.0,
    'lose': -5.0,
    'win': 0.0,
    'time': 0.01,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-snakes', type=int, default=3)
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--height', type=int, default=15)
    p.add_argument('--width', type=int, default=15)
    p.add_argument('--steps', type=int, default=80,
                   help='maximum steps to play')
    p.add_argument('--communal', choices=sorted(COMMUNAL_FNS), default='sum')
    p.add_argument('--num-simulations', type=int, default=80)
    p.add_argument('--max-depth', type=int, default=6)
    p.add_argument('--rollout-depth', type=int, default=10)
    p.add_argument('--max-joint-actions', type=int, default=None,
                   help='cap the branching factor '
                        '(3 ** num_snakes by default)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None,
                   help='output path (default: ./tmp/<timestamp>.gif)')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    env = gym.make(
        'Snake-v1',
        height=args.height,
        width=args.width,
        num_snakes=args.num_snakes,
        num_fruits=args.num_fruits,
        reward_dict=REWARD_DICT,
        disable_env_checker=True,
    )
    solver = MCTSSolver(
        communal_reward_fn=COMMUNAL_FNS[args.communal],
        num_simulations=args.num_simulations,
        max_depth=args.max_depth,
        rollout_depth=args.rollout_depth,
        max_joint_actions=args.max_joint_actions,
        seed=args.seed,
    )

    env.reset(seed=args.seed)
    base = env.unwrapped
    base.render('gif')  # capture the starting position

    returns = np.zeros(args.num_snakes)
    fruits = np.zeros(args.num_snakes)
    for step in range(1, args.steps + 1):
        action = solver.solve(env)
        _, rews, terminated, truncated, info = env.step(action)
        base.render('gif')

        alive = np.array([not done for done in terminated])
        returns += alive * np.asarray(rews)
        # snake.fruit is cleared by move() before step() returns, so read the
        # env's own tally -- which step() hands over in info as it resets it
        fruits = info.get('episode_fruits', base.epi_fruits).copy()
        if not args.quiet:
            print(f'step {step:3d}  action={action}  '
                  f'reward={np.round(rews, 2).tolist()}  '
                  f'alive={base.alive_snakes}/{args.num_snakes}')
        if all(terminated) or all(truncated):
            print(f'episode ended after {step} steps')
            break
    else:
        print(f'reached the {args.steps} step limit')

    print(f'communal={args.communal}  '
          f'returns={np.round(returns, 2).tolist()}  '
          f'total={returns.sum():.2f}  fruits={fruits.astype(int).tolist()}')

    path = base.save_gif(args.out) if args.out else base.save_gif()
    # the GIF encoder merges consecutive identical frames, and a snake
    # rotating within its own cells renders identically, so report what
    # actually landed in the file rather than the buffer length
    buffered = len(base.frame_buffer)
    with Image.open(path) as gif:
        written = gif.n_frames
    print(f'wrote {written} frames to {path}')
    if written != buffered:
        print(f'note: {buffered - written} of {buffered} rendered frames '
              f'were identical to their predecessor and got merged')


if __name__ == '__main__':
    main()
