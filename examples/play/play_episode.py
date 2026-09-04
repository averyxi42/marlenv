"""Play one episode with any of the solvers and render it to a GIF.

    python examples/play/play_episode.py --policy rollout --num-snakes 3
    python examples/play/play_episode.py --policy network --checkpoint az_best.pt

Every snake is driven by a single search over joint actions that maximises
the communal reward, so ``--communal`` is what changes the group's behaviour:
``sum`` maximises total return, ``min`` favours the worst-off snake.

``--policy`` selects what guides that search:

``mcts``
    The standalone rollout solver (UCT, random rollouts).
``uniform``
    PUCT with a uniform prior and zero value -- a control with no guidance.
``rollout``
    PUCT with a uniform prior and a random-rollout value.
``network``
    PUCT with the learned prior and value from ``--checkpoint``.

With a random rollout the search mostly learns to stay alive, since a random
rollout rarely stumbles onto fruit; raise ``--rollout-depth`` to trade
runtime for fruit-seeking, or use a trained ``--policy network``.
"""
import argparse

import numpy as np
from PIL import Image

import gymnasium as gym
import marlenv  # noqa: F401  (registers the Snake envs)
from marlenv.policies import (MCTSSolver, RolloutEvaluator,
                              UniformEvaluator)

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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy', default='rollout',
                   choices=['mcts', 'uniform', 'rollout', 'network'])
    p.add_argument('--checkpoint', default=None,
                   help='required for --policy network')
    p.add_argument('--device', default=None)
    p.add_argument('--num-snakes', type=int, default=3)
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--view-radius', type=int, default=None,
                   help='egocentric RGB view radius; also writes a second '
                        'gif tiling every agent\'s own view')
    p.add_argument('--tile-scale', type=int, default=12,
                   help='pixels per cell in the tiled observation gif')
    p.add_argument('--observation-noise', type=float, default=2.0,
                   help='sigma of the bound RGB observation noise '
                        '(classic style; 0 disables)')
    p.add_argument('--gradient-angle', type=float, default=0.0,
                   help='direction of the gradient stripes, in degrees; '
                        '0 runs along the rows')
    p.add_argument('--background-gradient', type=float, default=16.0,
                   help='amplitude of the world-anchored heading gradient '
                        '(0 disables)')
    p.add_argument('--gradient-period', type=int, default=6,
                   help='period of the heading gradient, in cells')
    p.add_argument('--snake-noise', type=float, default=8.0,
                   help='sigma of the per-snake noise; defaults to '
                        '--observation-noise')
    p.add_argument('--noise-period', type=int, default=3,
                   help='body-distance period of the per-snake noise')
    p.add_argument('--obstacle-density', type=float, default=0.0,
                   help='fraction of interior cells walled off')
    p.add_argument('--grid-size-range', type=int, nargs=2, default=None,
                   metavar=('LOW', 'HIGH'),
                   help='sample a square board size per episode')
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
    p.add_argument('--style', default='pixel', choices=['classic', 'pixel'],
                   help='pixel is the retro renderer, which also shows each '
                        'segment direction (default)')
    p.add_argument('--cell-size', type=int, default=16,
                   help='pixels per grid cell, for --style pixel')
    p.add_argument('--out', default=None,
                   help='output path (default: ./tmp/<timestamp>.gif)')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    if args.policy == 'network' and not args.checkpoint:
        p.error('--policy network needs --checkpoint')
    return args


def build_solver(args, num_actions=3):
    """Assemble the requested search. Only the evaluator differs."""
    if args.policy == 'mcts':
        return MCTSSolver(
            communal_reward_fn=COMMUNAL_FNS[args.communal],
            num_simulations=args.num_simulations,
            max_depth=args.max_depth,
            rollout_depth=args.rollout_depth,
            max_joint_actions=args.max_joint_actions,
            seed=args.seed)

    # every other policy is the same PUCT search with a different evaluator
    from marlenv.policies import AlphaZeroSolver
    if args.policy == 'uniform':
        evaluator = UniformEvaluator(num_actions)
    elif args.policy == 'rollout':
        evaluator = RolloutEvaluator(num_actions,
                                     rollout_depth=args.rollout_depth,
                                     seed=args.seed)
    else:
        evaluator = load_network(args)
    return AlphaZeroSolver(
        evaluator, objective=args.communal,
        num_simulations=args.num_simulations,
        max_depth=args.max_depth,
        max_joint_actions=args.max_joint_actions or 32,
        exploration_fraction=0.0,  # play greedily
        seed=args.seed)


def load_network(args):
    """Rebuild the trained network described by a checkpoint."""
    import torch
    from marlenv.policies import NetworkEvaluator, SnakeNet

    state = torch.load(args.checkpoint, map_location='cpu',
                       weights_only=False)
    net = SnakeNet(channels=state.get('channels', 32),
                   blocks=state.get('blocks', 2))
    net.load_state_dict(state['model'])
    saved = state.get('objective')
    if saved and saved != args.communal:
        print(f'note: checkpoint was trained on --communal {saved}, '
              f'playing with {args.communal}')
    return NetworkEvaluator(net, device=args.device)


def tile_views(views, alive, scale, gap=4):
    """Lay every agent's view out in a row, upscaled, dead ones dimmed."""
    tiles = []
    for view, is_alive in zip(views, alive):
        tile = np.repeat(np.repeat(view, scale, axis=0), scale, axis=1)
        if not is_alive:
            tile = (tile * 0.25).astype(np.uint8)
        tiles.append(tile)

    height = tiles[0].shape[0]
    width = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    sheet[:] = (24, 24, 28)
    x = 0
    for tile in tiles:
        sheet[:, x:x + tile.shape[1]] = tile
        x += tile.shape[1] + gap
    return sheet


def main():
    args = parse_args()

    env = gym.make(
        'Snake-v1',
        height=args.height,
        width=args.width,
        num_snakes=args.num_snakes,
        num_fruits=args.num_fruits,
        reward_dict=REWARD_DICT,
        render_style=args.style,
        cell_size=args.cell_size,
        view_radius=args.view_radius,
        observation_noise=args.observation_noise,
        snake_noise_sigma=args.snake_noise,
        background_gradient=args.background_gradient,
        gradient_period=args.gradient_period,
        gradient_angle=args.gradient_angle,
        noise_period=args.noise_period,
        obstacle_density=args.obstacle_density,
        grid_size_range=(tuple(args.grid_size_range)
                         if args.grid_size_range else None),
        disable_env_checker=True,
    )
    solver = build_solver(args, num_actions=len(env.unwrapped.action_dict))

    env.reset(seed=args.seed)
    base = env.unwrapped
    base.render('gif')  # capture the starting position

    returns = np.zeros(args.num_snakes)
    fruits = np.zeros(args.num_snakes)
    observation_frames = []

    def capture_views():
        if args.view_radius:
            observation_frames.append(
                tile_views(base.egocentric_rgb(),
                           [s.alive for s in base.snakes], args.tile_scale))

    capture_views()
    for step in range(1, args.steps + 1):
        action = solver.solve(env)
        _, rews, terminated, truncated, info = env.step(action)
        base.render('gif')
        capture_views()

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

    print(f'policy={args.policy}  style={args.style}  '
          f'communal={args.communal}  '
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
    if observation_frames:
        stem, _, ext = path.rpartition('.')
        obs_path = f'{stem}_obs.{ext or "gif"}'
        images = [Image.fromarray(f, 'RGB') for f in observation_frames]
        images[0].save(obs_path, save_all=True, append_images=images[1:],
                       format='GIF', loop=0)
        print(f'wrote {len(images)} tiled observation frames to {obs_path}')

    if written != buffered:
        print(f'note: {buffered - written} of {buffered} rendered frames '
              f'were identical to their predecessor and got merged')


if __name__ == '__main__':
    main()
