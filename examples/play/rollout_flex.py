"""Record the flex world action model playing itself, as a gif.

    python examples/play/rollout_flex.py \
        --model marlenv/demodata/flex_wam/model.pt

The same recording as rollout_wam, driven by the pair-based model instead.
An older checkpoint loads here too and behaves exactly as it did: one that
records no attention schedule is read as all-global, which is what a model
trained before scopes existed is.
"""
import argparse

import numpy as np
import torch

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.flex_wm.model import load_flex_model
from marlenv.flex_wm.runner import FlexRunner
from marlenv.wm.data import to_model_input
from marlenv.wm.showreel import (REWARD_DICT, Showreel, compose, save,
                                 world_views)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='marlenv/demodata/flex_wam/model.pt')
    p.add_argument('--out', default='showcase/flex_selfplay.gif')
    p.add_argument('--schedule', default=None,
                   help='override what the checkpoint recorded')
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--bootstrap', type=int, default=12)
    p.add_argument('--checkpoint', default=None,
                   help='AlphaZero network guiding the bootstrap search')
    p.add_argument('--bootstrap-sims', type=int, default=48)
    p.add_argument('--rollout-depth', type=int, default=10)
    p.add_argument('--denoise-steps', type=int, default=12)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--window', type=int, default=None)
    p.add_argument('--decay', type=float, default=0.94)
    p.add_argument('--raw', dest='snap', action='store_false', default=True)
    p.add_argument('--canvas-scale', type=int, default=22)
    p.add_argument('--tile-scale', type=int, default=14)
    p.add_argument('--duration', type=int, default=160)
    p.add_argument('--hold', type=int, default=1200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--side', type=int, default=15)
    p.add_argument('--num-agents', type=int, default=3)
    p.add_argument('--snake-colors', type=int, default=None)
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--view-radius', type=int, default=4)
    p.add_argument('--obstacle-density', type=float, default=0.12)
    p.add_argument('--death-patience', type=int, default=3)
    p.add_argument('--device', default=None)
    return p.parse_args()


def build_solver(args, num_actions):
    """The search that plays the bootstrap prefix."""
    from marlenv.policies import AlphaZeroSolver, RolloutEvaluator

    if args.checkpoint:
        from marlenv.policies import NetworkEvaluator, SnakeNet
        state = torch.load(args.checkpoint, map_location='cpu',
                           weights_only=False)
        net = SnakeNet(channels=state.get('channels', 32),
                       blocks=state.get('blocks', 2))
        net.load_state_dict(state['model'])
        evaluator = NetworkEvaluator(net, device=args.device)
    else:
        evaluator = RolloutEvaluator(num_actions,
                                     rollout_depth=args.rollout_depth,
                                     seed=args.seed)
    return AlphaZeroSolver(evaluator, objective='sum',
                           num_simulations=args.bootstrap_sims,
                           max_depth=6, max_joint_actions=32,
                           exploration_fraction=0.0, seed=args.seed)


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, state = load_flex_model(args.model, device, args.schedule)
    window = args.window or state.get('window') or state.get('context', 48)

    env = gym.make('Snake-v1', height=args.side, width=args.side,
                   num_snakes=args.num_agents, num_fruits=args.num_fruits,
                   reward_dict=REWARD_DICT, view_radius=args.view_radius,
                   observation_noise=2.0, snake_noise_sigma=8.0,
                   background_gradient=16.0,
                   obstacle_density=args.obstacle_density,
                   snake_colors=args.snake_colors,
                   disable_env_checker=True)
    env.reset(seed=args.seed)
    base = env.unwrapped

    heads = np.array([s.head_coord for s in base.snakes], dtype=np.int64)
    runner = FlexRunner(model, agents=list(range(args.num_agents)),
                        positions=heads - heads[0], window=window,
                        device=device, death_patience=args.death_patience)
    runner.reset(torch.from_numpy(
        to_model_input(world_views(env)[None, None])).to(device))

    reel = Showreel(model, env, runner, args.view_radius, args.side,
                    decay=args.decay, snap=args.snap)
    print(f'agents {args.num_agents}   schedule '
          f'{"".join(model.schedule)}   window {window}   '
          f'bootstrap {args.bootstrap}   rollout {args.steps}')

    if args.bootstrap > 1:
        reel.bootstrap(args.bootstrap - 1,
                       build_solver(args, len(base.action_dict)))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    frames = [compose(reel, args.canvas_scale, args.tile_scale)]
    for step in range(args.steps):
        reel.step(denoise_steps=args.denoise_steps,
                  action_steps=args.action_steps, generator=generator)
        frames.append(compose(reel, args.canvas_scale, args.tile_scale))
        if not reel.living:
            print(f'  every viewpoint retired after {step + 1} steps')
            break

    path = save(frames, args.out, duration=args.duration, hold=args.hold)
    print(f'wrote {len(frames)} frames to {path}   '
          f'canvas coverage {reel.canvas.coverage():.2f}')


if __name__ == '__main__':
    main()
