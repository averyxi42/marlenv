"""Record the world action model playing itself, and write it out as a gif.

    python examples/play/rollout_wam.py --model marlenv/demodata/wam_tuned/model.pt

Nobody steers. The rollout is bootstrapped with real steps from the
simulator so the model starts from something in distribution, and every
step after that is the model's own: it samples the joint action and then
generates the frames that follow.

The gif shows the stitched map across the top -- each agent's view pasted
where dead reckoning says it was taken, fading with age -- and the agents'
own generated views along the bottom, which is everything the model
actually sees.
"""
import argparse

import numpy as np
import torch

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.wm.data import to_model_input
from marlenv.wm.marunner import CachedMultiRunner, MultiAgentRunner
from marlenv.wm.multiagent import MultiAgentWorldModel
from marlenv.wm.showreel import (REWARD_DICT, Showreel, compose, save,
                                 world_views)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='marlenv/demodata/wam_tuned/model.pt')
    p.add_argument('--out', default='showcase/wam_selfplay.gif')
    p.add_argument('--steps', type=int, default=80,
                   help='steps the model drives itself for')
    p.add_argument('--bootstrap', type=int, default=12,
                   help='real steps fed in first; 1 hands over at once')
    p.add_argument('--checkpoint', default=None,
                   help='AlphaZero network guiding the bootstrap search')
    p.add_argument('--bootstrap-sims', type=int, default=48)
    p.add_argument('--rollout-depth', type=int, default=10)
    p.add_argument('--denoise-steps', type=int, default=12)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--window', type=int, default=None)
    p.add_argument('--decay', type=float, default=0.94,
                   help='canvas fade per step; 1.0 keeps everything bright')
    p.add_argument('--raw', dest='snap', action='store_false', default=True,
                   help='show the generated pixels rather than snapping '
                        'them to the palette')
    p.add_argument('--canvas-scale', type=int, default=22)
    p.add_argument('--tile-scale', type=int, default=14)
    p.add_argument('--duration', type=int, default=160,
                   help='milliseconds per gif frame')
    p.add_argument('--hold', type=int, default=1200,
                   help='milliseconds to hold the final frame, so the end '
                        'of a loop is readable')
    p.add_argument('--no-cache', dest='use_cache', action='store_false',
                   default=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--side', type=int, default=15)
    p.add_argument('--num-agents', type=int, default=None)
    p.add_argument('--snake-colors', type=int, default=None,
                   help='distinct snake colours to use, wrapped around. A '
                        'model only ever saw as many colours as it had '
                        'snakes in training, so a fourth snake in a fresh '
                        'hue is out of distribution; cycling keeps the '
                        'board inside what it knows')
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--view-radius', type=int, default=4)
    p.add_argument('--background-gradient', type=float, default=16.0,
                   help='must match the data the model was trained on. A '
                        'model trained without it reads a gradient as an '
                        'observation it has never seen')
    p.add_argument('--observation-noise', type=float, default=2.0)
    p.add_argument('--snake-noise', type=float, default=8.0)
    p.add_argument('--obstacle-density', type=float, default=0.12)
    p.add_argument('--death-patience', type=int, default=3)
    p.add_argument('--device', default=None)
    return p.parse_args()


def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = MultiAgentWorldModel(
        num_agents=state['num_agents'], view=state.get('view', 9),
        num_actions=state.get('num_actions', 4), frame='world',
        dim=state['dim'], depth=state['depth'], heads=state['heads'])
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state.get('context', 24)


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
    if not args.checkpoint:
        print('note: no --checkpoint, so the bootstrap prefix is played by '
              'random rollouts.\n      Those rarely take fruit, so the '
              'snakes stay short and the model is handed\n      a prefix '
              'unlike anything it trained on. Pass the policy the data '
              'was\n      collected with to avoid it.')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, context = load_model(args.model, device)
    agents = args.num_agents or model.num_agents

    env = gym.make('Snake-v1', height=args.side, width=args.side,
                   num_snakes=agents, num_fruits=args.num_fruits,
                   reward_dict=REWARD_DICT, view_radius=args.view_radius,
                   observation_noise=args.observation_noise,
                   snake_noise_sigma=args.snake_noise,
                   background_gradient=args.background_gradient,
                   obstacle_density=args.obstacle_density,
                   snake_colors=args.snake_colors,
                   disable_env_checker=True)
    env.reset(seed=args.seed)
    base = env.unwrapped

    heads = np.array([s.head_coord for s in base.snakes], dtype=np.int64)
    runner_class = CachedMultiRunner if args.use_cache else MultiAgentRunner
    runner = runner_class(model, torch.from_numpy(heads - heads[0])[None],
                          window=args.window or context, device=device,
                          num_agents=agents,
                          death_patience=args.death_patience)

    runner.reset(torch.from_numpy(
        to_model_input(world_views(env)[None, None])).to(device))
    reel = Showreel(model, env, runner, args.view_radius, args.side,
                    decay=args.decay, snap=args.snap)

    print(f'agents {agents}   context {args.window or context}   '
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
