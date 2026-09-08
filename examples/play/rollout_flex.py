"""Record the flex world action model playing itself, as a gif.

    python examples/play/rollout_flex.py \
        --model marlenv/demodata/flex_wam/model.pt

The same recording as rollout_wam, driven by the pair-based model instead.
An older checkpoint loads here too and behaves exactly as it did: one that
records no attention schedule is read as all-global, which is what a model
trained before scopes existed is.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.flex_wm.model import load_flex_model
from marlenv.flex_wm.runner import (CachedFlexRunner,
                                    FlexRunner)
from marlenv.wm.data import to_model_input, to_pixels
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
    p.add_argument('--background-gradient', type=float, default=16.0,
                   help='must match the data the model was trained on. A '
                        'model trained without it reads a gradient as an '
                        'observation it has never seen')
    p.add_argument('--observation-noise', type=float, default=2.0)
    p.add_argument('--snake-noise', type=float, default=8.0)
    p.add_argument('--noise-period', type=int, default=8,
                   help='match the collection setting; the six demo models use 3')
    p.add_argument('--obstacle-density', type=float, default=0.12)
    p.add_argument('--death-patience', type=int, default=3)
    p.add_argument('--no-cache', dest='use_cache',
                   action='store_false', default=True,
                   help='recompute the window every pass instead of\n'
                        'encoding each committed step once')
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
    if not args.checkpoint:
        print('note: no --checkpoint, so the bootstrap prefix is played by '
              'random rollouts.\n      Those rarely take fruit, so the '
              'snakes stay short and the model is handed\n      a prefix '
              'unlike anything it trained on. Pass the policy the data '
              'was\n      collected with to avoid it.')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, state = load_flex_model(args.model, device, args.schedule)
    torch.manual_seed(args.seed)
    window = args.window or state.get('window') or state.get('context', 48)

    env = gym.make('Snake-v1', height=args.side, width=args.side,
                   num_snakes=args.num_agents, num_fruits=args.num_fruits,
                   reward_dict=REWARD_DICT, view_radius=args.view_radius,
                   observation_noise=args.observation_noise,
                   snake_noise_sigma=args.snake_noise,
                   noise_period=args.noise_period,
                   background_gradient=args.background_gradient,
                   obstacle_density=args.obstacle_density,
                   snake_colors=args.snake_colors,
                   disable_env_checker=True)
    env.reset(seed=args.seed)
    base = env.unwrapped

    heads = np.array([s.head_coord for s in base.snakes], dtype=np.int64)
    runner_class = (CachedFlexRunner if args.use_cache
                    else FlexRunner)
    runner = runner_class(model, agents=list(range(args.num_agents)),
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

    # Hash the actual prefix, rather than assuming equal seeds made it equal.
    prefix = {name: getattr(runner.pairs, name).cpu().numpy().copy()
              for name in ('observations', 'actions', 'agent', 'time',
                           'position', 'acted')}
    prefix['actions_known'] = runner.actions_known.cpu().numpy().copy()
    prefix['alive'] = np.array(runner.live, bool)
    prefix_hash = hashlib.sha256()
    for name, values in sorted(prefix.items()):
        prefix_hash.update(name.encode())
        prefix_hash.update(str(values.shape).encode())
        prefix_hash.update(values.tobytes())
    trace = []
    last = {}
    for slot in range(runner.pairs.pairs):
        identity = int(prefix['agent'][0, slot])
        last[identity] = len(trace)
        trace.append(dict(agent=identity, time=int(prefix['time'][0, slot]),
                          position=prefix['position'][0, slot],
                          observation=to_pixels(prefix['observations'][0, slot]),
                          action=(int(prefix['actions'][0, slot])
                                  if prefix['actions_known'][0, slot] else -1)))
    bootstrap_steps = reel.steps
    retirement = {str(i): None for i in range(args.num_agents)}
    for i, alive in enumerate(runner.live):
        if not alive:
            retirement[str(i)] = (int(trace[last[i]]['time']) if i in last
                                  else 'before_retained_prefix')
    generator = torch.Generator(device=device).manual_seed(args.seed)
    frames = [compose(reel, args.canvas_scale, args.tile_scale)]
    for step in range(args.steps):
        if not reel.living:
            break
        moving = reel.living
        actions = reel.step(denoise_steps=args.denoise_steps,
                            action_steps=args.action_steps, generator=generator)
        pixels = reel.dream()
        for i in moving:
            trace[last[i]]['action'] = int(actions[i])
            last[i] = len(trace)
            trace.append(dict(agent=i, time=runner.time,
                              position=runner.position[i].cpu().numpy().copy(),
                              observation=pixels[i], action=-1))
            if not runner.live[i]:
                retirement[str(i)] = runner.time
        frames.append(compose(reel, args.canvas_scale, args.tile_scale))
        if not reel.living:
            print(f'  every viewpoint retired after {step + 1} steps')
            break

    path = save(frames, args.out, duration=args.duration, hold=args.hold)
    print(f'wrote {len(frames)} frames to {path}   '
          f'canvas coverage {reel.canvas.coverage():.2f}')
    out = Path(path)
    np.savez_compressed(out.with_suffix('.prefix.npz'), **prefix)
    np.savez_compressed(out.with_suffix('.rollout.npz'),
                        **{key: np.array([row[key] for row in trace])
                           for key in trace[0]})

    def digest(path):
        h = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    repo = Path(__file__).resolve().parents[2]
    metadata = dict(
        settings=vars(args), device=device, schedule=''.join(model.schedule),
        window=window, bootstrap_steps=bootstrap_steps,
        generated_steps=len(frames) - 1, retirement_step=retirement,
        prefix_sha256=prefix_hash.hexdigest(), model_sha256=digest(args.model),
        policy_sha256=digest(args.checkpoint) if args.checkpoint else None,
        gif_sha256=digest(out),
        trajectory_sha256=digest(out.with_suffix('.rollout.npz')),
        code_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        code_dirty=bool(subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            cwd=repo, text=True).strip()),
        torch_version=torch.__version__, numpy_version=np.__version__,
        trace_format='One row per emitted observation; action=-1 means no '
                     'outgoing action (death or end of recording). '
                     'Observations are uint8 north-up RGB, positions relative '
                     'to agent 0 at reset. No rows follow an agent death.')
    out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    env.close()


if __name__ == '__main__':
    main()
