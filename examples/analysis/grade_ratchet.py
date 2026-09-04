"""Do a model's dreamt snakes hold their length, or only ever lose it?

    python examples/analysis/grade_ratchet.py \
        --models a.pt b.pt --names global flex

Rolls each model forward under the simulator's own actions and reports,
one step at a time, the rate at which snake cells are lost and gained.

The asymmetry is the point. A model that only ever loses cells collapses in
a rollout however small its per-step error, because the shortened snake
becomes its own history and nothing puts a cell back. Frame accuracy cannot
see this: background and wall are most of the picture, and a model can be
right about nearly all of it while its snakes quietly dissolve.

Global, flex and single-agent checkpoints are all accepted and scored the
same way. The policy driving the simulator must be the one the data was
collected with -- with a weaker one the snakes never grow, and then there
is nothing for the measurement to find.
"""
import argparse

import numpy as np
import torch

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.grading.ratchet import REWARD_DICT, measure


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--names', nargs='+', default=None)
    p.add_argument('--policy', default='marlenv/demodata/az_policy.pt',
                   help='the search that drives the simulator; it must be '
                        'the one the data was collected with')
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--bootstrap', type=int, default=6)
    p.add_argument('--episodes', type=int, default=6)
    p.add_argument('--window', type=int, default=48)
    p.add_argument('--denoise-steps', type=int, default=12)
    p.add_argument('--action-steps', type=int, default=4)
    p.add_argument('--simulations', type=int, default=24)
    p.add_argument('--agent', type=int, default=0)
    p.add_argument('--seed', type=int, default=1300)
    p.add_argument('--side', type=int, default=15)
    p.add_argument('--num-agents', type=int, default=3)
    p.add_argument('--num-fruits', type=int, default=4)
    p.add_argument('--view-radius', type=int, default=4)
    p.add_argument('--obstacle-density', type=float, default=0.12)
    p.add_argument('--device', default=None)
    return p.parse_args()


def build_solver(args):
    from marlenv.policies import AlphaZeroSolver, NetworkEvaluator, SnakeNet

    state = torch.load(args.policy, map_location='cpu', weights_only=False)
    net = SnakeNet(channels=state.get('channels', 32),
                   blocks=state.get('blocks', 2))
    net.load_state_dict(state['model'])
    evaluator = NetworkEvaluator(net, device='cpu')
    return lambda episode: AlphaZeroSolver(
        evaluator, objective='sum', num_simulations=args.simulations,
        max_depth=6, max_joint_actions=32, exploration_fraction=0.0,
        seed=episode)


def load(path, args, device):
    """Build a runner factory for whichever kind of checkpoint this is."""
    state = torch.load(path, map_location='cpu', weights_only=False)

    if 'schedule' in state:
        from marlenv.flex_wm.model import load_flex_model
        from marlenv.flex_wm.runner import CachedFlexRunner
        model, _ = load_flex_model(path, device)
        agents = list(range(args.num_agents))
        return (lambda origins: CachedFlexRunner(
            model, agents, origins[0], window=args.window, device=device),
            'flex')

    if 'num_agents' in state:
        from marlenv.wm.marunner import CachedMultiRunner
        from marlenv.wm.multiagent import MultiAgentWorldModel
        model = MultiAgentWorldModel(
            num_agents=state['num_agents'], view=state.get('view', 9),
            num_actions=state.get('num_actions', 4), frame='world',
            dim=state['dim'], depth=state['depth'], heads=state['heads'])
        model.load_state_dict(state['model'])
        model = model.to(device).eval()
        return (lambda origins: CachedMultiRunner(
            model, origins, window=args.window, device=device), 'multi')

    from marlenv.wm.model import WorldModel
    from marlenv.wm.runner import SingleAgentAdapter
    model = WorldModel(
        view=state.get('view', 9), dim=state['dim'], depth=state['depth'],
        heads=state['heads'], num_actions=state.get('num_actions', 4),
        frame=state.get('frame', 'world'),
        align_coords=state.get('align_coords', True))
    model.load_state_dict(state['model'])
    model = model.to(device).eval()
    return (lambda origins: SingleAgentAdapter(
        model, agent=args.agent, window=args.window, device=device),
        'single')


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    names = args.names or args.models
    if len(names) != len(args.models):
        raise SystemExit('--names needs one label per model')

    def make_env(seed):
        env = gym.make(
            'Snake-v1', height=args.side, width=args.side,
            num_snakes=args.num_agents, num_fruits=args.num_fruits,
            reward_dict=REWARD_DICT, view_radius=args.view_radius,
            observation_noise=2.0, snake_noise_sigma=8.0,
            background_gradient=16.0,
            obstacle_density=args.obstacle_density,
            disable_env_checker=True)
        env.reset(seed=seed)
        return env

    solver_for = build_solver(args)
    for name, path in zip(names, args.models):
        make_runner, kind = load(path, args, device)
        tally = measure(make_runner, make_env, solver_for, steps=args.steps,
                        bootstrap=args.bootstrap, episodes=args.episodes,
                        seed=args.seed, denoise_steps=args.denoise_steps,
                        action_steps=args.action_steps, agent=args.agent,
                        device=device)
        tally.report(f'{name}  [{kind}]')
        torch.cuda.empty_cache() if device == 'cuda' else None


if __name__ == '__main__':
    main()
