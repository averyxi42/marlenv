"""Train the factorised AlphaZero solver on a communal reward.

    python examples/train/train_alphazero.py --objective sum --iterations 60

Each iteration plays a few self-play episodes under the search, adds the
positions to a replay buffer, and takes some gradient steps. Every snake is
driven by one shared network, so the same weights work for any number of
snakes; ``--eval-num-snakes`` exercises that.
"""
import argparse
import json
import time

import numpy as np
import torch

import gymnasium as gym
import marlenv  # noqa: F401  (registers the Snake envs)
from marlenv.policies import AlphaZeroSolver, NetworkEvaluator, SnakeNet
from marlenv.policies.objectives import get_objective
from marlenv.policies.training import (ReplayBuffer, self_play_episode,
                                       train_step)

REWARD_DICT = {
    'fruit': 1.0,
    'kill': 0.0,
    'lose': -5.0,
    'win': 0.0,
    'time': 0.01,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objective', default='sum',
                   choices=['sum', 'mean', 'min', 'max'])
    p.add_argument('--iterations', type=int, default=60)
    p.add_argument('--episodes-per-iter', type=int, default=4)
    p.add_argument('--train-steps-per-iter', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-simulations', type=int, default=32)
    p.add_argument('--max-steps', type=int, default=40)
    p.add_argument('--num-snakes', type=int, default=2)
    p.add_argument('--eval-num-snakes', type=int, nargs='*', default=None,
                   help='snake counts to evaluate on (default: --num-snakes)')
    p.add_argument('--height', type=int, default=11)
    p.add_argument('--width', type=int, default=11)
    p.add_argument('--num-fruits', type=int, default=3)
    p.add_argument('--obstacle-density', type=float, default=0.0,
                   help='fraction of interior cells walled off')
    p.add_argument('--grid-size-range', type=int, nargs=2, default=None,
                   metavar=('LOW', 'HIGH'),
                   help='sample a square board size per episode')
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-episodes', type=int, default=6)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None,
                   help='checkpoint prefix; writes <out>_latest.pt, '
                        '<out>_best.pt and <out>_iter<N>.pt')
    p.add_argument('--checkpoint-every', type=int, default=5,
                   help='iterations between rolling checkpoints')
    p.add_argument('--keep-every', type=int, default=0,
                   help='also keep a permanent snapshot this often '
                        '(0 disables)')
    p.add_argument('--resume', default=None,
                   help='continue a run: weights, optimiser and iteration')
    p.add_argument('--warm-start', default=None,
                   help='start a new run from existing weights only, with a '
                        'fresh optimiser and iteration count')
    p.add_argument('--log', default=None, help='JSON lines metrics path')
    return p.parse_args()


def save_checkpoint(path, net, optimizer, args, iteration, best):
    """Everything needed to resume or to reload for evaluation."""
    torch.save({
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
        'best_eval': best,
        'objective': args.objective,
        'channels': args.channels,
        'blocks': args.blocks,
        'num_snakes': args.num_snakes,
    }, path)


def make_env(args, num_snakes):
    return gym.make('Snake-v1', height=args.height, width=args.width,
                    num_snakes=num_snakes, num_fruits=args.num_fruits,
                    reward_dict=REWARD_DICT,
                    obstacle_density=args.obstacle_density,
                    grid_size_range=(tuple(args.grid_size_range)
                                     if args.grid_size_range else None),
                    disable_env_checker=True)


def evaluate(args, evaluator, num_snakes, seed_offset=0):
    """Greedy play with no root noise; returns mean communal return/steps."""
    objective = get_objective(args.objective)
    solver = AlphaZeroSolver(
        evaluator, objective=args.objective,
        num_simulations=args.num_simulations,
        exploration_fraction=0.0, seed=args.seed)
    env = make_env(args, num_snakes)
    returns, lengths = [], []
    for ep in range(args.eval_episodes):
        env.reset(seed=10_000 + seed_offset + ep)
        solver.reset()
        total, steps = 0.0, 0
        for _ in range(args.max_steps):
            action = solver.solve(env)
            _, rews, terminated, truncated, _ = env.step(action)
            total += float(objective.fold(rews))
            steps += 1
            if all(terminated) or all(truncated):
                break
        returns.append(total)
        lengths.append(steps)
    return float(np.mean(returns)), float(np.mean(lengths))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    eval_counts = args.eval_num_snakes or [args.num_snakes]

    net = SnakeNet(channels=args.channels, blocks=args.blocks)
    evaluator = NetworkEvaluator(net, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    buffer = ReplayBuffer(seed=args.seed)
    solver = AlphaZeroSolver(evaluator, objective=args.objective,
                             num_simulations=args.num_simulations,
                             seed=args.seed)
    env = make_env(args, args.num_snakes)
    rng = np.random.default_rng(args.seed)

    params = sum(p.numel() for p in net.parameters())
    print(f'device={device}  params={params}  objective={args.objective}  '
          f'snakes={args.num_snakes}  sims={args.num_simulations}')

    start_iteration = 1
    best_eval = -float('inf')
    if args.warm_start:
        # weights only: the optimiser state and the best-so-far belong to the
        # previous task, and the replay buffer starts empty either way
        state = torch.load(args.warm_start, map_location=device,
                           weights_only=False)
        net.load_state_dict(state['model'])
        print(f'warm started from {args.warm_start} '
              f'(trained {state.get("iteration", "?")} iterations)')
    if args.resume:
        state = torch.load(args.resume, map_location=device,
                           weights_only=False)
        net.load_state_dict(state['model'])
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        start_iteration = state.get('iteration', 0) + 1
        best_eval = state.get('best_eval', -float('inf'))
        print(f'resumed from {args.resume} at iteration {start_iteration}')

    log = open(args.log, 'a' if args.resume else 'w') if args.log else None
    start = time.time()
    for iteration in range(start_iteration, args.iterations + 1):
        play_returns = []
        for ep in range(args.episodes_per_iter):
            env.reset(seed=int(rng.integers(1 << 30)))
            positions, stats = self_play_episode(
                env, solver, args.objective, max_steps=args.max_steps,
                rng=rng)
            for position in positions:
                buffer.add(*position)
            play_returns.append(stats['communal_return'])

        losses = []
        if len(buffer) >= args.batch_size:
            for _ in range(args.train_steps_per_iter):
                batch = buffer.sample(args.batch_size, device)
                losses.append(train_step(net, optimizer, batch,
                                         args.objective))

        record = {
            'iteration': iteration,
            'selfplay_return': float(np.mean(play_returns)),
            'buffer': len(buffer),
            'elapsed': round(time.time() - start, 1),
        }
        if losses:
            for key in ('loss', 'value_loss', 'policy_loss'):
                record[key] = float(np.mean([x[key] for x in losses]))

        scored = None
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            for count in eval_counts:
                ret, length = evaluate(args, evaluator, count,
                                       seed_offset=iteration)
                record[f'eval_return_n{count}'] = round(ret, 3)
                record[f'eval_steps_n{count}'] = round(length, 1)
            # the training configuration is what selects the best model
            scored = record[f'eval_return_n{args.num_snakes}']

        if args.out:
            if scored is not None and scored > best_eval:
                best_eval = scored
                save_checkpoint(f'{args.out}_best.pt', net, optimizer, args,
                                iteration, best_eval)
                record['saved_best'] = round(best_eval, 3)
            if (iteration % args.checkpoint_every == 0
                    or iteration == args.iterations):
                save_checkpoint(f'{args.out}_latest.pt', net, optimizer,
                                args, iteration, best_eval)
            if args.keep_every and iteration % args.keep_every == 0:
                save_checkpoint(f'{args.out}_iter{iteration}.pt', net,
                                optimizer, args, iteration, best_eval)

        print('  '.join(f'{k}={v}' for k, v in record.items()), flush=True)
        if log:
            log.write(json.dumps(record) + '\n')
            log.flush()

    if log:
        log.close()
    if args.out:
        print(f'checkpoints under {args.out}_*.pt  '
              f'(best eval_return_n{args.num_snakes}={best_eval:.3f})')


if __name__ == '__main__':
    main()
