"""Train the multi-agent world action model.

    python examples/train_wm_multi.py --steps 12000

Frames and actions are diffused together, so the model is a policy and a
dynamics model at once. That is what makes a multi-agent rollout possible:
the other agents' actions can be sampled instead of supplied.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import load_from_disk

from marlenv.wm.madata import build_multi_sequences
from marlenv.wm.matrain import MultiBatcher, multi_training_loss
from marlenv.wm.multiagent import MultiAgentWorldModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='marlenv/demodata')
    p.add_argument('--components', nargs='+', default=['expert', 'explore'])
    p.add_argument('--episodes-per-component', type=int, default=None)
    p.add_argument('--val-fraction', type=float, default=0.05)
    p.add_argument('--context', type=int, default=24)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--action-weight', type=float, nargs='+', default=[1.0],
                   help='how hard the action term pulls on the shared '
                        'trunk; at 1.0 four action components weigh as '
                        'much as 243 frame pixels. One value, or one per '
                        'component')
    p.add_argument('--action-dropout', type=float, nargs='+', default=[0.0],
                   help='fraction of each component\'s action tokens left '
                        'out of the loss each step. One value, or one per '
                        'component')
    p.add_argument('--init', default=None,
                   help='warm start from a checkpoint, to carry on rather '
                        'than begin again. A deeper --depth grafts: each '
                        'trained block is followed by a silenced copy of '
                        'itself, so step zero computes what it did before')
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--checkpoint-every', type=int, default=2000)
    p.add_argument('--save-at-start', action='store_true',
                   help='write the checkpoint before training touches it. '
                        'With --init and a deeper --depth that is the graft '
                        'itself, which should behave exactly like what it '
                        'grew from -- worth being able to check')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='marlenv/demodata/wm_multi')
    return p.parse_args()


def split(sequences, fraction, seed):
    count = len(sequences['observations'])
    order = np.random.default_rng(seed).permutation(count)
    cut = max(1, int(round(count * fraction)))
    return [{k: v[i] for k, v in sequences.items()}
            for i in (order[cut:], order[:cut])]


@torch.no_grad()
def evaluate(model, batcher, batch_size, batches=6, seed=1234):
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=batcher.device).manual_seed(seed)
    totals = np.zeros(3)
    for _ in range(batches):
        parts = multi_training_loss(model, *batcher.batch(batch_size),
                                    generator=generator)
        totals += [float(p) for p in parts]
    model.train(was_training)
    return totals / batches


def spread(values, count, flag):
    """One value for every component, from one given or one each."""
    if len(values) == 1:
        return [values[0]] * count
    if len(values) != count:
        raise SystemExit(f'{flag} takes one value or one per component')
    return list(values)


def trunk_pull(model, batcher, args):
    """Gradient each loss term puts into the shared trunk.

    The two terms compete for one trunk, and the loss values alone do not
    say who is winning -- the action term can sit at its floor and still
    dominate. Logged so the balance is visible while it runs.
    """
    trunk = [p for name, p in model.named_parameters()
             if name.startswith('blocks') or name.startswith('to_tokens')]

    def norm(term):
        model.zero_grad(set_to_none=True)
        term.backward(retain_graph=True)
        total = sum((p.grad ** 2).sum() for p in trunk if p.grad is not None)
        return float(total.sqrt())

    batch = batcher.batch(args.batch_size)
    total, frame_loss, _ = multi_training_loss(model, *batch)
    pull = norm(frame_loss), norm(total - frame_loss)
    model.zero_grad(set_to_none=True)
    return pull


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    weights = spread(args.action_weight, len(args.components),
                     '--action-weight')
    dropouts = spread(args.action_dropout, len(args.components),
                      '--action-dropout')

    datasets = []
    for name in args.components:
        dataset = load_from_disk(os.path.join(args.data_root, name))
        if args.episodes_per_component:
            dataset = dataset.select(
                range(min(args.episodes_per_component, len(dataset))))
        datasets.append(dataset)
        print(f'  {name}: {len(dataset)} episodes  '
              f'action weight {weights[len(datasets) - 1]:g}  '
              f'dropout {dropouts[len(datasets) - 1]:g}')

    sequences = build_multi_sequences(datasets, action_weights=weights,
                                      action_dropouts=dropouts)
    train_set, val_set = split(sequences, args.val_fraction, args.seed)
    agents = sequences['observations'].shape[2]
    print(f'  {len(sequences["observations"])} episodes, {agents} agents, '
          f'{int(sequences["mask"].sum())} steps')

    batcher = MultiBatcher(train_set, args.context, seed=args.seed,
                           device=device)
    validation = MultiBatcher(val_set, args.context, seed=args.seed + 1,
                              device=device)

    model = MultiAgentWorldModel(
        num_agents=agents, view=sequences['observations'].shape[3],
        num_actions=4, frame='world', dim=args.dim, depth=args.depth,
        heads=args.heads).to(device)
    if args.init:
        from marlenv.wm.graft import graft_depth
        saved = torch.load(args.init, map_location='cpu',
                           weights_only=False)
        silenced = graft_depth(model, saved['model'])
        note = (f' as {saved.get("depth", "?")} blocks grown to '
                f'{args.depth}, {silenced} silenced duplicates'
                if silenced else '')
        print(f'  warm started from {args.init}{note}')
    params = sum(p.numel() for p in model.parameters())
    print(f'device={device}  params={params / 1e6:.2f}M  agents={agents}  '
          f'context={args.context}  '
          f'tokens/seq={args.context * model.tokens_per_step - agents}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01, betas=(0.9, 0.95))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / 200, 1.0)
        * 0.5 * (1 + np.cos(np.pi * min(step / max(args.steps, 1), 1.0))))

    os.makedirs(args.out, exist_ok=True)

    def snapshot(step, history):
        state = {'model': model.state_dict(), 'num_agents': agents,
                 'view': sequences['observations'].shape[3],
                 'dim': args.dim, 'depth': args.depth,
                 'heads': args.heads, 'context': args.context,
                 'frame': 'world', 'num_actions': 4,
                 'align_coords': True, 'history': history}
        # a stamped copy as well as the rolling one: intermediate
        # checkpoints are how an early rollout gets inspected, and
        # saving only model.pt threw every one of them away
        torch.save(state, os.path.join(args.out, 'model.pt'))
        torch.save(state, os.path.join(args.out, f'model_step{step}.pt'))

    if args.save_at_start:
        snapshot(0, [])
        print(f'  saved step 0 to {args.out}')

    history, window, start = [], [], time.time()
    for step in range(args.steps):
        loss, frame_loss, action_loss = multi_training_loss(
            model, *batcher.batch(args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        window.append([float(loss), float(frame_loss), float(action_loss)])

        if (step + 1) % args.log_every == 0:
            mean = np.mean(window, axis=0)
            val = evaluate(model, validation, args.batch_size)
            pull = trunk_pull(model, batcher, args)
            record = {'step': step + 1, 'loss': mean[0], 'frame': mean[1],
                      'action': mean[2], 'val_loss': val[0],
                      'val_frame': val[1], 'val_action': val[2],
                      'grad_frame': pull[0], 'grad_action': pull[1],
                      'elapsed': round(time.time() - start, 1)}
            history.append(record)
            window = []
            print(f'  step {record["step"]:6d}  loss {record["loss"]:.4f}  '
                  f'frame {record["frame"]:.4f}  action {record["action"]:.4f}'
                  f'   val {record["val_loss"]:.4f} '
                  f'(f {record["val_frame"]:.4f} a {record["val_action"]:.4f})'
                  f'   pull a/f {pull[1] / max(pull[0], 1e-9):.2f}x'
                  f'  {record["elapsed"]:.0f}s', flush=True)

        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            snapshot(step + 1, history)

    with open(os.path.join(args.out, 'history.json'), 'w') as handle:
        json.dump(history, handle, indent=2)
    print(f'saved to {args.out}  ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    main()
