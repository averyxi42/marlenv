"""Train the world action model over sets of observation/action pairs.

    python examples/train/train_flex_wam.py --schedule FAGFAGAAGAAG

Same weights and same rule as train_wm_multi, with two differences that are
the point of it.

**Attention scope varies by block.** ``--schedule`` is one letter per block,
repeated to fill the depth: F stays inside one observation, A inside one
agent's own history, G reaches everywhere causality allows. Identity comes
from comparing agent ids in the mask rather than from an embedding, so the
model stays permutation equivariant and an episode may contain far more
identities than are ever live at once. ``G`` alone reproduces the older
model exactly.

**The window is separate from the crop.** ``--context`` is how many frames
a crop holds; ``--window`` is how far back a token may look inside it. They
used to be the same thing by accident, which meant a token late in a crop
trained on more history than a sliding cache will ever hand it. Setting a
crop wider than the window fixes that for any token deep enough in the
crop, and ``--warmup-frames`` drops the shallow ones from the loss so only
the matched region trains.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import load_from_disk

from marlenv.flex_wm.model import FlexWorldModel
from marlenv.flex_wm.train import PairBatcher, flex_training_loss
from marlenv.wm.madata import build_multi_sequences


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='marlenv/demodata')
    p.add_argument('--components', nargs='+', default=['expert', 'explore'])
    p.add_argument('--episodes-per-component', type=int, default=None)
    p.add_argument('--val-fraction', type=float, default=0.05)
    p.add_argument('--schedule', default='FAGFAGAAGAAG',
                   help='attention scope per block, repeated to fill the '
                        'depth; G alone is the older uniform behaviour')
    p.add_argument('--context', type=int, default=None,
                   help='frames per crop; with --init, the checkpoint\'s')
    p.add_argument('--window', type=int, default=None,
                   help='frames of history a token may look back over. '
                        'Defaults to the crop, where it changes nothing; '
                        'set it smaller than --context to train the same '
                        'computation a sliding cache plays')
    p.add_argument('--warmup-frames', type=int, default=0,
                   help='frames at the start of a crop to leave out of the '
                        'loss, being the ones whose history is not yet as '
                        'deep as it would be mid-rollout')
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--depth', type=int, default=None)
    p.add_argument('--heads', type=int, default=None)
    p.add_argument('--action-weight', type=float, nargs='+', default=[1.0],
                   help='one value, or one per component')
    p.add_argument('--action-dropout', type=float, nargs='+', default=[0.0])
    p.add_argument('--keep-retired', dest='drop_retired',
                   action='store_false', default=True,
                   help='pin retired pairs at maximum noise instead of '
                        'removing them from the set')
    p.add_argument('--init', default=None,
                   help='warm start; a deeper --depth grafts')
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--checkpoint-every', type=int, default=2000)
    p.add_argument('--save-at-start', action='store_true')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='marlenv/demodata/flex_wam')
    return p.parse_args()


def spread(values, count, flag):
    """One value per component, from one given or one each."""
    if len(values) == 1:
        return [values[0]] * count
    if len(values) != count:
        raise SystemExit(f'{flag} takes one value or one per component')
    return list(values)


def split(sequences, fraction, seed):
    count = len(sequences['observations'])
    order = np.random.default_rng(seed).permutation(count)
    cut = max(1, int(round(count * fraction)))
    return [{k: v[i] for k, v in sequences.items()}
            for i in (order[cut:], order[:cut])]


def drop_warmup(pairs, frames):
    """Leave the shallow end of a crop out of the loss.

    A token near the start of a crop has less history than the same token
    would have mid-rollout, however the window is set, simply because there
    is nothing to its left. Excluding those frames from the loss leaves only
    the region where training and play agree.
    """
    if frames <= 0:
        return pairs
    deep = pairs.time >= frames
    pairs.trained = pairs.trained & deep
    pairs.acted = pairs.acted & deep
    return pairs


@torch.no_grad()
def evaluate(model, batcher, args, batches=6, seed=1234):
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=batcher.device).manual_seed(seed)
    totals = np.zeros(3)
    for _ in range(batches):
        pairs, weight, dropout = batcher.pairs(args.batch_size, model,
                                               args.drop_retired)
        parts = flex_training_loss(model, drop_warmup(pairs,
                                                      args.warmup_frames),
                                   weight, dropout, window=args.window,
                                   generator=generator)
        totals += [float(p) for p in parts]
    model.train(was_training)
    return totals / batches


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    saved = (torch.load(args.init, map_location='cpu', weights_only=False)
             if args.init else None)
    fallback = {'context': 48, 'dim': 256, 'depth': 12, 'heads': 8}
    for name, default in fallback.items():
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name, default) if saved
                    else default)
    if args.window is None:
        args.window = args.context

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
              f'dropout {dropouts[len(datasets) - 1]:g}', flush=True)

    sequences = build_multi_sequences(datasets, action_weights=weights,
                                      action_dropouts=dropouts)
    train_set, val_set = split(sequences, args.val_fraction, args.seed)
    print(f'  {len(sequences["observations"])} episodes, '
          f'{int(sequences["mask"].sum())} steps')

    batcher = PairBatcher(train_set, args.context, seed=args.seed,
                          device=device)
    validation = PairBatcher(val_set, args.context, seed=args.seed + 1,
                             device=device)

    model = FlexWorldModel(
        schedule=args.schedule, view=sequences['observations'].shape[3],
        num_actions=4, frame='world', dim=args.dim, depth=args.depth,
        heads=args.heads).to(device)
    if saved is not None:
        from marlenv.wm.graft import graft_depth
        silenced = graft_depth(model, saved['model'])
        note = (f' as {saved.get("depth", "?")} blocks grown to '
                f'{args.depth}, {silenced} silenced duplicates'
                if silenced else '')
        print(f'  warm started from {args.init}{note}',
              flush=True)

    params = sum(p.numel() for p in model.parameters())
    print(f'device={device}  params={params / 1e6:.2f}M  '
          f'schedule={"".join(model.schedule)}  crop={args.context}  '
          f'window={args.window}  warmup={args.warmup_frames}  '
          f'retired={"dropped" if args.drop_retired else "pinned"}',
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01, betas=(0.9, 0.95))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / 200, 1.0)
        * 0.5 * (1 + np.cos(np.pi * min(step / max(args.steps, 1), 1.0))))

    os.makedirs(args.out, exist_ok=True)

    def snapshot(step, history):
        torch.save({'model': model.state_dict(),
                    'view': sequences['observations'].shape[3],
                    'dim': args.dim, 'depth': args.depth,
                    'heads': args.heads, 'context': args.context,
                    'window': args.window, 'schedule': args.schedule,
                    'frame': 'world', 'num_actions': 4,
                    'align_coords': True, 'history': history},
                   os.path.join(args.out, 'model.pt'))
        torch.save(torch.load(os.path.join(args.out, 'model.pt'),
                              map_location='cpu', weights_only=False),
                   os.path.join(args.out, f'model_step{step}.pt'))

    if args.save_at_start:
        snapshot(0, [])
        print(f'  saved step 0 to {args.out}', flush=True)

    history, window, start = [], [], time.time()
    for step in range(args.steps):
        pairs, weight, dropout = batcher.pairs(args.batch_size, model,
                                               args.drop_retired)
        loss, frame_loss, action_loss = flex_training_loss(
            model, drop_warmup(pairs, args.warmup_frames), weight, dropout,
            window=args.window)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        window.append([float(loss.detach()), float(frame_loss.detach()),
                       float(action_loss.detach())])

        if (step + 1) % args.log_every == 0:
            mean = np.mean(window, axis=0)
            val = evaluate(model, validation, args)
            record = {'step': step + 1, 'loss': mean[0], 'frame': mean[1],
                      'action': mean[2], 'val_loss': val[0],
                      'val_frame': val[1], 'val_action': val[2],
                      'elapsed': round(time.time() - start, 1)}
            history.append(record)
            window = []
            print(f'  step {record["step"]:6d}  loss {record["loss"]:.4f}  '
                  f'frame {record["frame"]:.4f}  '
                  f'action {record["action"]:.4f}'
                  f'   val {record["val_loss"]:.4f} '
                  f'(f {record["val_frame"]:.4f} a {record["val_action"]:.4f})'
                  f'  {record["elapsed"]:.0f}s', flush=True)

        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            snapshot(step + 1, history)

    with open(os.path.join(args.out, 'history.json'), 'w') as handle:
        json.dump(history, handle, indent=2)
    print(f'saved to {args.out}  ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    main()
