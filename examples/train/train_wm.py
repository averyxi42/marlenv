"""Train the single-agent world model on collected episodes.

    python examples/train/train_wm.py --steps 8000 --components expert explore
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import load_from_disk

from marlenv.wm import SequenceBatcher, WorldModel, build_sequences, train
from marlenv.wm.diagnostics import format_report, noise_level_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='marlenv/demodata')
    p.add_argument('--components', nargs='+', default=['expert', 'explore'])
    p.add_argument('--episodes-per-component', type=int, default=None)
    p.add_argument('--val-fraction', type=float, default=0.05)
    p.add_argument('--frame', choices=['ego', 'world'], default='ego',
                   help="'world' un-rotates views to north-up and uses the "
                        'four cardinal actions, so consecutive frames differ '
                        'by a pure translation')
    p.add_argument('--align-coords', dest='align_coords',
                   action='store_true', default=True,
                   help='share spatial RoPE coordinates across the sequence '
                        '(default)')
    p.add_argument('--no-align-coords', dest='align_coords',
                   action='store_false')
    p.add_argument('--context', type=int, default=24)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--checkpoint-every', type=int, default=1000,
                   help='iterations between rolling checkpoints')
    p.add_argument('--report-every', type=int, default=1000,
                   help='iterations between per-tau reconstruction reports')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='marlenv/demodata/wm')
    return p.parse_args()


def load_components(args):
    datasets = []
    for name in args.components:
        dataset = load_from_disk(os.path.join(args.data_root, name))
        if args.episodes_per_component:
            keep = min(args.episodes_per_component, len(dataset))
            dataset = dataset.select(range(keep))
        datasets.append(dataset)
        print(f'  {name}: {len(dataset)} episodes')
    return datasets


def split(sequences, fraction, seed):
    """Hold out whole sequences, so no frame is in both halves."""
    count = len(sequences['observations'])
    order = np.random.default_rng(seed).permutation(count)
    cut = max(1, int(round(count * fraction)))
    parts = []
    for indices in (order[cut:], order[:cut]):
        parts.append({key: value[indices]
                      for key, value in sequences.items()})
    return parts


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print('loading components:')
    sequences = build_sequences(load_components(args),
                                frame=args.frame)
    train_set, val_set = split(sequences, args.val_fraction, args.seed)
    print(f'  {len(sequences["observations"])} agent sequences '
          f'({len(train_set["observations"])} train / '
          f'{len(val_set["observations"])} val)')
    print(f'  {int(sequences["mask"].sum())} agent frames, '
          f'{int(sequences["died"].sum())} ending in death')

    batcher = SequenceBatcher(train_set, args.context, seed=args.seed,
                              device=device)
    validation = SequenceBatcher(val_set, args.context, seed=args.seed + 1,
                                 device=device)

    view = sequences['observations'].shape[2]
    num_actions = 3 if args.frame == 'ego' else 4
    model = WorldModel(view=view, dim=args.dim, depth=args.depth,
                       heads=args.heads, num_actions=num_actions,
                       frame=args.frame, align_coords=args.align_coords)
    params = sum(p.numel() for p in model.parameters())
    print(f'frame={args.frame}  actions={num_actions}  '
          f'aligned_coords={args.align_coords}')
    print(f'device={device}  params={params / 1e6:.2f}M  '
          f'context={args.context}  tokens/seq='
          f'{args.context * (model.tokens_per_frame + 1) - 1}')

    os.makedirs(args.out, exist_ok=True)

    def save(path, history):
        torch.save({'model': model.state_dict(), 'view': view,
                    'dim': args.dim, 'depth': args.depth, 'heads': args.heads,
                    'context': args.context, 'frame': args.frame,
                    'num_actions': num_actions,
                    'align_coords': args.align_coords,
                    'history': history}, path)

    # tau values worth watching: the low end is nearly free, and the high end
    # is what a rollout actually starts from
    watched = (0.2, 0.6, 0.9, 1.0)
    records = []

    def log(record):
        records.append(record)
        step = record['step']
        line = (f'  step {step:6d}  loss {record["loss"]:.4f}  '
                f'val {record.get("val_loss", float("nan")):.4f}  '
                f'lr {record["lr"]:.2e}  {record["elapsed"]:.0f}s')
        if step % args.report_every == 0 or step == args.steps:
            report = noise_level_report(model, validation, levels=watched,
                                        batches=2, batch_size=16)
            record['noise_levels'] = report
            recon = '  '.join(
                f'{row["tau"]:.1f}:{row["reconstruction_mse"]:.4f}'
                for row in report)
            line += f'\n         recon by tau  {recon}'
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(os.path.join(args.out, 'model.pt'), records)
            save(os.path.join(args.out, f'model_step{step}.pt'), records)
            line += f'   [saved step {step}]'
        print(line, flush=True)

    start = time.time()
    history = train(model, batcher, steps=args.steps,
                    batch_size=args.batch_size, lr=args.lr,
                    log_every=args.log_every, device=device,
                    validation=validation, on_log=log)

    print('\nfinal reconstruction by noise level '
          '(what a rollout depends on):')
    print(format_report(noise_level_report(model, validation)))

    save(os.path.join(args.out, 'model.pt'), history)
    with open(os.path.join(args.out, 'history.json'), 'w') as handle:
        json.dump(history, handle, indent=2)
    print(f'saved to {args.out}  ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    main()
