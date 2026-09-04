"""Checking that an egocentric reconstruction is not quietly omniscient.

The reconstruction is supposed to throw information away: an observer keeps
its own trajectory in full, and recovers another agent only while that
agent's head is inside its view. Two things would betray a leak.

**Segment lengths.** The observer's own run should be long -- as long as it
lived -- while everything recovered from it should be short, a handful of
steps while a neighbour passes through view. A histogram with long
transient segments means agents are being followed after they should have
been lost.

**Identities.** Nothing survives an absence, so a snake that leaves and
comes back is a new agent. Every identity must therefore cover exactly one
unbroken stretch of time. An identity with a gap in it is the observer
holding on to something it could not have known.

Both are checked against visibility recomputed from the recorded poses,
rather than against the reconstruction's own idea of what was visible.
"""
import argparse
import os

import numpy as np
from datasets import load_from_disk

from marlenv.data import decode_episode
from marlenv.flex_wm.egocentric import egocentric_pairs, patch_offsets

OBSERVER = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='marlenv/demodata')
    p.add_argument('--components', nargs='+', default=['expert_nogradient'])
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def segments(times):
    """Split a sorted time index into contiguous stretches."""
    if not len(times):
        return []
    breaks = np.flatnonzero(np.diff(times) != 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [len(times)]])
    return [(int(times[a]), int(times[b - 1]) + 1)
            for a, b in zip(starts, stops)]


def truly_visible(episode, ego, radius):
    """Runs of head-in-view, recomputed from the poses alone."""
    alive, poses = episode['alive_mask'], episode['poses']
    frames, agents = alive.shape
    runs = 0
    for other in range(agents):
        if other == ego:
            continue
        flags = [bool(alive[t, ego] and alive[t, other]
                      and (np.abs(poses[t, other, :2]
                                  - poses[t, ego, :2]) <= radius).all())
                 for t in range(frames)]
        stretches, start = [], None
        for index, flag in enumerate(flags + [False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                stretches.append(index - start)
                start = None
        runs += sum(1 for length in stretches if length >= 2)
    return runs


def histogram(lengths, width=48):
    """A text histogram over integer lengths."""
    if not lengths:
        return ['  (none)']
    counts = np.bincount(lengths)
    peak = counts.max()
    lines = []
    for value, count in enumerate(counts):
        if not count:
            continue
        bar = '#' * max(int(round(width * count / peak)), 1)
        lines.append(f'  {value:4d}  {count:7d}  {bar}')
    return lines


def main():
    args = parse_args()
    offsets = patch_offsets(9, 3)
    rng = np.random.default_rng(args.seed)

    own_lengths, other_lengths = [], []
    episodes = gaps = reused = short = 0
    found = expected = 0

    for name in args.components:
        dataset = load_from_disk(os.path.join(args.data_root, name))
        for row in dataset.select(range(min(args.episodes, len(dataset)))):
            source = decode_episode(row)
            alive = source['alive_mask']
            living = [a for a in range(alive.shape[1])
                      if alive[:, a].sum() >= 2]
            if not living:
                continue
            # chosen here rather than inside, so the cross-check below knows
            # whose view the reconstruction is supposed to be limited to
            ego = int(rng.choice(living))
            pairs = egocentric_pairs(source, offsets, ego=ego)
            if not len(pairs['time']):
                continue
            episodes += 1

            for identity in np.unique(pairs['agent']):
                rows = pairs['agent'] == identity
                times = np.sort(pairs['time'][rows])
                if len(times) != len(set(times.tolist())):
                    reused += 1
                runs = segments(times)
                if len(runs) > 1:
                    gaps += 1
                length = len(times)
                if identity == OBSERVER:
                    own_lengths.append(length)
                else:
                    other_lengths.append(length)
                    found += 1
                    if length < 2:
                        short += 1

            expected += truly_visible(source, ego, radius=4)

    print(f'{episodes} episodes')
    print(f'\nobserver segment lengths ({len(own_lengths)} segments, '
          f'mean {np.mean(own_lengths):.1f})')
    for line in histogram(own_lengths):
        print(line)
    print(f'\nrecovered agent segment lengths ({len(other_lengths)} '
          f'segments, mean {np.mean(other_lengths):.1f})')
    for line in histogram(other_lengths):
        print(line)

    print(f'\nidentities covering more than one stretch   {gaps}')
    print(f'identities with a repeated time              {reused}')
    print(f'recovered segments shorter than two steps    {short}')
    print(f'recovered segments   {found}   visible runs in the poses  '
          f'{expected}')


if __name__ == '__main__':
    main()
