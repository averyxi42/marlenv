"""Do two dreamt agents agree about the ground they both can see?

Every other measure here scores one agent's view against the truth. This
scores two agents against *each other*, on the cells their views overlap,
and it is the one question a single-agent model cannot be asked at all.

It matters because of what the architecture gives away for free. Positions
are axial RoPE over time, row and column, so two agents' tokens already
carry their true spatial offset: a model could place a snake consistently
in both views because the embedding says where it is, not because it
learned that two accounts of one board have to match. Comparing a model
trained on several agents against one trained on a single agent separates
those, and this is the number that separates them.

Actions are forced to the simulator's, so dead reckoning is exact and the
overlap is where the views really meet, not where a drifting model thinks
they do. Any disagreement is the model's.

The truth is measured the same way alongside, where it agrees with itself
perfectly by construction -- a control that catches a broken harness before
it is mistaken for a broken model.
"""
from itertools import combinations

import numpy as np
import torch

from marlenv.grading.compare import ConfusionMatrix, align_local_obs
from marlenv.grading.ratchet import REWARD_DICT, world_views  # noqa: F401
from marlenv.wm.canvas import make_pose
from marlenv.wm.data import to_model_input, to_pixels
from marlenv.wm.model import HEADINGS

NORTH = HEADINGS[0]      # dreamt views are already north-up, so this is a
                         # pose that leaves them alone


class Agreement:
    """Cross-agent confusion for a dream, and for the truth beside it."""

    def __init__(self):
        self.dream = ConfusionMatrix()
        self.truth = ConfusionMatrix()
        self.pairs = 0
        self.steps = 0

    def add(self, poses, dreamt, real, alive):
        """Fold in every pair of living agents at one step."""
        self.steps += 1
        for a, b in combinations([i for i, on in enumerate(alive) if on], 2):
            for matrix, views in ((self.dream, dreamt), (self.truth, real)):
                shared = align_local_obs(poses[a], views[a],
                                         poses[b], views[b])
                if len(shared):
                    matrix.update_from(shared)
            self.pairs += 1

    def snake_rate(self, matrix):
        """Agreement restricted to cells either agent drew as a snake.

        The headline number is mostly a measure of how much empty board
        there is: background and wall are five sixths of any overlap, and a
        model that draws less snake scores better on the total for that
        reason alone. This is the part worth comparing.
        """
        rows = [i for i, label in enumerate(matrix.labels)
                if label.startswith(('head', 'body', 'tail'))]
        block = matrix.matrix[rows]
        total = int(block.sum())
        if not total:
            return float('nan'), 0
        hit = sum(int(matrix.matrix[i, i]) for i in rows)
        return hit / total, total

    def rate(self, matrix):
        total = int(matrix.matrix.sum())
        if not total:
            return float('nan'), 0
        return float(np.trace(matrix.matrix)) / total, total

    def report(self, name, limit=5):
        agree, cells = self.rate(self.dream)
        control, _ = self.rate(self.truth)
        print(f'{name}   {self.pairs} agent pairs over {self.steps} steps, '
              f'{cells} overlapping cells')
        print(f'   agreement  dream {agree:.4f}   truth {control:.4f}')
        snake, drawn = self.snake_rate(self.dream)
        print(f'   snake cells only  {snake:.4f}  ({drawn} cells)')
        by_class = self.per_class(self.dream)
        for label, share, count in by_class:
            print(f'      {label:<12} {share:.4f}  ({count})')
        for expected, observed, count in self.dream.top_confusions(limit):
            print(f'   confused {expected} for {observed}: {count}')

    def per_class(self, matrix):
        """Agreement rate for each class that actually turns up."""
        out = []
        for index, label in enumerate(matrix.labels):
            total = int(matrix.matrix[index].sum())
            if not total:
                continue
            out.append((label, float(matrix.matrix[index, index]) / total,
                        total))
        return sorted(out, key=lambda row: -row[2])


def measure(make_runner, make_env, solver_for, steps=80, bootstrap=6,
            episodes=6, seed=1300, denoise_steps=12, action_steps=4,
            device='cuda'):
    """Roll a model forward and score its agents against each other."""
    tally = Agreement()
    for episode in range(episodes):
        env = make_env(seed + episode)
        base = env.unwrapped
        solver = solver_for(episode)
        heads = np.array([s.head_coord for s in base.snakes], dtype=np.int64)
        origins = torch.from_numpy(heads - heads[0])[None].to(device)
        runner = make_runner(origins)
        runner.reset(torch.from_numpy(
            to_model_input(world_views(env)[None, None])).to(device))
        generator = torch.Generator(device=device).manual_seed(episode)

        for _ in range(bootstrap):
            env.step(solver.solve(env))
            live = torch.tensor([s.alive for s in base.snakes], device=device)
            actions = torch.tensor(
                [HEADINGS.index(s.direction) for s in base.snakes],
                dtype=torch.long, device=device)
            runner.observe(actions, torch.from_numpy(
                to_model_input(world_views(env)[None, None])).to(device),
                live)

        for _ in range(steps):
            alive = [s.alive for s in base.snakes]
            if sum(alive) < 2:
                break                  # nothing left to disagree about
            env.step(solver.solve(env))
            forced = {index: HEADINGS.index(snake.direction)
                      for index, snake in enumerate(base.snakes)}
            runner.step(fixed=forced, denoise_steps=denoise_steps,
                        action_steps=action_steps, generator=generator)

            alive = [s.alive for s in base.snakes]
            poses = [make_pose(int(s.head_coord[0]), int(s.head_coord[1]),
                               NORTH) for s in base.snakes]
            dreamt = [to_pixels(runner.frames[0, -1, i].cpu().numpy())
                      for i in range(len(base.snakes))]
            tally.add(poses, dreamt, world_views(env), alive)
    return tally
