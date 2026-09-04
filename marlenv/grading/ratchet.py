"""Does a dreamt snake hold its length, or only ever lose it?

Frame-level accuracy hides this completely. A model can be right about
almost every pixel -- background and wall are most of them -- while its
snakes quietly shed a cell now and then, and in a rollout that is fatal:
the shortened snake is fed back as history, nothing ever puts a cell back,
and the length ratchets down until there is no snake left.

So this measures the two rates separately, one step at a time against the
simulator:

    lost    cells that hold a snake and were not drawn as one
    gained  cells drawn as a snake that hold none

A model whose errors are symmetric wanders around the right length. A model
that only ever loses collapses, however small its per-step error. That
asymmetry, not the total, is what decides whether a rollout survives.

Actions are forced to the simulator's, so the dream and the truth stay on
one trajectory and every step is comparable. The policy driving them must
be the one the data was collected with: a weaker one never grows a snake,
and then nothing being measured here ever happens.
"""
import numpy as np
import torch

from marlenv.core.palette import decode_grid
from marlenv.core.snake import Cell
from marlenv.grading.compare import PALETTE_SNAKES, unrotate_view
from marlenv.wm.data import to_model_input, to_pixels
from marlenv.wm.model import HEADINGS

REWARD_DICT = {'fruit': 1.0, 'kill': 0.0, 'lose': -5.0, 'win': 0.0,
               'time': 0.01}


def world_views(env):
    """Every agent's view, north-up."""
    base = env.unwrapped
    return np.stack([unrotate_view(view, snake.direction) for view, snake
                     in zip(base.egocentric_rgb(), base.snakes)])


def own_length(grid):
    """Cells of the snake whose head is at the centre of this view.

    grid ``(view, view)`` of palette classes

    The centre is the viewer's own head by construction, so its colour says
    which snake is the viewer's. Only what is inside the view is counted --
    a long snake may reach past the edge -- so this is a floor on the true
    length, not the length itself.
    """
    middle = grid.shape[0] // 2
    who = grid[middle, middle] // 10
    kind = grid % 10
    return int(((grid // 10 == who) & (kind >= Cell.HEAD.value)
                & (kind <= Cell.TAIL.value)).sum())


def snake_cells(grid):
    """Mask of every cell holding any snake."""
    kind = grid % 10
    return (kind >= Cell.HEAD.value) & (kind <= Cell.TAIL.value)


class Tally:
    """Per-step sums over episodes, kept so buckets can be read off."""

    def __init__(self, steps):
        self.steps = steps
        self.dreamt = np.zeros(steps)
        self.true = np.zeros(steps)
        self.lost = np.zeros(steps)
        self.gained = np.zeros(steps)
        self.cells = np.zeros(steps)
        self.count = np.zeros(steps)
        self.peak = []

    def add(self, step, truth, dream):
        real, drew = snake_cells(truth), snake_cells(dream)
        self.lost[step] += int((real & ~drew).sum())
        self.gained[step] += int((~real & drew).sum())
        self.cells[step] += int(real.sum())
        self.dreamt[step] += own_length(dream)
        self.true[step] += own_length(truth)
        self.count[step] += 1

    def report(self, name, width=15):
        print(f'{name}   longest dreamt per episode {self.peak}   '
              f'mean {np.mean(self.peak) if self.peak else 0:.1f}')
        for low in range(0, self.steps, width):
            high = min(low + width, self.steps)
            live = self.count[low:high] > 0
            if not live.any():
                continue
            seen = self.count[low:high][live]
            total = max(self.cells[low:high][live].sum(), 1)
            print(f'   steps {low:2d}-{high - 1:2d}   '
                  f'dreamt {(self.dreamt[low:high][live] / seen).mean():5.2f}'
                  f'   true {(self.true[low:high][live] / seen).mean():5.2f}'
                  f'   lost {self.lost[low:high][live].sum() / total:.3f}'
                  f'  gained {self.gained[low:high][live].sum() / total:.3f}'
                  f'   ({int(seen.mean())} eps)')


def measure(make_runner, make_env, solver_for, steps=80, bootstrap=6,
            episodes=6, seed=1300, denoise_steps=12, action_steps=4,
            agent=0, device='cuda'):
    """Roll a model forward under the simulator's own actions.

    make_runner  ``origins -> runner``; the runner keeps the dream
    make_env     ``seed -> env``
    solver_for   ``episode -> solver`` driving every snake
    agent        whose view is scored

    Returns a filled :class:`Tally`.
    """
    tally = Tally(steps)
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

        died = False
        for _ in range(bootstrap):
            env.step(solver.solve(env))
            if not base.snakes[agent].alive:
                died = True
                break
            live = torch.tensor([s.alive for s in base.snakes],
                                device=device)
            actions = torch.tensor(
                [HEADINGS.index(s.direction) for s in base.snakes],
                dtype=torch.long, device=device)
            runner.observe(actions, torch.from_numpy(
                to_model_input(world_views(env)[None, None])).to(device),
                live)
        if died:
            continue

        best = 0
        for step in range(steps):
            if not base.snakes[agent].alive:
                break
            env.step(solver.solve(env))
            forced = {index: HEADINGS.index(snake.direction)
                      for index, snake in enumerate(base.snakes)}
            runner.step(fixed=forced, denoise_steps=denoise_steps,
                        action_steps=action_steps, generator=generator)
            truth = decode_grid(world_views(env)[agent], PALETTE_SNAKES)
            dream = decode_grid(to_pixels(
                runner.frames[0, -1, agent].cpu().numpy()), PALETTE_SNAKES)
            tally.add(step, truth, dream)
            best = max(best, own_length(dream))
        tally.peak.append(best)
    return tally
