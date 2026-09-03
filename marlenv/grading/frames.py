"""Scoring a world model's next frame by what kind of cell it got wrong.

A single number over all pixels is close to useless here, because the cells
are not equally hard and not equally common. Most of a view is background,
wall and fruit, which sit still and are recoverable from position alone; a
model that paints those perfectly and smears every snake still scores well.
Splitting the score by cell type is what makes the difference visible, and
the centre cell is worth calling out on its own -- it is the head of the
agent doing the looking, the single most predictable cell in the image, and
the one a rollout reads to decide whether that agent is still alive.

Crops are taken full length with nothing padded and nobody dead, so the
history is unambiguously known and can be presented at noise level zero.
Scoring a crop with padding or a retired viewpoint in it measures the
scorer's own handling of those, not the model.
"""
import numpy as np
import torch

from marlenv.core.palette import decode_grid
from marlenv.core.snake import Cell
from marlenv.grading.compare import PALETTE_SNAKES
from marlenv.wm.data import to_pixels
from marlenv.wm.diffusion import alpha_sigma, from_velocity

SNAKE_KINDS = (Cell.HEAD.value, Cell.BODY.value, Cell.TAIL.value)


def denoise_last(step_fn, shape, device, denoise_steps=16, seed=0):
    """DDIM sample the final frame, given a model already conditioned."""
    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn(shape, device=device, generator=generator)
    levels = torch.linspace(1.0, 0.0, denoise_steps + 1, device=device)
    for index in range(denoise_steps):
        level = float(levels[index])
        predicted = step_fn(current, level)
        tau = torch.full_like(current[..., :1, :1, :1], level)
        clean, noise = from_velocity(current, predicted, tau)
        alpha, sigma = alpha_sigma(levels[index + 1])
        current = alpha * clean.clamp(-1, 1) + sigma * noise
    return current


def to_classes(frames):
    """Palette classes for a stack of frames, as ``(n, view, view)``."""
    pixels = to_pixels(frames.detach().cpu().numpy())
    pixels = pixels.reshape(-1, *pixels.shape[-3:])
    return np.stack([decode_grid(frame, PALETTE_SNAKES) for frame in pixels])


def report(truth, dream):
    """Accuracy overall and split by what the cell actually held."""
    middle = truth.shape[1] // 2
    kind = truth % 10
    snake = np.isin(kind, SNAKE_KINDS) & (truth >= Cell.HEAD.value)
    static = truth <= Cell.FRUIT.value
    centre_true = truth[:, middle, middle]
    centre_seen = dream[:, middle, middle]
    return {
        'viewpoints': int(len(truth)),
        'overall': float((truth == dream).mean()),
        'static': float((truth[static] == dream[static]).mean()),
        'snake': float((truth[snake] == dream[snake]).mean()),
        'snake_cells': int(snake.sum()),
        'centre_exact': float((centre_true == centre_seen).mean()),
        'centre_is_head': float(
            ((centre_seen % 10) == Cell.HEAD.value).mean()),
    }


def show(name, scores):
    print(f'{name}   ({scores["viewpoints"]} viewpoints, '
          f'{scores["snake_cells"]} snake cells)')
    for key, caption in (('centre_is_head', 'centre reads as a head'),
                         ('centre_exact', 'centre cell exact'),
                         ('snake', 'snake cells'),
                         ('static', 'empty / wall / fruit'),
                         ('overall', 'overall')):
        print(f'    {caption:<24s} {scores[key]:.3f}')


# ------------------------------------------------------------ multi agent
def multi_crops(sequences, context, stride=7, limit=48):
    """Full-length crops in which every agent is alive throughout."""
    lengths = sequences['mask'].sum(axis=1)
    crops = []
    for row in range(len(lengths)):
        for start in range(0, int(lengths[row]) - context + 1, stride):
            if sequences['alive'][row, start:start + context].all():
                crops.append((row, start))
            if len(crops) >= limit:
                return crops
    return crops


def grade_multi(model, sequences, context, device, denoise_steps=16,
                stride=7, limit=48, seed=0):
    from marlenv.wm.data import to_model_input
    from marlenv.wm.multiagent import actions_to_signal

    crops = multi_crops(sequences, context, stride, limit)
    if not crops:
        raise ValueError('no clean full-length crops')
    agents = sequences['observations'].shape[2]

    frames = torch.from_numpy(to_model_input(np.stack(
        [sequences['observations'][r, s:s + context] for r, s in crops]
    ))).to(device)
    actions = torch.from_numpy(np.stack(
        [sequences['actions'][r, s:s + context - 1] for r, s in crops]
    )).to(device)
    origins = torch.from_numpy(np.stack(
        [sequences['positions'][r, s] - sequences['positions'][r, s, 0]
         for r, s in crops])).to(device)
    alive = torch.ones(len(crops), context, agents, dtype=torch.bool,
                       device=device)
    signal = actions_to_signal(actions, model.action_out.out_features)
    frame_tau = torch.zeros(len(crops), context, agents, device=device)
    action_tau = torch.zeros(len(crops), context - 1, agents, device=device)

    def step_fn(current, level):
        frame_tau[:, -1] = level
        with torch.no_grad():
            predicted, _ = model(torch.cat([frames[:, :-1], current], 1),
                                 signal, frame_tau, action_tau,
                                 origins=origins, action_indices=actions,
                                 alive=alive)
        return predicted[:, -1:]

    dream = denoise_last(step_fn, frames[:, -1:].shape, device,
                         denoise_steps, seed)
    return report(to_classes(frames[:, -1]), to_classes(dream[:, 0]))


# ----------------------------------------------------------- single agent
def single_crops(sequences, context, stride=11, limit=144):
    """Full-length crops that stop before any aftermath frame."""
    lengths = sequences['mask'].sum(axis=1)
    died = sequences.get('died')
    crops = []
    for row in range(len(lengths)):
        usable = int(lengths[row])
        if died is not None and bool(died[row]):
            usable -= 1                       # the last frame is the death
        for start in range(0, usable - context + 1, stride):
            crops.append((row, start))
            if len(crops) >= limit:
                return crops
    return crops


def grade_single(model, sequences, context, device, denoise_steps=16,
                 stride=11, limit=144, seed=0):
    from marlenv.wm.data import to_model_input

    crops = single_crops(sequences, context, stride, limit)
    if not crops:
        raise ValueError('no clean full-length crops')

    frames = torch.from_numpy(to_model_input(np.stack(
        [sequences['observations'][r, s:s + context] for r, s in crops]
    ))).to(device)
    actions = torch.from_numpy(np.stack(
        [sequences['actions'][r, s:s + context - 1] for r, s in crops]
    )).to(device)
    tau = torch.zeros(len(crops), context, device=device)

    def step_fn(current, level):
        tau[:, -1] = level
        with torch.no_grad():
            return model(torch.cat([frames[:, :-1], current], 1), actions,
                         tau)[:, -1:]

    dream = denoise_last(step_fn, frames[:, -1:].shape, device,
                         denoise_steps, seed)
    return report(to_classes(frames[:, -1]), to_classes(dream[:, 0]))
