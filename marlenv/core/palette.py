"""Cell palette, chosen so a rendered frame decodes back to the grid.

Three requirements pull against each other, and the palette is the
compromise:

*learnability*
    a pixel's class must be unambiguous despite the observation noise and
    the heading gradient, so a model never has to guess what it is looking
    at;
*gradeability*
    the grid must be recoverable from an image, so a learned world model's
    output can be scored against the true state rather than eyeballed;
*appeal*
    each snake reads as one hue family with head, body and tail as
    decreasing brightness, and the board stays legible to a person.

The first two are the same property stated twice: every class colour must
sit far enough from every other that no perturbation can carry one into
another's territory. Concretely, for a nearest-centroid decoder to be right
with probability ~1 the distance between two class colours must exceed
``5 * (sigma_a + sigma_b)``, five standard deviations of the noise on each
side. Empty cells additionally carry the heading gradient, which displaces
them, so their pairs need that displacement allowed for as well.

:func:`safety_report` computes all of this for a given configuration, and
the tests use it to assert the shipped defaults keep a healthy margin.

The bound is deliberately conservative for snakes. It assumes every pixel
draws its own noise, but the snake field holds only ``num_snakes * period``
draws per episode, shared across every pixel of a segment class. Errors are
therefore rarer than the bound suggests and correlated when they happen --
a whole class flips, not scattered pixels. Measured over 400 episodes at
sigma_bg 2 and gradient 16:

======  ===============  ==================
sigma   part confusions  identity/occupancy
======  ===============  ==================
6       0                0
8       0                0
10      1                0
12      11               1
16      74               11
20      232              53
======  ===============  ==================

Two things follow. Decoding is clean well past the analytic budget, and it
degrades into the *harmless* failure first: confusing head for body within
one snake, which an evaluator that supplies or samples the actions can
recover anyway. Identity and occupancy errors only appear once the noise is
roughly double what the analysis permits.
"""
import itertools

import numpy as np

from marlenv.core.snake import Cell

# Six evenly spaced, fully saturated hues. Six rather than four because it
# is the largest wheel that still clears the margin; past that the families
# crowd each other and fruit.
HEAD_WHEEL = [
    (254, 130, 3), (127, 254, 3), (3, 254, 130),
    (3, 127, 254), (130, 3, 254), (254, 3, 127),
]
BODY_WHEEL = [
    (190, 98, 2), (94, 190, 2), (2, 190, 98),
    (2, 94, 190), (98, 2, 190), (190, 2, 94),
]
TAIL_WHEEL = [
    (126, 66, 2), (61, 126, 2), (2, 126, 66),
    (2, 61, 126), (66, 2, 126), (126, 2, 61),
]

EMPTY_RGB = (17, 16, 20)
# a dark blue slate: the wall was never the binding pair, so it can
# recede instead of framing the board in bright grey
WALL_RGB = (78, 86, 108)
FRUIT_RGB = (243, 32, 36)

CELL_COLORS = {
    Cell.EMPTY.value: [EMPTY_RGB],
    Cell.WALL.value: [WALL_RGB],
    Cell.FRUIT.value: [FRUIT_RGB],
    Cell.HEAD.value: HEAD_WHEEL,
    Cell.BODY.value: BODY_WHEEL,
    Cell.TAIL.value: TAIL_WHEEL,
}

# only empty cells carry the heading gradient, so only they are displaced
SNAKE_KINDS = (Cell.HEAD.value, Cell.BODY.value, Cell.TAIL.value)


def cell_color(kind, snake_id=0):
    """The colour of one cell class, matching ``rgb_from_grid``."""
    wheel = CELL_COLORS[kind]
    base = np.array(wheel[snake_id % len(wheel)], dtype=np.float64)
    return base * 0.7 ** (snake_id // len(wheel))


def palette_entries(num_snakes):
    """Every class in play: ``(values, colors)`` of shape ``(K,)``/``(K, 3)``.

    ``values`` are raw grid values, ``kind + 10 * snake_id``, so decoding
    returns a grid directly comparable with ``env.grid``.
    """
    values, colors = [], []
    for kind in (Cell.EMPTY.value, Cell.WALL.value, Cell.FRUIT.value):
        values.append(kind)
        colors.append(cell_color(kind))
    for snake_id in range(num_snakes):
        for kind in SNAKE_KINDS:
            values.append(kind + 10 * snake_id)
            colors.append(cell_color(kind, snake_id))
    return np.array(values), np.array(colors)


def nearest_class(frame, num_snakes):
    """Index of the closest palette colour for every pixel."""
    _, colors = palette_entries(num_snakes)
    diff = frame.astype(np.float64)[:, :, None, :] - colors[None, None]
    return np.argmin((diff ** 2).sum(axis=-1), axis=-1)


def decode_grid(frame, num_snakes):
    """Recover the grid from a rendered frame by nearest class colour.

    This is the grading path: run it on a world model's generated frame and
    compare with the true grid.
    """
    values, _ = palette_entries(num_snakes)
    return values[nearest_class(frame, num_snakes)]


def snap_to_palette(frame, num_snakes):
    """Quantise a frame onto the palette, removing noise and gradient.

    Two frames that mean the same thing snap to identical pixels, so a
    prediction and the truth can be compared exactly rather than by a
    tolerance that would have to be tuned against the noise level.
    """
    _, colors = palette_entries(num_snakes)
    return colors[nearest_class(frame, num_snakes)].astype(np.uint8)


def class_labels(num_snakes):
    """Readable name per palette class, ordered as :func:`palette_entries`."""
    values, _ = palette_entries(num_snakes)
    labels = []
    for value in values:
        kind, snake_id = int(value) % 10, int(value) // 10
        name = Cell(kind).name.lower()
        labels.append(name if kind not in SNAKE_KINDS else f'{name}{snake_id}')
    return labels


def class_index(values, num_snakes):
    """Map raw grid values onto palette class indices."""
    table, _ = palette_entries(num_snakes)
    lookup = {int(v): i for i, v in enumerate(table)}
    flat = np.asarray(values).reshape(-1)
    return np.array([lookup[int(v)] for v in flat]).reshape(
        np.asarray(values).shape)


def gradient_shift(amplitude):
    """Worst-case displacement an empty cell suffers from the gradient."""
    return float(np.hypot(amplitude, amplitude))


def safety_report(num_snakes, sigma_bg, sigma_snake,
                  gradient_amplitude=28.0, sigmas=5.0, strict=True):
    """Worst-case decoding margin for a configuration.

    Returns ``(slack, description)``; ``slack`` is how much distance is left
    over on the tightest pair after allowing for the gradient displacement
    and ``sigmas`` standard deviations of noise on each side. Positive means
    every class is separable.

    ``strict=True`` requires every class to be distinguishable, head from
    body from tail included. ``strict=False`` drops pairs that differ only
    in which part of the *same* snake they are: an evaluator that supplies
    or samples the actions can infer the body ordering anyway, so confusing
    a body cell for a tail cell of the same snake costs it nothing. The
    relaxed budget is the larger of the two, and the difference is the
    headroom that assumption buys.
    """
    values, colors = palette_entries(num_snakes)
    names = []
    for value in values:
        kind, snake_id = value % 10, value // 10
        base = Cell(kind).name
        names.append(base if kind not in SNAKE_KINDS else f'{base}{snake_id}')

    worst = (float('inf'), '')
    for i, j in itertools.combinations(range(len(values)), 2):
        same_snake = (values[i] // 10 == values[j] // 10
                      and values[i] % 10 in SNAKE_KINDS
                      and values[j] % 10 in SNAKE_KINDS)
        if not strict and same_snake:
            continue
        distance = float(np.linalg.norm(colors[i] - colors[j]))
        needed = 0.0
        for index in (i, j):
            kind = values[index] % 10
            if kind in SNAKE_KINDS:
                needed += sigmas * sigma_snake
            else:
                needed += sigmas * sigma_bg
                if kind == Cell.EMPTY.value:
                    needed += gradient_shift(gradient_amplitude)
        slack = distance - needed
        if slack < worst[0]:
            worst = (slack, f'{names[i]} vs {names[j]} '
                            f'(distance {distance:.1f}, needs {needed:.1f})')
    return worst
