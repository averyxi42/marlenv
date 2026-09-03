"""Egocentric RGB views: what one snake actually sees.

Each view is an odd-sized square centred on a snake's head and rotated into
that snake's heading, so "up" is always forward. Everything a world model
would be trained on is therefore expressed in the frame of the agent whose
actions it is conditioned on.

Regions past the edge of the board are filled with **free space**, not wall.
Padding with wall would invent a barrier that is not there and teach a model
that the world is smaller than it is; free space is the honest default for
"outside the observation". Because those cells still have to carry the same
persistent noise as real ones, the noise field is sized to the padded board
rather than the board itself.
"""
import numpy as np

from marlenv.core.snake import Cell, Direction

# 90-degree counter-clockwise rotations that bring each heading round to UP.
# np.rot90 maps a direction (dr, dc) to (-dc, dr), which gives this cycle.
HEADING_ROTATIONS = {
    Direction.UP: 0,
    Direction.RIGHT: 1,
    Direction.DOWN: 2,
    Direction.LEFT: 3,
}


def pad_grid(grid, pad, empty_value=Cell.EMPTY.value):
    """The grid surrounded by ``pad`` cells of free space."""
    if pad <= 0:
        return grid
    height, width = grid.shape
    padded = np.full((height + 2 * pad, width + 2 * pad), empty_value,
                     dtype=grid.dtype)
    padded[pad:pad + height, pad:pad + width] = grid
    return padded


def egocentric_crop(frame, head_coord, direction, radius, pad):
    """Crop ``frame`` around a head and rotate it into the head's frame.

    ``frame`` must already be padded by ``pad >= radius`` cells, and
    ``head_coord`` is given in unpadded grid coordinates.
    """
    row = int(head_coord[0]) + pad
    col = int(head_coord[1]) + pad
    view = frame[row - radius:row + radius + 1,
                 col - radius:col + radius + 1]

    rotations = HEADING_ROTATIONS[direction]
    if rotations:
        view = np.rot90(view, rotations, axes=(0, 1))
    return np.ascontiguousarray(view)


def heading_gradient(shape, pad=0, period=6, amplitude=28.0, angle=0.0):
    """A quadrature-encoded stripe field anchored to world coordinates.

    Egocentric views are rotated into the head's frame, which throws away
    the one thing the crop cannot otherwise express: which way the snake is
    actually facing. A world-anchored field gives it back, because rotating
    the view rotates the field with it.

    The field runs along a *single* direction, so it reads as one clean set
    of stripes with a definite angle. Two colour channels carry the same
    phase in quadrature -- red the cosine, green the sine -- which is what
    resolves the remaining 180-degree ambiguity: a lone sinusoid looks the
    same reversed, but the ordered pair traces a circle in colour space and
    reverses its direction of travel, exactly like a quadrature encoder.

    Both channels are offset to sit in ``[0, amplitude]`` rather than
    straddling zero. Against a black background the negative half of a
    zero-mean signal would clip away, destroying half the encoding.

    Periodic rather than monotone so it tiles any board size; position is
    recoverable only modulo ``period``.
    """
    rows = (np.arange(shape[0], dtype=np.float32) - pad)[:, None]
    cols = (np.arange(shape[1], dtype=np.float32) - pad)[None, :]
    radians = np.deg2rad(angle)
    # distance along the gradient direction; constant across the stripes
    projection = rows * np.cos(radians) + cols * np.sin(radians)
    phase = 2.0 * np.pi * projection / period

    field = np.zeros((*shape, 3), dtype=np.float32)
    field[..., 0] = 0.5 * amplitude * (1.0 + np.cos(phase))
    field[..., 1] = 0.5 * amplitude * (1.0 + np.sin(phase))
    return field


def composite_background(rgb, snakes, pad=0, background=None, noise=None):
    """Add background effects to every cell except the snakes.

    Snake cells take their base colour plus their own noise, so background
    texture and the heading gradient never bleed onto a body -- a snake
    looks the same wherever it stands.
    """
    out = rgb.astype(np.float32)
    if background is not None:
        out += background

    height, width = rgb.shape[:2]
    for snake in snakes:
        if not snake.alive:
            continue
        for distance, (r, c) in enumerate(snake.coords):
            r, c = r + pad, c + pad
            if 0 <= r < height and 0 <= c < width:
                offset = 0.0
                if noise is not None:
                    offset = noise.snake_offset(snake.idx, distance)
                out[r, c] = rgb[r, c] + offset
    return np.clip(out, 0, 255).astype(np.uint8)
