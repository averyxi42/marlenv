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
