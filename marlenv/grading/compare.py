"""Aligning observations and counting where they disagree.

Everything here decodes against the **full** palette, all
:data:`PALETTE_SNAKES` snakes' worth of classes, not just the snakes an
episode happens to contain. A world model that invents a colour belonging
to a snake that is not on the board should be caught doing it, not quietly
snapped onto the nearest snake that does exist.
"""
from typing import NamedTuple

import numpy as np

from marlenv.core.observation import HEADING_ROTATIONS, pad_grid
from marlenv.core.palette import (class_index, class_labels, decode_grid,
                                  palette_entries)
from marlenv.core.snake import Cell

#: the whole colour wheel, so the class space is the same for every episode
PALETTE_SNAKES = 6
NUM_CLASSES = len(palette_entries(PALETTE_SNAKES)[0])


class Disagreement(NamedTuple):
    """Where two observations differ, and what each of them claims."""

    coords: np.ndarray      # (n, 2) world row/col
    expected: np.ndarray    # (n,) grid values from the reference
    observed: np.ndarray    # (n,) grid values from the candidate
    compared: int           # cells actually compared

    def __len__(self):
        return len(self.coords)

    @property
    def agreement(self):
        if self.compared == 0:
            return float('nan')
        return 1.0 - len(self.coords) / self.compared


def unrotate_view(view, direction):
    """Turn a head-frame view back into world orientation."""
    rotations = HEADING_ROTATIONS[direction]
    if rotations:
        view = np.rot90(view, -rotations, axes=(0, 1))
    return np.ascontiguousarray(view)


def view_radius(view):
    size = view.shape[0]
    if size % 2 == 0:
        raise ValueError(f'a head-centred view must be odd sized; got {size}')
    return size // 2


def view_grid(local_obs, pose):
    """Decode a local observation into a world-oriented grid of classes."""
    grid = decode_grid(local_obs, PALETTE_SNAKES)
    return unrotate_view(grid, pose.direction)


def view_origin(pose, radius):
    """World coordinate of a view's top-left cell, once unrotated."""
    return pose.row - radius, pose.col - radius


class Alignment(NamedTuple):
    """Every world cell two observations both cover, and what each says."""

    coords: np.ndarray      # (n, 2) world row/col
    expected: np.ndarray    # (n,) grid values from the reference
    observed: np.ndarray    # (n,) grid values from the candidate

    def __len__(self):
        return len(self.coords)

    def disagreements(self):
        """Just the cells that differ."""
        differs = self.expected != self.observed
        return Disagreement(self.coords[differs], self.expected[differs],
                            self.observed[differs], len(self.coords))


def _window_coords(top, left, shape):
    rows, cols = np.meshgrid(np.arange(shape[0]) + top,
                             np.arange(shape[1]) + left, indexing='ij')
    return np.stack([rows.ravel(), cols.ravel()], axis=1)


def align_obs(pose, local_obs, global_obs):
    """Line one agent's view up against the whole board.

    Cells the view sees beyond the edge of the board are compared against
    free space, which is what the environment pads them with.
    """
    observed = view_grid(local_obs, pose)
    radius = view_radius(observed)
    top, left = view_origin(pose, radius)

    global_grid = decode_grid(global_obs, PALETTE_SNAKES)
    padded = pad_grid(global_grid, radius, empty_value=Cell.EMPTY.value)
    expected = padded[top + radius:top + radius + observed.shape[0],
                      left + radius:left + radius + observed.shape[1]]

    return Alignment(_window_coords(top, left, observed.shape),
                     expected.ravel(), observed.ravel())


def align_local_obs(pose_a, obs_a, pose_b, obs_b):
    """Line two head-frame views up on the world cells they share.

    Under partial observability two views rarely cover the same ground, so
    only their overlap is meaningful; the rest is not disagreement, it is
    simply unseen. The alignment is empty when they do not meet at all.
    """
    grid_a, grid_b = view_grid(obs_a, pose_a), view_grid(obs_b, pose_b)
    radius_a, radius_b = view_radius(grid_a), view_radius(grid_b)
    top_a, left_a = view_origin(pose_a, radius_a)
    top_b, left_b = view_origin(pose_b, radius_b)

    top, left = max(top_a, top_b), max(left_a, left_b)
    bottom = min(top_a + grid_a.shape[0], top_b + grid_b.shape[0])
    right = min(left_a + grid_a.shape[1], left_b + grid_b.shape[1])
    if bottom <= top or right <= left:
        empty = np.zeros((0, 2), dtype=int)
        return Alignment(empty, np.zeros(0, int), np.zeros(0, int))

    window_a = grid_a[top - top_a:bottom - top_a, left - left_a:right - left_a]
    window_b = grid_b[top - top_b:bottom - top_b, left - left_b:right - left_b]
    return Alignment(_window_coords(top, left, window_a.shape),
                     window_a.ravel(), window_b.ravel())


def diff_obs(pose, local_obs, global_obs):
    """Cells where an agent's view contradicts the board."""
    return align_obs(pose, local_obs, global_obs).disagreements()


def diff_local_obs(pose_a, obs_a, pose_b, obs_b):
    """Cells where two agents' views contradict, on their overlap."""
    return align_local_obs(pose_a, obs_a, pose_b, obs_b).disagreements()


class ConfusionMatrix:
    """Counts of (expected class, observed class) over the full palette.

    Always ``NUM_CLASSES`` square, so matrices from episodes with different
    snake counts can be summed. The diagonal is agreement.
    """

    def __init__(self):
        self.labels = class_labels(PALETTE_SNAKES)
        self.matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, expected, observed):
        rows = class_index(expected, PALETTE_SNAKES).ravel()
        cols = class_index(observed, PALETTE_SNAKES).ravel()
        np.add.at(self.matrix, (rows, cols), 1)
        return self

    def update_from(self, alignment):
        """Fold in an :class:`Alignment`, agreements included.

        Takes the alignment rather than the disagreements because the
        diagonal needs the matching cells too, and which class those were
        is not recoverable once they have been filtered out.
        """
        return self.update(alignment.expected, alignment.observed)

    @property
    def errors(self):
        return int(self.matrix.sum() - np.trace(self.matrix))

    def top_confusions(self, limit=10):
        """Most frequent off-diagonal pairs, largest first."""
        counts = self.matrix.copy()
        np.fill_diagonal(counts, 0)
        order = np.argsort(counts, axis=None)[::-1]
        out = []
        for flat in order[:limit]:
            row, col = divmod(int(flat), NUM_CLASSES)
            if counts[row, col] == 0:
                break
            out.append((self.labels[row], self.labels[col],
                        int(counts[row, col])))
        return out
