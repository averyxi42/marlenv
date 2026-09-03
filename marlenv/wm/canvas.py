"""Stitching egocentric views into one map.

Each observation is a window onto the board, so pasting successive windows
at the positions dead reckoning says they came from rebuilds a global view.
Later frames overwrite earlier ones where they overlap, so the canvas shows
the model's most recent opinion of every cell.

This is the sharpest read on whether a world model keeps its geometry
straight. A consistent model paints a coherent board; a drifting one leaves
walls that move between visits and corridors that do not line up.

The pose used here is bookkeeping for compositing only. It never reaches the
model, which sees nothing but head-frame pixels and relative actions.
"""
import numpy as np

from marlenv.grading.poses import Pose, step_pose


class CanvasIntegrator:
    """Accumulates world-up views onto a fixed canvas, fading with age.

    Every paste first attenuates the whole canvas by ``decay`` and then
    writes the new view at full brightness, so a cell's brightness measures
    how recently it was last seen and unvisited canvas stays black. Beyond
    looking right, that is what makes inconsistency legible: an area revisited
    after a long absence lights up again, and any disagreement with what is
    still faintly painted there shows up as a seam.

    ``decay=1.0`` disables fading and keeps the newest opinion of every cell
    at full brightness.
    """

    def __init__(self, height, width, radius, origin=None, margin=None,
                 decay=0.95):
        self.radius = radius
        self.margin = radius if margin is None else margin
        self.height = height + 2 * self.margin
        self.width = width + 2 * self.margin
        self.origin = origin or (self.margin, self.margin)
        self.decay = float(decay)

        # kept in float so repeated attenuation does not stall on rounding
        self.buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.seen = np.zeros((self.height, self.width), dtype=bool)
        self.visits = []

    @property
    def image(self):
        """The canvas as uint8 pixels."""
        return np.clip(self.buffer, 0, 255).astype(np.uint8)

    def to_canvas(self, row, col):
        return row + self.origin[0], col + self.origin[1]

    def fade(self):
        """Age the whole canvas by one step.

        Separate from pasting because several viewpoints can contribute to
        the same step -- one per agent -- and the canvas should age once per
        step rather than once per view.
        """
        if self.decay != 1.0:
            self.buffer *= self.decay

    def add(self, view, pose):
        """Fade the canvas, then paste a world-up view centred on ``pose``."""
        self.fade()
        return self.paste(view, pose)

    def paste(self, view, pose):
        """Write a world-up view centred on ``pose``, without ageing."""
        view = np.asarray(view, dtype=np.float32)
        row, col = self.to_canvas(pose.row, pose.col)
        top, left = row - self.radius, col - self.radius
        bottom, right = top + view.shape[0], left + view.shape[1]

        # clip against the canvas, keeping the two windows aligned
        src_top = max(0, -top)
        src_left = max(0, -left)
        src_bottom = view.shape[0] - max(0, bottom - self.height)
        src_right = view.shape[1] - max(0, right - self.width)
        if src_top >= src_bottom or src_left >= src_right:
            return False

        dst = (slice(top + src_top, top + src_bottom),
               slice(left + src_left, left + src_right))
        self.buffer[dst] = view[src_top:src_bottom, src_left:src_right]
        self.seen[dst] = True
        self.visits.append((pose.row, pose.col))
        return True

    def coverage(self):
        """Fraction of the canvas any view has reached."""
        return float(self.seen.mean())

    def head_marker(self, pose):
        """Canvas coordinate of a pose, for drawing a cursor."""
        return self.to_canvas(pose.row, pose.col)


def dead_reckon(initial_pose, actions):
    """Poses implied by a sequence of relative actions."""
    poses = [initial_pose]
    for action in actions:
        poses.append(step_pose(poses[-1], action))
    return poses


def make_pose(row, col, direction):
    return Pose(int(row), int(col), direction)
