"""Head pose kinematics, independent of the environment.

A world model is evaluated against a known initial state and a chosen
action sequence, so the head's path has to be reconstructible without
stepping the simulator. These helpers integrate it from actions alone.

The rotation table mirrors ``SnakeEnv._next_direction`` exactly, and
``tests/test_grading.py`` checks it against the env for every
(heading, action) pair rather than trusting the transcription.
"""
from typing import NamedTuple

from marlenv.core.snake import Direction

# action 0 keeps the heading, 1 turns left (+90 degrees), 2 turns right
LEFT_TURN = {
    Direction.UP: Direction.LEFT,
    Direction.LEFT: Direction.DOWN,
    Direction.DOWN: Direction.RIGHT,
    Direction.RIGHT: Direction.UP,
}
RIGHT_TURN = {value: key for key, value in LEFT_TURN.items()}


class Pose(NamedTuple):
    """A head position and the direction it is travelling."""

    row: int
    col: int
    direction: Direction

    @property
    def coord(self):
        return self.row, self.col


def turn(direction, action):
    """The heading after taking ``action`` from ``direction``."""
    if action == 0:
        return direction
    if action == 1:
        return LEFT_TURN[direction]
    if action == 2:
        return RIGHT_TURN[direction]
    raise ValueError(f'action must be 0, 1 or 2; got {action!r}')


def step_pose(pose, action):
    """Turn, then advance one cell -- the order ``SnakeEnv.step`` uses."""
    direction = turn(pose.direction, action)
    row = pose.row + direction.value[0]
    col = pose.col + direction.value[1]
    return Pose(row, col, direction)


def action_seq_to_pose_seq(initial_pose, actions):
    """Integrate a pose forward through an action sequence.

    Returns ``len(actions) + 1`` poses, starting with ``initial_pose``.

    This is pure kinematics: it does not know about walls, other snakes or
    death, so it keeps walking a snake that the simulator would have
    killed. Compare against the simulator to find where that happens.
    """
    poses = [initial_pose]
    for action in actions:
        poses.append(step_pose(poses[-1], action))
    return poses


def pose_from_snake(snake):
    """The pose of a live snake in the environment."""
    return Pose(int(snake.head_coord[0]), int(snake.head_coord[1]),
                snake.direction)
