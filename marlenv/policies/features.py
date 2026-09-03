"""Structured (non-pixel) observations for the learned solver.

Each snake gets its own view built from the env's grid, laid out so that the
network can be shared across snakes:

* **identity equivariant** -- a snake's view never encodes *which* snake it
  is, only "mine" versus "theirs", so permuting snake indices permutes the
  outputs and nothing else;
* **cardinality flexible** -- the channel count does not depend on the number
  of snakes, so one network handles any number of them;
* **heading canonical** -- every view is rotated so the snake faces up, which
  is what makes the relative actions (noop/left/right) mean the same thing in
  every view.
"""
import numpy as np

from marlenv.core.observation import HEADING_ROTATIONS
from marlenv.core.snake import Cell

# channel layout of a per-snake view
CHANNELS = (
    'wall',
    'fruit',
    'my_head',
    'my_body',
    'my_tail',
    'other_head',
    'other_body',
    'other_tail',
)
NUM_CHANNELS = len(CHANNELS)

class GridFeatures:
    """Per-snake feature planes derived from a SnakeEnv's grid.

    Splitting the work in two matters for search throughput: the parts that
    depend only on the grid are computed once per state, and only the
    "mine versus theirs" split is redone per snake.
    """

    def __init__(self, env):
        base = env.unwrapped
        grid = base.grid
        if grid.shape[0] != grid.shape[1]:
            raise ValueError(
                'heading canonicalisation rotates the grid, so it must be '
                f'square; got {grid.shape}')
        self.base = base
        cell = grid % 10
        self.snake_id = grid // 10
        self.wall = cell == Cell.WALL.value
        self.fruit = cell == Cell.FRUIT.value
        self.head = cell == Cell.HEAD.value
        self.body = cell == Cell.BODY.value
        self.tail = cell == Cell.TAIL.value

    def for_snake(self, idx):
        """The view of snake ``idx``: ``(NUM_CHANNELS, H, W)`` float32."""
        snake = self.base.snakes[idx]
        mine = self.snake_id == idx
        planes = np.stack([
            self.wall,
            self.fruit,
            self.head & mine,
            self.body & mine,
            self.tail & mine,
            self.head & ~mine,
            self.body & ~mine,
            self.tail & ~mine,
        ]).astype(np.float32)

        rotations = HEADING_ROTATIONS[snake.direction]
        if rotations:
            planes = np.rot90(planes, rotations, axes=(1, 2))
        return np.ascontiguousarray(planes)


def observe(env, indices=None):
    """Stack the views of the given snakes into ``(n, C, H, W)`` float32.

    ``indices`` defaults to every living snake.
    """
    base = env.unwrapped
    if indices is None:
        indices = [i for i, s in enumerate(base.snakes) if s.alive]
    if not indices:
        grid = base.grid
        return (np.zeros((0, NUM_CHANNELS, *grid.shape), dtype=np.float32),
                list(indices))

    features = GridFeatures(env)
    stacked = np.stack([features.for_snake(i) for i in indices])
    return stacked, list(indices)


def head_positions(planes):
    """Row/col of each view's own head, as ``(n, 2)`` int64.

    Read back off the rotated ``my_head`` plane rather than transformed by
    hand, so it cannot drift out of sync with the rotation above.
    """
    my_head = planes[:, CHANNELS.index('my_head')]
    flat = my_head.reshape(len(planes), -1).argmax(axis=1)
    return np.stack([flat // my_head.shape[2], flat % my_head.shape[2]],
                    axis=1)
