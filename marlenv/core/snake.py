from collections import deque
from enum import Enum


class Cell(Enum):
    EMPTY = 0
    WALL = 1
    FRUIT = 2
    HEAD = 3
    BODY = 4
    TAIL = 5


# the palette lives in marlenv.core.palette, which owns the separation
# analysis that keeps a rendered frame decodable back to the grid
def _load_cell_colors():
    from marlenv.core.palette import CELL_COLORS
    return CELL_COLORS


class _CellColors(dict):
    """Lazy palette view, breaking the snake/palette import cycle."""

    def _ensure(self):
        if not super().__len__():
            self.update(_load_cell_colors())

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def __len__(self):
        self._ensure()
        return super().__len__()

    def __iter__(self):
        self._ensure()
        return super().__iter__()


CellColors = _CellColors()


class Direction(Enum):
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)

    """
    little magic for convenience, coord + direction
    """

    def __radd__(self, other):
        dr, dc = self.value
        return other[0] + dr, other[1] + dc

    def __rsub__(self, other):
        dr, dc = self.value
        return other[0] - dr, other[1] - dc


class Snake:
    def __init__(self, idx, coords):
        assert len(coords) > 1
        self.idx: int = idx
        self.head_coord: tuple = coords[0]
        self.tail_coord: tuple = coords[-1]
        self.direction: Direction = Direction(
            (coords[0][0] - coords[1][0],
            coords[0][1] - coords[1][1])
        )
        prev_coord = self.head_coord
        direction_list = []
        for next_coord in coords[1:]:
            direction = Direction(
                (prev_coord[0] - next_coord[0], prev_coord[1] - next_coord[1])
            )
            direction_list.append(direction)
            prev_coord = next_coord

        self.directions = deque(direction_list)

        self.alive = True
        self._reset_reward_state()

    def __len__(self):
        return len(self.directions + 1)

    def _reset_reward_state(self):
        self.fruit = False
        self.death = False
        self.kills = 0
        self.win = False
        self.reward = 0.

    @property
    def coords(self):
        coord = self.head_coord
        coords = [coord]
        for direction in self.directions:
            coord -= direction
            coords.append(coord)

        return coords

    def move(self):
        self.head_coord += self.direction
        self.directions.appendleft(self.direction)

        prev_tail_coord = None
        if not self.fruit:
            prev_tail_coord = self.tail_coord
            tail_direction = self.directions.pop()
            self.tail_coord += tail_direction
        self._reset_reward_state()

        return prev_tail_coord
