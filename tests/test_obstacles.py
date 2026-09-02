import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.grid_util import (add_obstacles, dfs_sweep_empty,
                                    empty_space_is_connected, make_grid)
from marlenv.core.snake import Cell


def test_connectivity_check_detects_a_split_board():
    grid = make_grid(9, 9)
    assert empty_space_is_connected(grid)

    grid[:, 4] = Cell.WALL.value  # wall straight down the middle
    assert not empty_space_is_connected(grid)


def test_obstacles_never_disconnect_the_board():
    """Connectivity is the property the generator exists to preserve."""
    for seed in range(40):
        rng = np.random.default_rng(seed)
        grid = make_grid(13, 13)
        add_obstacles(grid, 25, np_random=rng)

        assert empty_space_is_connected(grid), f'seed {seed} split the board'


def test_obstacles_are_placed_inside_the_border():
    rng = np.random.default_rng(0)
    grid = make_grid(11, 11)
    add_obstacles(grid, 12, np_random=rng)
    interior = grid[1:-1, 1:-1]

    assert (interior == Cell.WALL.value).any()
    assert (grid[0] == Cell.WALL.value).all()
    assert (grid[-1] == Cell.WALL.value).all()


def test_crowded_board_places_fewer_rather_than_failing():
    rng = np.random.default_rng(0)
    grid = make_grid(7, 7)

    placed = add_obstacles(grid, 200, np_random=rng)

    assert placed < 200
    assert empty_space_is_connected(grid)


def test_generation_is_reproducible():
    grids = []
    for _ in range(2):
        grid = make_grid(13, 13)
        add_obstacles(grid, 15, np_random=np.random.default_rng(7))
        grids.append(grid)

    assert np.array_equal(grids[0], grids[1])


@pytest.mark.parametrize('seed', range(12))
def test_env_boards_stay_playable(seed):
    """Every generated board must be connected and have spawn room."""
    env = gym.make('Snake-v1', num_snakes=3, snake_length=3,
                   obstacle_density=0.08, grid_size_range=(11, 17),
                   disable_env_checker=True)
    env.reset(seed=seed)
    base = env.unwrapped

    walls = base.grid % 10 == Cell.WALL.value
    empty_or_snake = np.where(walls, Cell.WALL.value, Cell.EMPTY.value)
    assert empty_space_is_connected(empty_or_snake)
    assert base.grid.shape[0] == base.grid.shape[1]
    assert 11 <= base.grid.shape[0] <= 17
    assert len({tuple(s.head_coord) for s in base.snakes}) == 3

    # fruit and snakes must never be placed on a wall
    assert not (walls & (base.grid % 10 == Cell.FRUIT.value)).any()
    for snake in base.snakes:
        for coord in snake.coords:
            assert base.grid[coord] % 10 != Cell.WALL.value


def test_episodes_run_on_obstacle_boards():
    env = gym.make('Snake-v1', num_snakes=3, obstacle_density=0.1,
                   grid_size_range=(11, 15), disable_env_checker=True)
    for seed in range(6):
        env.reset(seed=seed)
        env.action_space.seed(seed)
        for _ in range(40):
            _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
            if all(term) or all(trunc):
                break


def test_defaults_are_unchanged():
    """An env built the old way must still get a fixed, empty board."""
    env = gym.make('Snake-v1', height=12, width=12, num_snakes=1,
                   disable_env_checker=True)
    env.reset(seed=0)
    base = env.unwrapped

    assert base.grid.shape == (12, 12)
    assert (base.grid[1:-1, 1:-1] % 10 != Cell.WALL.value).all()
