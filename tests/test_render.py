import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.render import draw_frame
from marlenv.core.snake import Cell


@pytest.fixture
def env():
    return gym.make('Snake-v1', height=11, width=11, num_snakes=2,
                    num_fruits=3, render_style='pixel', cell_size=12,
                    disable_env_checker=True)


def test_frame_is_scaled_by_cell_size(env):
    env.reset(seed=0)
    base = env.unwrapped

    frame = draw_frame(base.grid, base.snakes, cell_size=12)

    assert frame.shape == (11 * 12, 11 * 12, 3)
    assert frame.dtype == np.uint8


def test_env_render_honours_the_style(env):
    env.reset(seed=0)
    base = env.unwrapped

    pixel = base.render('rgb_array')
    classic = base.render('rgb_array', style='classic')

    assert pixel.shape == (11 * 12, 11 * 12, 3)
    assert classic.shape == (11, 11, 3)


def test_classic_stays_the_default():
    """The library default must not change under existing callers."""
    env = gym.make('Snake-v1', height=11, width=11, num_snakes=1,
                   disable_env_checker=True)
    env.reset(seed=0)

    assert env.unwrapped.render_style == 'classic'
    assert env.unwrapped.render('rgb_array').shape == (11, 11, 3)


def test_dead_snakes_are_not_drawn(env):
    env.reset(seed=0)
    base = env.unwrapped

    both = draw_frame(base.grid, base.snakes, 12)
    base.snakes[0].alive = False
    one = draw_frame(base.grid, base.snakes, 12)

    assert not np.array_equal(both, one)


def test_heading_is_visible_in_the_frame(env):
    """Two states with identical cells but different headings must differ.

    This is the case that made the flat renderer drop GIF frames: a snake
    rotating within its own occupied cells rendered byte-identically.
    """
    from marlenv.core.snake import Snake

    env.reset(seed=0)
    base = env.unwrapped
    # a length-4 snake coiled in a 2x2 loop, chasing its own tail: stepping
    # it leaves the occupied cells identical and only rotates the roles
    coiled = [(5, 8), (6, 8), (6, 7), (5, 7)]
    stepped = [(5, 7), (5, 8), (6, 8), (6, 7)]
    assert sorted(coiled) == sorted(stepped)

    base.grid[:] = Cell.EMPTY.value
    for coord in coiled:
        base.grid[coord] = Cell.BODY.value

    before = draw_frame(base.grid, [Snake(0, coiled)], 12)
    after = draw_frame(base.grid, [Snake(0, stepped)], 12)

    assert not np.array_equal(before, after)


def test_fruit_is_smaller_than_its_cell(env):
    env.reset(seed=0)
    base = env.unwrapped
    base.grid[:] = Cell.EMPTY.value
    base.grid[5, 5] = Cell.FRUIT.value

    frame = draw_frame(base.grid, [], 12)
    cell = frame[5 * 12:6 * 12, 5 * 12:6 * 12]
    background = frame[2 * 12:3 * 12, 2 * 12:3 * 12]

    coloured = ~np.all(cell == cell[0, 0], axis=-1)
    assert coloured.any(), 'fruit was not drawn'
    assert coloured.mean() < 0.6, 'fruit should not fill the cell'
    assert not np.array_equal(cell, background)


def test_gif_buffer_uses_the_styled_frame(env):
    env.reset(seed=0)
    base = env.unwrapped

    base.render('gif')

    assert base.frame_buffer[0].size == (11 * 12, 11 * 12)
