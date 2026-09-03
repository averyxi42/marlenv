import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.snake import Cell, CellColors

EMPTY_RGB = np.array(CellColors[Cell.EMPTY.value][0], dtype=np.uint8)
WALL_RGB = np.array(CellColors[Cell.WALL.value][0], dtype=np.uint8)


def make(radius=4, noise=0.0, num_snakes=3, side=13, gradient=0.0):
    # the heading gradient is off unless a test is about it, so the others
    # can compare against exact palette colours
    env = gym.make('Snake-v1', height=side, width=side,
                   num_snakes=num_snakes, num_fruits=3,
                   view_radius=radius, observation_noise=noise,
                   background_gradient=gradient,
                   disable_env_checker=True)
    return env


def test_views_are_odd_squares_one_per_snake():
    env = make(radius=4)
    env.reset(seed=0)

    views = env.unwrapped.egocentric_rgb()

    assert views.shape == (3, 9, 9, 3)
    assert views.shape[1] % 2 == 1


def test_view_is_centred_on_the_head():
    env = make(radius=3, noise=6.0)
    env.reset(seed=0)
    base = env.unwrapped
    frame = base._padded_rgb(3)

    views = base.egocentric_rgb()

    for i, snake in enumerate(base.snakes):
        head = (snake.head_coord[0] + 3, snake.head_coord[1] + 3)
        assert np.array_equal(views[i][3, 3], frame[head])


def test_view_is_rotated_into_the_head_frame():
    """The neck sits directly below centre whatever the snake's heading."""
    env = make(radius=4, num_snakes=2)
    env.reset(seed=0)
    env.action_space.seed(0)
    base = env.unwrapped
    headings = set()

    for _ in range(30):
        views = base.egocentric_rgb()
        for i, snake in enumerate(base.snakes):
            if not snake.alive:
                continue
            neck_world = snake.coords[1]
            frame = base._padded_rgb(4)
            expected = frame[neck_world[0] + 4, neck_world[1] + 4]
            assert np.array_equal(views[i][5, 4], expected)
            headings.add(snake.direction)
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        if all(term) or all(trunc):
            break

    assert len(headings) > 1, 'only one heading was exercised'


def test_beyond_the_board_is_free_space_not_wall():
    """Padding with wall would invent a barrier that is not there."""
    env = make(radius=6, noise=0.0, num_snakes=1, side=11)
    env.reset(seed=0)
    base = env.unwrapped
    # put the snake's head in a corner so the view overhangs two edges
    base.snakes[0].head_coord = (1, 1)

    view = base.egocentric_rgb()[0]
    # with radius 6 and the head at (1, 1), rows 0-3 of the unrotated view
    # are outside the board; the snake faces UP so no rotation applies
    outside = view[0:4]

    assert np.all(outside == EMPTY_RGB), 'padding was not free space'
    assert not np.any(np.all(outside == WALL_RGB, axis=-1))


def test_the_board_border_still_reads_as_wall():
    env = make(radius=6, noise=0.0, num_snakes=1, side=11)
    env.reset(seed=0)
    base = env.unwrapped
    base.snakes[0].head_coord = (1, 1)

    view = base.egocentric_rgb()[0]

    # the real border wall must be visible between padding and interior
    assert np.any(np.all(view == WALL_RGB, axis=-1))


def test_padding_carries_persistent_noise():
    """Free space outside the grid needs the same fixed noise as inside."""
    env = make(radius=5, noise=8.0, num_snakes=1, side=11)
    env.reset(seed=0)
    base = env.unwrapped

    assert base.obs_noise.cell_noise.shape == (11 + 10, 11 + 10, 3)

    base.snakes[0].head_coord = (1, 1)
    first = base.egocentric_rgb()[0]
    second = base.egocentric_rgb()[0]
    assert np.array_equal(first, second)
    # and the overhang is not flat, i.e. it actually carries noise
    assert len(np.unique(first[0:3].reshape(-1, 3), axis=0)) > 1


def test_dead_snakes_observe_nothing():
    env = make(radius=3)
    env.reset(seed=0)
    base = env.unwrapped
    base.snakes[1].alive = False

    views = base.egocentric_rgb()

    assert np.all(views[1] == 0)
    assert np.any(views[0] != 0)


def test_view_radius_is_required():
    env = gym.make('Snake-v1', height=11, width=11, num_snakes=1,
                   disable_env_checker=True)
    env.reset(seed=0)

    with pytest.raises(RuntimeError, match='view_radius'):
        env.unwrapped.egocentric_rgb()
