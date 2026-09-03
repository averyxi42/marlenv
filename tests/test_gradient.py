"""The heading gradient: an absolute cue inside a head-frame observation."""
import itertools

import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.observation import egocentric_crop, heading_gradient
from marlenv.core.snake import Cell, CellColors, Direction

EMPTY_RGB = np.array(CellColors[Cell.EMPTY.value][0], dtype=np.uint8)


def test_stripes_run_along_one_direction_and_repeat():
    field = heading_gradient((40, 40), pad=0, period=6, amplitude=28.0)

    # constant across the stripes, periodic along them
    assert np.allclose(field[3, 0], field[3, 17])
    assert np.allclose(field[3, 0], field[9, 0])
    # and it is one direction, not a plaid: no variation along a stripe
    assert np.allclose(field[5, :, 0], field[5, 0, 0])


def test_channels_are_in_quadrature_and_never_clip():
    """Red and green carry the same phase 90 degrees apart.

    Offset into [0, amplitude] so the negative half survives against a black
    background; a zero-mean signal would clip and lose half the encoding.
    """
    amplitude = 28.0
    field = heading_gradient((40, 40), pad=0, period=8, amplitude=amplitude)

    assert field.min() >= 0.0
    assert field.max() <= amplitude + 1e-5

    red = field[:, 0, 0] - amplitude / 2
    green = field[:, 0, 1] - amplitude / 2
    radius = red ** 2 + green ** 2
    assert np.allclose(radius, radius[0])


def test_angle_rotates_the_stripes():
    along_rows = heading_gradient((30, 30), period=6, angle=0.0)
    along_cols = heading_gradient((30, 30), period=6, angle=90.0)

    assert np.allclose(along_rows[4, :, 0], along_rows[4, 0, 0])
    assert np.allclose(along_cols[:, 4, 0], along_cols[0, 4, 0])


def test_every_heading_looks_different_in_the_head_frame():
    """The point of the gradient: four headings, four distinguishable views.

    Without it a rotated crop of a uniform background is identical whichever
    way the snake faces, so heading is unrecoverable from the observation.
    """
    field = heading_gradient((41, 41), pad=0, period=6, amplitude=28.0)
    head = (20, 20)

    views = {}
    for direction in Direction:
        views[direction] = egocentric_crop(field, head, direction,
                                           radius=4, pad=0)

    for a, b in itertools.combinations(Direction, 2):
        assert not np.allclose(views[a], views[b]), \
            f'{a.name} and {b.name} are indistinguishable'


def test_quadrature_is_what_separates_opposite_headings():
    """A single channel cannot tell UP from DOWN; the pair can.

    Rotating a lone sinusoid by 180 degrees reproduces it up to a phase
    shift, so its crop is not distinctive. The ordered (red, green) pair
    reverses its direction of travel round the colour circle, which is.
    """
    field = heading_gradient((41, 41), pad=0, period=6, amplitude=28.0)
    # a row where the cosine is centred, so red is symmetric about the head
    # and therefore survives the half turn unchanged
    head = (21, 20)

    up = egocentric_crop(field, head, Direction.UP, radius=4, pad=0)
    down = egocentric_crop(field, head, Direction.DOWN, radius=4, pad=0)

    message = 'red alone should not separate these two headings'
    assert np.allclose(up[..., 0], down[..., 0]), message
    # the quadrature pair does not
    assert not np.allclose(up, down)


def test_a_flat_background_cannot_encode_heading():
    """Control: with no gradient the four views coincide, as expected."""
    flat = np.zeros((41, 41, 3), dtype=np.float32)
    views = [egocentric_crop(flat, (20, 20), d, radius=4, pad=0)
             for d in Direction]

    assert all(np.allclose(views[0], v) for v in views[1:])


def test_gradient_reaches_the_egocentric_view():
    env = gym.make('Snake-v1', height=15, width=15, num_snakes=1,
                   num_fruits=3, view_radius=4, observation_noise=0.0,
                   background_gradient=28.0, disable_env_checker=True)
    env.reset(seed=0)

    view = env.unwrapped.egocentric_rgb()[0]
    background = view.reshape(-1, 3)

    # the background is no longer a single flat colour
    assert len(np.unique(background, axis=0)) > 4


def test_gradient_never_touches_the_snakes():
    """A snake must look the same wherever it stands.

    Checked against the palette rather than by counting colours, since head,
    body and tail are deliberately different from one another.
    """
    from marlenv.core.palette import cell_color

    env = gym.make('Snake-v1', height=15, width=15, num_snakes=2,
                   num_fruits=3, observation_noise=0.0, snake_noise_sigma=0.0,
                   background_gradient=30.0, disable_env_checker=True)
    env.reset(seed=0)
    env.action_space.seed(0)
    base = env.unwrapped

    for _ in range(15):
        frame = base.render('rgb_array')
        for snake in base.snakes:
            if not snake.alive:
                continue
            for coord in snake.coords:
                kind = base.grid[coord] % 10
                expected = cell_color(kind, snake.idx).astype(np.uint8)
                assert np.array_equal(frame[coord], expected), \
                    'gradient or noise bled onto a snake'
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        if all(term) or all(trunc):
            break


def test_gradient_is_on_by_default():
    env = gym.make('Snake-v1', height=11, width=11, num_snakes=1,
                   disable_env_checker=True)
    env.reset(seed=0)

    assert env.unwrapped.background_gradient > 0
    frame = env.unwrapped.render('rgb_array')
    empty = env.unwrapped.grid == Cell.EMPTY.value
    assert len(np.unique(frame[empty], axis=0)) > 1


def test_gradient_can_be_disabled():
    env = gym.make('Snake-v1', height=11, width=11, num_snakes=1,
                   background_gradient=0.0, disable_env_checker=True)
    env.reset(seed=0)

    frame = env.unwrapped.render('rgb_array')
    empty = env.unwrapped.grid == Cell.EMPTY.value
    assert np.all(frame[empty] == EMPTY_RGB)
