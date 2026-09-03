import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.noise import ObservationNoise
from marlenv.core.snake import Cell, Snake

SIGMA = 8.0


@pytest.fixture
def env():
    return gym.make('Snake-v1', height=13, width=13, num_snakes=2,
                    num_fruits=3, observation_noise=SIGMA, noise_period=4,
                    disable_env_checker=True)


def test_noise_fields_have_the_documented_shapes(env):
    env.reset(seed=0)
    noise = env.unwrapped.obs_noise

    assert noise.cell_noise.shape == (13, 13, 3)
    assert noise.snake_noise.shape == (2, 4, 3)


def test_noise_makes_the_observation_continuous(env):
    env.reset(seed=0)
    noisy = env.unwrapped.render('rgb_array')

    clean_env = gym.make('Snake-v1', height=13, width=13, num_snakes=2,
                         num_fruits=3, disable_env_checker=True)
    clean_env.reset(seed=0)
    clean = clean_env.unwrapped.render('rgb_array')

    assert len(np.unique(noisy.reshape(-1, 3), axis=0)) > \
        10 * len(np.unique(clean.reshape(-1, 3), axis=0))


def test_static_cells_never_flicker(env):
    """The point of binding: unchanged cells must render identically.

    A world model can only learn noise that the state determines; per-frame
    noise would churn every static pixel.
    """
    env.reset(seed=0)
    env.action_space.seed(0)
    base = env.unwrapped

    previous, previous_grid = base.render('rgb_array'), base.grid.copy()
    for _ in range(20):
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        frame, grid = base.render('rgb_array'), base.grid.copy()

        static = ((previous_grid == Cell.EMPTY.value)
                  & (grid == Cell.EMPTY.value))
        assert np.array_equal(frame[static], previous[static])

        walls = (previous_grid % 10 == Cell.WALL.value) & \
                (grid % 10 == Cell.WALL.value)
        assert np.array_equal(frame[walls], previous[walls])

        previous, previous_grid = frame, grid
        if all(term) or all(trunc):
            break


def test_snake_noise_repeats_with_the_body_period():
    """Body cells are indexed by distance from the head, modulo the period."""
    period = 3
    coords = [(5, 3 + i) for i in range(7)]
    snake = Snake(0, coords)
    grid = np.zeros((11, 11), dtype=np.int64)
    for coord in coords:
        grid[coord] = Cell.BODY.value

    rgb = np.zeros((11, 11, 3), dtype=np.uint8)
    rgb[:] = 100  # flat base colour, so any difference is the noise
    noise = ObservationNoise(grid.shape, 1, sigma=SIGMA, period=period,
                             np_random=np.random.default_rng(0))
    out = noise.apply(rgb, [snake])

    for distance in range(len(coords) - period):
        here = out[coords[distance]]
        later = out[coords[distance + period]]
        assert np.array_equal(here, later)


def test_noise_does_not_disturb_the_dynamics():
    """Noise draws come from a spawned stream, so episodes stay paired."""
    def trajectory(sigma):
        env = gym.make('Snake-v1', height=13, width=13, num_snakes=2,
                       observation_noise=sigma, disable_env_checker=True)
        env.reset(seed=7)
        env.action_space.seed(7)
        grids = [env.unwrapped.grid.copy()]
        for _ in range(20):
            _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
            grids.append(env.unwrapped.grid.copy())
            if all(term) or all(trunc):
                break
        return np.array(grids)

    assert np.array_equal(trajectory(0.0), trajectory(SIGMA))


def test_noise_is_reproducible(env):
    env.reset(seed=3)
    first = env.unwrapped.render('rgb_array')
    env.reset(seed=3)
    second = env.unwrapped.render('rgb_array')

    assert np.array_equal(first, second)


def test_noise_is_off_by_default():
    env = gym.make('Snake-v1', height=11, width=11, num_snakes=1,
                   disable_env_checker=True)
    env.reset(seed=0)

    assert env.unwrapped.obs_noise is None
    frame = env.unwrapped.render('rgb_array')
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) <= 6
