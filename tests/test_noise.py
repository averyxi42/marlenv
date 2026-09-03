import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.noise import ObservationNoise
from marlenv.core.snake import Cell, Snake

SIGMA = 8.0


@pytest.fixture
def env():
    # gradient off: these tests are about the noise fields, and the gradient
    # would add its own colour variation on top
    return gym.make('Snake-v1', height=13, width=13, num_snakes=2,
                    num_fruits=3, observation_noise=SIGMA, noise_period=4,
                    background_gradient=0.0, disable_env_checker=True)


def test_noise_fields_have_the_documented_shapes(env):
    env.reset(seed=0)
    noise = env.unwrapped.obs_noise

    assert noise.cell_noise.shape == (13, 13, 3)
    assert noise.snake_noise.shape == (2, 4, 3)


def test_noise_makes_the_observation_continuous(env):
    env.reset(seed=0)
    noisy = env.unwrapped.render('rgb_array')

    clean_env = gym.make('Snake-v1', height=13, width=13, num_snakes=2,
                         num_fruits=3, background_gradient=0.0,
                         disable_env_checker=True)
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


def test_texture_does_not_drift_along_the_body():
    """The colour sequence read head-to-tail must be the same every step.

    Path distance from the head is a *material* coordinate: the body slides
    along its own path, so a material element keeps its distance from the
    head and therefore its colour. Keying the noise to anything fixed in
    world space instead would make the pattern shift along the body each
    step, i.e. paint a trail rather than scales.
    """
    env = gym.make('Snake-v1', height=13, width=13, num_snakes=1,
                   num_fruits=3, observation_noise=SIGMA, noise_period=4,
                   background_gradient=0.0, disable_env_checker=True)
    env.reset(seed=2)
    env.action_space.seed(2)
    base = env.unwrapped

    def along_body():
        frame = base.render('rgb_array')
        return [tuple(frame[coord]) for coord in base.snakes[0].coords]

    reference = along_body()
    steps = 0
    for _ in range(20):
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        if not base.snakes[0].alive:
            break
        current = along_body()
        shared = min(len(reference), len(current))
        assert current[:shared] == reference[:shared], \
            'texture drifted along the body'
        reference = current
        steps += 1
        if all(term) or all(trunc):
            break

    assert steps > 5, 'test never followed enough steps'


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
                       observation_noise=sigma, background_gradient=0.0,
                       disable_env_checker=True)
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
                   background_gradient=0.0, disable_env_checker=True)
    env.reset(seed=0)

    assert env.unwrapped.obs_noise is None
    frame = env.unwrapped.render('rgb_array')
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) <= 6


def _snake_deviations(base, snake_idx=0):
    """How far each snake cell sits from its palette colour."""
    from marlenv.core.palette import cell_color

    frame = base.render('rgb_array')
    snake = base.snakes[snake_idx]
    out = []
    for coord in snake.coords:
        kind = base.grid[coord] % 10
        expected = cell_color(kind, snake.idx)
        out.append(float(np.abs(frame[coord].astype(float) - expected).max()))
    return out


def _background_colours(base):
    frame = base.render('rgb_array')
    empty = base.grid == Cell.EMPTY.value
    return {tuple(c) for c in frame[empty]}


def test_background_and_snake_sigmas_are_independent():
    """Each field can be scaled, or switched off, without the other."""
    quiet_snakes = gym.make('Snake-v1', height=13, width=13, num_snakes=2,
                            observation_noise=10.0, snake_noise_sigma=0.0,
                            background_gradient=0.0,
                            disable_env_checker=True)
    quiet_snakes.reset(seed=0)
    base = quiet_snakes.unwrapped
    assert max(_snake_deviations(base)) == 0, 'snake noise leaked in'
    assert len(_background_colours(base)) > 5

    quiet_background = gym.make('Snake-v1', height=13, width=13, num_snakes=2,
                                observation_noise=0.0, snake_noise_sigma=25.0,
                                background_gradient=0.0,
                                disable_env_checker=True)
    quiet_background.reset(seed=0)
    base = quiet_background.unwrapped
    assert len(_background_colours(base)) == 1, 'background noise leaked in'
    assert max(_snake_deviations(base)) > 0


def test_snake_sigma_defaults_to_the_background_sigma():
    env = gym.make('Snake-v1', height=13, width=13, num_snakes=2,
                   observation_noise=7.0, background_gradient=0.0,
                   disable_env_checker=True)
    env.reset(seed=0)

    noise = env.unwrapped.obs_noise
    assert noise.sigma == 7.0
    assert noise.snake_sigma == 7.0
