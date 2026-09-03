"""The palette must let a rendered frame decode back to the grid."""
import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.palette import (CELL_COLORS, decode_grid, palette_entries,
                                  safety_report)
from marlenv.core.snake import Cell

SIGMA_BG = 4.0
SIGMA_SNAKE = 6.0


def test_every_class_has_a_distinct_colour():
    """The old palette gave head, body and tail the same colour."""
    _, colors = palette_entries(6)
    unique = np.unique(colors, axis=0)

    assert len(unique) == len(colors)


def test_shipped_sigmas_leave_a_decoding_margin():
    slack, why = safety_report(6, SIGMA_BG, SIGMA_SNAKE)

    assert slack > 0, f'shipped noise defaults are not decodable: {why}'


def test_report_flags_a_configuration_that_is_too_noisy():
    slack, _ = safety_report(6, SIGMA_BG, 40.0)

    assert slack < 0


def test_relaxing_part_identity_buys_headroom():
    """Parts of one snake are fungible when the evaluator knows the actions."""
    strict, _ = safety_report(6, SIGMA_BG, 8.0, strict=True)
    relaxed, _ = safety_report(6, SIGMA_BG, 8.0, strict=False)

    assert relaxed > strict


@pytest.mark.parametrize('num_snakes', [1, 4, 6])
def test_frames_decode_back_to_the_grid_exactly(num_snakes):
    """The grading path: image in, true grid out, noise and gradient on."""
    env = gym.make('Snake-v1', height=15, width=15, num_snakes=num_snakes,
                   num_fruits=4, observation_noise=SIGMA_BG,
                   snake_noise_sigma=SIGMA_SNAKE, background_gradient=28.0,
                   disable_env_checker=True)
    env.reset(seed=0)
    env.action_space.seed(0)
    base = env.unwrapped

    wrong = 0
    for _ in range(40):
        decoded = decode_grid(base.render('rgb_array'), num_snakes)
        wrong += int((decoded != base.grid).sum())
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        if all(term) or all(trunc):
            break

    assert wrong == 0


def test_decoding_survives_obstacles_and_a_variable_board():
    env = gym.make('Snake-v1', num_snakes=3, num_fruits=4,
                   obstacle_density=0.12, grid_size_range=(11, 17),
                   observation_noise=SIGMA_BG, snake_noise_sigma=SIGMA_SNAKE,
                   background_gradient=28.0, disable_env_checker=True)
    for seed in range(6):
        env.reset(seed=seed)
        base = env.unwrapped
        decoded = decode_grid(base.render('rgb_array'), 3)
        assert np.array_equal(decoded, base.grid)


def test_gradient_stays_off_walls_and_fruit():
    """Objects keep fixed colours, which is what makes decoding phase-free."""
    env = gym.make('Snake-v1', height=15, width=15, num_snakes=2,
                   num_fruits=4, observation_noise=0.0, snake_noise_sigma=0.0,
                   background_gradient=30.0, disable_env_checker=True)
    env.reset(seed=0)
    base = env.unwrapped
    frame = base.render('rgb_array')

    for kind, expected in ((Cell.WALL.value, CELL_COLORS[Cell.WALL.value][0]),
                           (Cell.FRUIT.value,
                            CELL_COLORS[Cell.FRUIT.value][0])):
        mask = base.grid % 10 == kind
        if mask.any():
            assert np.all(frame[mask] == np.array(expected, dtype=np.uint8))

    # while empty cells do vary, that being the point of the gradient
    empty = base.grid == Cell.EMPTY.value
    assert len(np.unique(frame[empty], axis=0)) > 1


def test_snake_colors_wraps_hues_for_a_crowded_board():
    """A model only knows the hues it trained with.

    Snake colour is assigned by index, so a fourth snake on a board trained
    with three arrives in a hue the model has never seen -- and it renders
    those cells wrong every single time. Capping the number of colours wraps
    the extra snakes back onto known hues, which costs nothing where
    identity is carried by position rather than by colour.
    """
    import gymnasium as gym
    import marlenv  # noqa: F401
    from marlenv.core.palette import decode_grid
    from marlenv.grading.compare import PALETTE_SNAKES

    def centres(colors):
        env = gym.make('Snake-v1', height=15, width=15, num_snakes=5,
                       num_fruits=4, view_radius=4, observation_noise=0.0,
                       snake_noise_sigma=0.0, background_gradient=0.0,
                       snake_colors=colors, disable_env_checker=True)
        env.reset(seed=0)
        base = env.unwrapped
        middle = base.view_radius
        return [int(decode_grid(view, PALETTE_SNAKES)[middle, middle])
                for view in base.egocentric_rgb()]

    own = centres(None)
    assert [code // 10 for code in own] == [0, 1, 2, 3, 4]

    wrapped = centres(3)
    assert [code // 10 for code in wrapped] == [0, 1, 2, 0, 1]
    # every viewpoint is still centred on a head, whichever hue it wears
    assert all(code % 10 == Cell.HEAD.value for code in wrapped)
