"""Observation alignment and the grading histogram."""
import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.palette import (cell_color, decode_grid, snap_to_palette)
from marlenv.core.snake import Cell, Direction
from marlenv.grading import (ConfusionMatrix, NUM_CLASSES, align_obs,
                             diff_local_obs, diff_obs, grade,
                             action_seq_to_pose_seq, pose_from_snake, turn)
from marlenv.grading.poses import Pose
from marlenv.grading.rollout import record_rollout, save_rollout

RADIUS = 4


def make(num_snakes=3, noise=True, side=15, seed=0):
    env = gym.make('Snake-v1', height=side, width=side,
                   num_snakes=num_snakes, num_fruits=4, view_radius=RADIUS,
                   observation_noise=2.0 if noise else 0.0,
                   snake_noise_sigma=8.0 if noise else 0.0,
                   background_gradient=16.0 if noise else 0.0,
                   disable_env_checker=True)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ------------------------------------------------------------------ palette
def test_snake_colour_is_determined_by_agent_index():
    """The whole pipeline assumes colour <-> index is fixed and stable."""
    for seed in range(8):
        env = make(num_snakes=4, noise=False, seed=seed)
        base = env.unwrapped
        for _ in range(10):
            frame = base.render('rgb_array')
            for snake in base.snakes:
                if not snake.alive:
                    continue
                for coord in snake.coords:
                    kind = base.grid[coord] % 10
                    expected = cell_color(kind, snake.idx).astype(np.uint8)
                    assert np.array_equal(frame[coord], expected)
                    assert base.grid[coord] // 10 == snake.idx
            _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
            if all(term) or all(trunc):
                break


def test_indices_are_stable_across_a_reset_with_the_same_seed():
    first = make(num_snakes=4, noise=False, seed=3).unwrapped.grid.copy()
    second = make(num_snakes=4, noise=False, seed=3).unwrapped.grid.copy()

    assert np.array_equal(first, second)


def test_snapping_removes_noise_without_changing_meaning():
    env = make(noise=True)
    base = env.unwrapped
    noisy = base.render('rgb_array')

    snapped = snap_to_palette(noisy, 6)

    assert len(np.unique(snapped.reshape(-1, 3), axis=0)) < \
        len(np.unique(noisy.reshape(-1, 3), axis=0))
    assert np.array_equal(decode_grid(snapped, 6), decode_grid(noisy, 6))


# -------------------------------------------------------------------- poses
def test_turn_table_matches_the_environment():
    """Transcribed from SnakeEnv._next_direction, so check it against it."""
    env = make(num_snakes=1, noise=False)
    base = env.unwrapped

    for direction in Direction:
        for action in (0, 1, 2):
            assert turn(direction, action) == \
                base._next_direction(direction, action)


def test_pose_integration_tracks_the_simulator():
    """Kinematics alone must reproduce the head path the env produces."""
    env = make(num_snakes=1, noise=False, side=21)
    base = env.unwrapped
    snake = base.snakes[0]
    start = pose_from_snake(snake)

    actions = [0, 0, 1, 0, 0, 2, 0, 1, 0, 0]
    poses = action_seq_to_pose_seq(start, actions)

    for step, action in enumerate(actions):
        _, _, term, _, _ = env.step([action])
        if not base.snakes[0].alive:
            break
        actual = pose_from_snake(base.snakes[0])
        assert actual == poses[step + 1], f'diverged at step {step}'


# ------------------------------------------------------- observation diffing
def test_a_view_agrees_with_the_board_it_came_from():
    env = make(num_snakes=3, noise=True)
    base = env.unwrapped

    for _ in range(12):
        global_obs = base.render('rgb_array')
        views = base.egocentric_rgb()
        for snake in base.snakes:
            if not snake.alive:
                continue
            diff = diff_obs(pose_from_snake(snake), views[snake.idx],
                            global_obs)
            assert len(diff) == 0, f'{len(diff)} cells disagreed'
            assert diff.compared == (2 * RADIUS + 1) ** 2
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        if all(term) or all(trunc):
            break


def test_two_agents_agree_wherever_their_views_overlap():
    """The strongest check on the alignment maths: same world, two frames."""
    env = make(num_snakes=4, noise=True, side=13)
    base = env.unwrapped
    overlaps = 0

    for _ in range(15):
        views = base.egocentric_rgb()
        alive = [s for s in base.snakes if s.alive]
        for a, b in ((x, y) for i, x in enumerate(alive)
                     for y in alive[i + 1:]):
            diff = diff_local_obs(pose_from_snake(a), views[a.idx],
                                  pose_from_snake(b), views[b.idx])
            assert len(diff) == 0, 'two views of one world disagreed'
            overlaps += diff.compared > 0
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        if all(term) or all(trunc):
            break

    assert overlaps > 0, 'no pair of views ever overlapped'


def test_disjoint_views_compare_nothing():
    env = make(num_snakes=1, noise=False, side=15)
    views = env.unwrapped.egocentric_rgb()
    far_apart = diff_local_obs(Pose(2, 2, Direction.UP), views[0],
                               Pose(40, 40, Direction.UP), views[0])

    assert far_apart.compared == 0
    assert len(far_apart) == 0


def test_corruption_is_localised_and_named():
    env = make(num_snakes=2, noise=False)
    base = env.unwrapped
    snake = base.snakes[0]
    pose = pose_from_snake(snake)
    view = base.egocentric_rgb()[snake.idx].copy()

    # paint one cell of the view as another snake's head
    view[0, 0] = cell_color(Cell.HEAD.value, 5).astype(np.uint8)
    diff = diff_obs(pose, view, base.render('rgb_array'))

    assert len(diff) == 1
    assert diff.observed[0] == Cell.HEAD.value + 10 * 5


# ------------------------------------------------------------------ histogram
def test_confusion_matrix_is_the_full_palette():
    matrix = ConfusionMatrix()

    assert matrix.matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert NUM_CLASSES == 21


def test_confusion_matrix_counts_agreements_and_errors():
    env = make(num_snakes=2, noise=True)
    base = env.unwrapped
    snake = base.snakes[0]
    view = base.egocentric_rgb()[snake.idx].copy()
    view[0, 0] = cell_color(Cell.FRUIT.value).astype(np.uint8)

    matrix = ConfusionMatrix()
    matrix.update_from(align_obs(pose_from_snake(snake), view,
                                 base.render('rgb_array')))

    assert matrix.matrix.sum() == (2 * RADIUS + 1) ** 2
    assert matrix.errors >= 1
    assert matrix.top_confusions(1)[0][1] == 'fruit'


# ------------------------------------------------------------------ rollouts
def make_recorded(steps=12, num_snakes=3, noise=False, seed=1):
    env = make(num_snakes=num_snakes, noise=noise, seed=seed)
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 3, size=(steps, num_snakes))
    return record_rollout(env, actions, agent=0), actions


def test_rollout_records_aligned_arrays():
    rollout, actions = make_recorded()

    frames = rollout.steps + 1
    assert len(rollout.poses) == frames
    assert len(rollout.dead_reckoned) == frames
    assert rollout.local_obs.shape[0] == frames
    assert rollout.global_obs.shape[0] == frames
    assert rollout.alive.shape == (frames,)
    assert rollout.steps <= len(actions)


def test_grading_a_rollout_against_itself_is_perfect():
    rollout, _ = make_recorded()

    for reference in ('local', 'global'):
        matrix = grade(rollout, rollout.local_obs, reference=reference)
        assert matrix.errors == 0
        assert matrix.matrix.sum() > 0


def test_noise_does_not_change_the_grade():
    """Snapping must make a noisy recording score like a clean one."""
    clean, actions = make_recorded(noise=False, seed=4)
    env = make(num_snakes=3, noise=True, seed=4)
    noisy = record_rollout(env, actions[:clean.steps], agent=0)

    matrix = grade(clean, noisy.local_obs, reference='local')

    assert matrix.errors == 0


def test_grading_attributes_corruption_to_the_right_classes():
    rollout, _ = make_recorded()
    broken = rollout.local_obs.copy()
    broken[:, 0, 0] = cell_color(Cell.BODY.value, 5).astype(np.uint8)

    matrix = grade(rollout, broken, reference='local')

    assert matrix.errors > 0
    assert all(observed == 'body5'
               for _, observed, _ in matrix.top_confusions(3))


def test_dead_reckoning_flags_where_the_agent_stopped():
    """Kinematics keeps walking a snake the simulator has killed."""
    rollout, _ = make_recorded(steps=60, num_snakes=4, seed=9)

    if not rollout.alive[-1]:
        assert rollout.pose_drift(), 'death should show up as pose drift'
    else:
        assert rollout.pose_drift() == []


def test_saved_rollout_round_trips(tmp_path):
    rollout, _ = make_recorded()
    path = tmp_path / 'rollout.npz'
    save_rollout(path, rollout)

    data = np.load(path)
    assert np.array_equal(data['local_obs'], rollout.local_obs)
    assert np.array_equal(data['actions'], rollout.actions)
    assert data['poses'].shape == (rollout.steps + 1, 4)
