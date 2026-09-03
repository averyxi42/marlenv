"""Episode collection: state grids, schema, and round-tripping."""
import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.snake import Direction
from marlenv.data import (build_dataset, collect_episode, decode_episode,
                          heading_of, poses_from_state, random_policy,
                          snake_bodies, state_grids)
from marlenv.data.state import NO_BODY, SNAKE_BASE, EMPTY, FRUIT, WALL
from marlenv.grading.poses import Pose, step_pose

HEADINGS = list(Direction)


def make(num_snakes=3, side=13, radius=4, noise=True, seed=0):
    env = gym.make('Snake-v1', height=side, width=side,
                   num_snakes=num_snakes, num_fruits=4, view_radius=radius,
                   observation_noise=2.0 if noise else 0.0,
                   snake_noise_sigma=8.0 if noise else 0.0,
                   background_gradient=16.0 if noise else 0.0,
                   disable_env_checker=True)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ------------------------------------------------------------ state grids
def test_state_grids_reconstruct_every_body_exactly():
    """The two grids are the whole state: bodies, order and heading."""
    env = make(num_snakes=4)
    base = env.unwrapped

    for _ in range(15):
        content, body_index = state_grids(env)
        bodies = snake_bodies(content, body_index)
        for snake in base.snakes:
            if not snake.alive:
                assert snake.idx not in bodies
                continue
            assert bodies[snake.idx] == [tuple(c) for c in snake.coords]
            assert heading_of(bodies[snake.idx]) == snake.direction
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        if all(term) or all(trunc):
            break


def test_content_codes_are_unbounded_in_agent_id():
    env = make(num_snakes=8, side=17)
    content, _ = state_grids(env)

    ids = sorted(int(v) - SNAKE_BASE for v in np.unique(content)
                 if v >= SNAKE_BASE)
    assert ids == list(range(8))


def test_static_cells_use_the_documented_codes():
    env = make(num_snakes=1)
    base = env.unwrapped
    content, body_index = state_grids(env)

    assert (content[0] == WALL).all()
    assert FRUIT in content
    assert EMPTY in content
    assert (body_index[content < SNAKE_BASE] == NO_BODY).all()


def test_poses_from_state_matches_the_simulator():
    env = make(num_snakes=3)
    base = env.unwrapped
    content, body_index = state_grids(env)

    poses = poses_from_state(content, body_index, 3)
    for snake in base.snakes:
        assert poses[snake.idx][0] == snake.head_coord[0]
        assert poses[snake.idx][1] == snake.head_coord[1]
        assert HEADINGS[poses[snake.idx][2]] == snake.direction


# --------------------------------------------------------------- collection
def collected(num_snakes=3, episodes=3, steps=30, seed=0):
    env = make(num_snakes=num_snakes, seed=seed)
    rng = np.random.default_rng(seed)
    return [collect_episode(env, random_policy(rng), seed=s, max_steps=steps)
            for s in range(episodes)]


def test_every_column_shares_the_frame_axis():
    row = collected(episodes=1)[0]
    frames = row['steps'] + 1

    for key in ('observations', 'ego_actions', 'cardinal_actions',
                'alive_mask', 'poses', 'rewards', 'content', 'body_index',
                'transition_mask'):
        assert row[key].shape[0] == frames, key


def test_terminal_frame_is_padded_and_masked():
    """The last row has no outgoing transition and must be flagged."""
    row = collected(episodes=1)[0]

    assert row['transition_mask'][:-1].all()
    assert not row['transition_mask'][-1]
    assert (row['ego_actions'][-1] == 0).all()
    assert (row['rewards'][-1] == 0).all()


def test_ego_actions_explain_the_pose_sequence():
    """Applying the stored action to a pose must give the next pose."""
    row = collected(num_snakes=2, episodes=1)[0]
    checked = 0

    for step in range(row['steps']):
        for agent in range(row['num_agents']):
            if not (row['alive_mask'][step, agent]
                    and row['alive_mask'][step + 1, agent]):
                continue
            here = row['poses'][step, agent]
            action = int(np.argmax(row['ego_actions'][step, agent]))
            expected = step_pose(
                Pose(int(here[0]), int(here[1]), HEADINGS[here[2]]), action)
            nxt = row['poses'][step + 1, agent]
            assert (expected.row, expected.col) == (nxt[0], nxt[1])
            assert HEADINGS[nxt[2]] == expected.direction
            checked += 1

    assert checked > 5


def test_cardinal_actions_are_the_resulting_heading():
    row = collected(num_snakes=2, episodes=1)[0]

    for step in range(row['steps']):
        for agent in range(row['num_agents']):
            if not row['alive_mask'][step + 1, agent]:
                continue
            cardinal = int(np.argmax(row['cardinal_actions'][step, agent]))
            assert cardinal == row['poses'][step + 1, agent, 2]


# ------------------------------------------------------------------ dataset
def test_dataset_round_trips_shapes_and_dtypes():
    dataset = build_dataset(collected())
    row = decode_episode(dataset[0])

    assert row['observations'].dtype == np.uint8
    assert row['content'].dtype == np.int16
    assert row['rewards'].dtype == np.float32
    assert row['alive_mask'].dtype == np.bool_
    frames = row['steps'] + 1
    view = 2 * row['view_radius'] + 1
    assert row['observations'].shape == (frames, row['num_agents'],
                                         view, view, 3)


def test_dataset_values_survive_the_round_trip():
    rows = collected(episodes=2)
    dataset = build_dataset(rows)

    for original, stored in zip(rows, dataset):
        decoded = decode_episode(stored)
        for key in ('observations', 'content', 'body_index', 'poses',
                    'rewards', 'alive_mask'):
            assert np.array_equal(decoded[key], original[key]), key


def test_flat_storage_stays_compact():
    """Guards against ArrayXD, which nests lists at 2.5x the raw bytes."""
    rows = collected(episodes=6, steps=40)
    dataset = build_dataset(rows)

    raw = sum(row['observations'].nbytes for row in rows)
    stored = dataset.data.column('observations').nbytes
    assert stored < 1.3 * raw, f'{stored / raw:.2f}x raw bytes'


def test_mismatched_episode_shapes_are_rejected():
    small = collected(num_snakes=2, episodes=1)
    large = collected(num_snakes=3, episodes=1)

    with pytest.raises(ValueError, match='num_agents'):
        build_dataset(small + large)


# ----------------------------------------------------------------- parallel
def small_config(**kwargs):
    from marlenv.data import CollectConfig
    defaults = dict(height=11, width=11, num_snakes=2, num_fruits=3,
                    view_radius=3, max_steps=12)
    defaults.update(kwargs)
    return CollectConfig(**defaults)


def test_parallel_collection_covers_every_seed(tmp_path):
    from marlenv.data import collect_dataset

    dataset = collect_dataset(small_config(), num_episodes=6,
                              out_dir=str(tmp_path / 'shards'), workers=3)

    assert len(dataset) == 6
    assert sorted(dataset['seed']) == list(range(6))


def test_sharding_does_not_change_the_episodes_a_seed_produces(tmp_path):
    """Board layout comes from the seed, so it must survive sharding."""
    from marlenv.data import collect_dataset

    one = collect_dataset(small_config(), 4, str(tmp_path / 'a'), workers=1)
    many = collect_dataset(small_config(), 4, str(tmp_path / 'b'), workers=2)

    for seed in range(4):
        first = decode_episode(one[one['seed'].index(seed)])
        second = decode_episode(many[many['seed'].index(seed)])
        # the initial state is fixed by the seed; later frames depend on the
        # policy, whose stream is per shard
        assert np.array_equal(first['content'][0], second['content'][0])


def test_epsilon_injects_exploration():
    from marlenv.data import make_policy

    rng = np.random.default_rng(0)
    env = small_config().make_env()
    env.reset(seed=0)
    greedy = make_policy(small_config(epsilon=0.0), rng)
    noisy = make_policy(small_config(epsilon=1.0), rng)

    assert len(list(greedy(env))) == 2
    assert all(0 <= a < 3 for a in noisy(env))
