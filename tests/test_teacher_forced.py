"""Information boundaries and scoring contracts for the independent evaluator."""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch

import marlenv  # noqa: F401
from marlenv.core.palette import cell_color
from marlenv.flex_wm.model import FlexWorldModel
from marlenv.grading.teacher_forced import (
    CellCounts, MOVES, Prediction, Recording, Scores, collect_recording,
    evaluate_recording, overlap, predict_next, prediction_query,
)
from marlenv.wm.data import to_model_input


@pytest.fixture
def recording():
    env = gym.make('Snake-v1', height=15, width=15, num_snakes=3,
                   view_radius=4, background_gradient=0., observation_noise=2.,
                   snake_noise_sigma=8., noise_period=3, disable_env_checker=True)
    rng = np.random.default_rng(2)
    try:
        return collect_recording(env, lambda _: rng.integers(0, 3, 3),
                                 seed=12, steps=12)
    finally:
        env.close()


@pytest.fixture
def model():
    torch.manual_seed(42)
    return FlexWorldModel(schedule='FAG', dim=32, depth=3, heads=4,
                          num_actions=4, frame='world').eval()


def test_union_counts_both_directions_and_preserves_class_identity():
    left = np.array([3, 0, 0, 4, 13])
    right = np.array([3, 3, 3, 5, 3])
    forward, reverse = CellCounts(), CellCounts()
    forward.add(left, right)
    reverse.add(right, left)
    assert forward.report() == reverse.report()
    assert forward.snake_union == 5
    assert forward.snake_exact == 1
    assert forward.report()['snake_exact_rate'] == .2
    assert forward.report()['snake_occupancy_iou'] == .6


def test_empty_snake_union_is_undefined():
    score = CellCounts()
    score.add(np.array([0, 1]), np.array([0, 1]))
    assert score.report()['snake_exact_rate'] is None
    assert score.report()['all_exact_rate'] == 1.
    assert CellCounts().report()['all_exact_rate'] is None


def test_overlap_matches_world_cells_and_is_symmetric():
    board = np.arange(400).reshape(20, 20)
    left, right = board[2:11, 3:12], board[5:14, 7:16]
    a, b = overlap(left, right, [6, 7], [9, 11])
    assert np.array_equal(a, board[5:11, 7:12])
    assert np.array_equal(a, b)
    reverse = overlap(right, left, [9, 11], [6, 7])
    assert np.array_equal(reverse[0], b)
    assert not overlap(left, right, [0, 0], [50, 50])[0].size


def test_fatal_poses_are_actual_and_survive_round_trip(tmp_path):
    env = gym.make('Snake-v1', height=15, width=15, num_snakes=1,
                   view_radius=4, background_gradient=0., disable_env_checker=True)
    try:
        record = collect_recording(env, lambda _: [0], seed=0, steps=40)
        assert not record.alive[-1, 0]
        assert np.array_equal(record.positions[-1, 0], env.unwrapped.snakes[0].head_coord)
        assert np.array_equal(record.positions[-1, 0] - record.positions[-2, 0],
                              MOVES[record.actions[-1, 0]])
        path = tmp_path / 'recording.npz'
        record.save(path)
        loaded = Recording.load(path)
        for key in ('observations', 'positions', 'alive', 'actions', 'metadata'):
            assert np.array_equal(getattr(record, key), getattr(loaded, key))
    finally:
        env.close()


def test_bad_fatal_geometry_is_rejected(recording):
    positions = recording.positions.copy()
    positions[1, 0] = [-1, -1]
    with pytest.raises(ValueError, match='positions disagree'):
        Recording(recording.observations, positions, recording.alive, recording.actions)


def test_target_and_future_truth_never_reach_prediction(recording, model):
    target = 2
    original = predict_next(model, recording, target, window=5, denoise_steps=2, seed=17)
    changed = deepcopy(recording)
    # Deliberately invalid future data: the predictor must not read it at all.
    changed.observations[target:] ^= 255
    changed.positions[target:] += 1000
    changed.alive[target:] = ~changed.alive[target:]
    changed.actions[target:] = 3
    altered = predict_next(model, changed, target, window=5, denoise_steps=2, seed=17)
    assert np.array_equal(original.observations, altered.observations)
    assert np.array_equal(original.agents, altered.agents)
    assert np.array_equal(original.positions, altered.positions)


def test_prediction_is_stateless_and_does_not_mutate_recording(recording, model):
    before = deepcopy(recording)
    expected = predict_next(model, recording, 2, denoise_steps=2, seed=9)
    throwaway = predict_next(model, recording, 1, denoise_steps=2, seed=100)
    throwaway.observations[:] = 255
    actual = predict_next(model, recording, 2, denoise_steps=2, seed=9)
    assert np.array_equal(expected.observations, actual.observations)
    for key in ('observations', 'positions', 'alive', 'actions'):
        assert np.array_equal(getattr(recording, key), getattr(before, key))


def test_every_denoising_pass_receives_clean_real_history(recording, model):
    pairs, past, _, _ = prediction_query(recording, 2, 5, 'cpu')
    calls = []

    def inspect(module, arguments, kwargs):
        query, frames, actions, frame_tau, action_tau = arguments
        assert torch.equal(frames[:, :past], pairs.observations[:, :past])
        assert torch.all(frame_tau[:, :past] == 0)
        assert torch.all(action_tau[:, past:] == 1)
        assert torch.all(query.observations[:, past:] == 0)
        calls.append(frames[:, past:].clone())

    hook = model.register_forward_pre_hook(inspect, with_kwargs=True)
    try:
        predict_next(model, recording, 2, window=5, denoise_steps=3)
    finally:
        hook.remove()
    assert len(calls) == 3
    assert not torch.equal(calls[0], calls[-1])


def test_query_keeps_correct_agents_after_middle_death():
    # Agent 1 dies on transition 0; its aftermath remains historical truth,
    # but it cannot receive a target or action on transition 1.
    positions = np.zeros((4, 3, 2), np.int64)
    positions[0, :, 1] = [0, 3, 6]
    alive = np.ones((4, 3), bool)
    alive[1:, 1] = False
    actions = np.array([[0, 1, 2], [1, -1, 3], [2, -1, 0]])
    for t in range(3):
        positions[t + 1] = positions[t]
        positions[t + 1, alive[t]] += MOVES[actions[t, alive[t]]]
    record = Recording(np.zeros((4, 3, 9, 9, 3), np.uint8), positions, alive, actions)
    pairs, past, agents, targets = prediction_query(record, 2, 5, 'cpu')
    assert agents.tolist() == [0, 2]
    assert pairs.agent[0, past:].tolist() == [0, 2]
    assert np.array_equal(targets, positions[2, [0, 2]])
    last_real = pairs.time[0, :past] == 1
    assert pairs.actions[0, :past][last_real].tolist() == [1, 0, 3]
    assert pairs.acted[0, :past][last_real].tolist() == [True, False, True]
    _, _, fatal_targets, _ = prediction_query(record, 1, 5, 'cpu')
    assert fatal_targets.tolist() == [0, 1, 2]


def test_window_is_target_plus_real_history(recording):
    pairs, _, _, _ = prediction_query(recording, 5, 3, 'cpu')
    assert pairs.time.min() == 0 and pairs.time.max() == 2
    at_start = pairs.time[0] == 0
    agents = pairs.agent[0, at_start].numpy()
    expected = to_model_input(recording.observations[3, agents])
    assert np.array_equal(pairs.observations[0, at_start].numpy(), expected)


def test_truth_control_and_oracle_are_perfect_including_deaths(recording):
    score = Scores()
    for target in range(1, recording.steps + 1):
        agents = np.flatnonzero(recording.alive[target - 1])
        oracle = Prediction(agents, recording.positions[target, agents],
                             recording.observations[target, agents])
        score.add(recording, target, oracle)
    assert score.views == recording.alive[:-1].sum()
    assert score.death_views == (recording.alive[:-1] & ~recording.alive[1:]).sum()
    assert score.accuracy.report()['snake_exact_rate'] == 1.
    assert score.consistency.report()['snake_exact_rate'] == 1.
    assert score.truth_control.report()['all_exact_rate'] == 1.


def test_jointly_empty_predictions_cannot_hide_from_accuracy(recording):
    agents = np.flatnonzero(recording.alive[0])
    empty = np.empty_like(recording.observations[1, agents])
    empty[:] = cell_color(0).astype(np.uint8)
    score = Scores()
    score.add(recording, 1, Prediction(agents, recording.positions[1, agents], empty))
    assert score.consistency.report()['snake_exact_rate'] is None
    assert score.accuracy.report()['snake_exact_rate'] == 0.


def test_scoring_agent_order_does_not_change_result(recording):
    agents = np.flatnonzero(recording.alive[0])
    images = recording.observations[1, agents].copy()
    images[0] = cell_color(0).astype(np.uint8)
    left, right = Scores(), Scores()
    left.add(recording, 1, Prediction(agents, recording.positions[1, agents], images))
    right.add(recording, 1, Prediction(agents[::-1], recording.positions[1, agents[::-1]], images[::-1]))
    assert left.report() == right.report()


def test_evaluation_does_not_stop_when_predictions_lack_heads(recording, model, monkeypatch):
    import marlenv.grading.teacher_forced as instrument
    targets = []

    def empty_prediction(model, record, target, **kwargs):
        targets.append(target)
        agents = np.flatnonzero(record.alive[target - 1])
        frames = np.empty_like(record.observations[target, agents])
        frames[:] = cell_color(0).astype(np.uint8)
        return Prediction(agents, record.positions[target, agents], frames)

    monkeypatch.setattr(instrument, 'predict_next', empty_prediction)
    scores = evaluate_recording(model, recording)
    assert targets == list(range(1, recording.steps + 1))
    assert scores.views == recording.alive[:-1].sum()


def test_episode_aggregation_uses_counts_not_average_rates():
    left, right = Scores(), Scores()
    left.consistency.add(np.array([3]), np.array([3]))
    right.consistency.add(np.zeros(9, int), np.full(9, 3))
    left.merge(right)
    assert left.consistency.report()['snake_exact_rate'] == .1
