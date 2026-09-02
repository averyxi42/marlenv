"""Evaluators and the PUCT search, exercised without torch.

AlphaZeroSolver depends on an evaluator, not on a network, so the search is
importable and testable with no deep-learning stack present.
"""
import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.policies import RolloutEvaluator, UniformEvaluator
from marlenv.policies.alphazero import AlphaZeroSolver
from marlenv.policies.objectives import get_objective


@pytest.fixture
def env():
    return gym.make('Snake-v1', height=11, width=11, num_snakes=3,
                    num_fruits=3, disable_env_checker=True)


def test_uniform_evaluator_shapes_and_values(env):
    env.reset(seed=0)
    indices, priors, values = UniformEvaluator().evaluate(env)

    assert indices == [0, 1, 2]
    assert priors.shape == (3, 3)
    assert np.allclose(priors, 1 / 3)
    assert np.allclose(values, 0.0)


def test_evaluators_only_report_living_snakes(env):
    env.reset(seed=0)
    env.unwrapped.snakes[1].alive = False

    for evaluator in (UniformEvaluator(), RolloutEvaluator(seed=0)):
        indices, priors, values = evaluator.evaluate(env)
        assert indices == [0, 2]
        assert priors.shape == (2, 3)
        assert values.shape == (2,)


def test_rollout_evaluator_returns_per_snake_values(env):
    """Per-snake values are what let one evaluator serve every objective."""
    env.reset(seed=0)
    _, _, values = RolloutEvaluator(rollout_depth=8, seed=0).evaluate(env)

    assert values.shape == (3,)
    # the vector folds under any objective without the evaluator knowing it
    for name in ('sum', 'mean', 'min', 'max'):
        assert np.isfinite(get_objective(name).fold(list(values)))


def test_rollout_evaluator_does_not_mutate_the_env(env):
    env.reset(seed=0)
    before = env.unwrapped.grid.copy()
    RolloutEvaluator(rollout_depth=8, seed=0).evaluate(env)

    assert np.array_equal(env.unwrapped.grid, before)


@pytest.mark.parametrize('make_evaluator',
                         [UniformEvaluator,
                          lambda: RolloutEvaluator(rollout_depth=5, seed=0)])
def test_search_runs_with_any_evaluator(env, make_evaluator):
    env.reset(seed=0)
    solver = AlphaZeroSolver(make_evaluator(), objective='sum',
                             num_simulations=16, seed=0)

    action, target = solver.search(env)

    assert len(action) == 3
    assert np.allclose(target.sum(axis=1), 1.0)
    env.step(action)


def test_rollout_evaluator_beats_uniform_on_survival(env):
    """A value signal should outlive having none at all."""
    def play(evaluator, seed):
        solver = AlphaZeroSolver(evaluator, objective='sum',
                                 num_simulations=24, seed=0,
                                 exploration_fraction=0.0)
        env.reset(seed=seed)
        solver.reset()
        for step in range(40):
            _, _, term, trunc, _ = env.step(solver.solve(env))
            if all(term) or all(trunc):
                return step + 1
        return 40

    seeds = range(4)
    uniform = np.mean([play(UniformEvaluator(), s) for s in seeds])
    rollout = np.mean([play(RolloutEvaluator(rollout_depth=10, seed=0), s)
                       for s in seeds])

    assert rollout >= uniform
