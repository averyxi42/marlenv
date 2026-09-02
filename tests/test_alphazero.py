import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.policies.features import (CHANNELS, NUM_CHANNELS,
                                       head_positions, observe)
from marlenv.policies.objectives import get_objective

torch = pytest.importorskip('torch')

from marlenv.policies.alphazero import AlphaZeroSolver  # noqa: E402
from marlenv.policies.networks import (NetworkEvaluator,  # noqa: E402
                                       SnakeNet)
from marlenv.policies.training import (ReplayBuffer,  # noqa: E402
                                       compute_losses, self_play_episode,
                                       train_step)


@pytest.fixture
def env():
    return gym.make('Snake-v1', height=11, width=11, num_snakes=3,
                    num_fruits=3, disable_env_checker=True)


@pytest.fixture
def evaluator():
    return NetworkEvaluator(SnakeNet(channels=8, blocks=1, hidden=16),
                            device='cpu')


@pytest.fixture
def solver(evaluator):
    return AlphaZeroSolver(evaluator, objective='sum', num_simulations=16,
                           seed=0)


# ----------------------------------------------------------------- features
def test_views_are_one_per_living_snake(env):
    env.reset(seed=0)
    planes, indices = observe(env)

    assert planes.shape == (3, NUM_CHANNELS, 11, 11)
    assert indices == [0, 1, 2]


def test_view_never_encodes_snake_identity(env):
    """Every snake's view must be built from the same channel semantics."""
    env.reset(seed=0)
    planes, _ = observe(env)
    mine = [CHANNELS.index(c) for c in ('my_head', 'my_body', 'my_tail')]
    theirs = [CHANNELS.index(c) for c in
              ('other_head', 'other_body', 'other_tail')]

    for view in planes:
        # each snake sees exactly its own cells as "mine" and the rest as
        # "theirs", so the totals are identical across snakes
        assert view[mine].sum() == planes[0][mine].sum()
        assert view[theirs].sum() == planes[0][theirs].sum()


def test_heading_is_canonicalised_to_up(env):
    """After rotation the neck sits directly below the head, always."""
    env.reset(seed=0)
    base = env.unwrapped
    body = CHANNELS.index('my_body')
    tail = CHANNELS.index('my_tail')
    seen = set()

    for _ in range(40):
        for i, snake in enumerate(base.snakes):
            if not snake.alive:
                continue
            planes, _ = observe(env, [i])
            row, col = head_positions(planes)[0]
            neck = max(planes[0][body][row + 1, col],
                       planes[0][tail][row + 1, col])
            assert neck == 1.0
            seen.add(snake.direction)
        _, _, term, trunc, _ = env.step(list(env.action_space.sample()))
        if all(term) or all(trunc):
            break

    assert len(seen) > 1, 'test never exercised more than one heading'


def test_non_square_grid_is_rejected():
    env = gym.make('Snake-v1', height=10, width=12, num_snakes=1,
                   disable_env_checker=True)
    env.reset(seed=0)

    with pytest.raises(ValueError, match='square'):
        observe(env)


# --------------------------------------------------------------- objectives
def test_torch_folds_ignore_dead_agents():
    values = torch.tensor([[1.0, 5.0, 0.0]])
    alive = torch.tensor([[1.0, 1.0, 0.0]])

    assert get_objective('sum').torch_fold(values, alive).item() == 6.0
    assert get_objective('min').torch_fold(values, alive).item() == 1.0
    assert get_objective('mean').torch_fold(values, alive).item() == 3.0


# ------------------------------------------------------------------ network
def test_network_is_cardinality_flexible(evaluator):
    """One set of weights must serve any number of snakes."""
    for num_snakes in (1, 2, 5):
        env = gym.make('Snake-v1', height=11, width=11,
                       num_snakes=num_snakes, disable_env_checker=True)
        env.reset(seed=0)
        indices, priors, values = evaluator.evaluate(env)

        assert len(indices) == num_snakes
        assert priors.shape == (num_snakes, 3)
        assert values.shape == (num_snakes,)


def test_network_is_identity_equivariant(evaluator, env):
    """Swapping two snakes' indices must swap their outputs, nothing else."""
    env.reset(seed=0)
    base = env.unwrapped
    _, priors, values = evaluator.evaluate(env)

    base.snakes[0], base.snakes[1] = base.snakes[1], base.snakes[0]
    base.snakes[0].idx, base.snakes[1].idx = 0, 1
    base.grid = np.where(
        base.grid // 10 == 0, base.grid + 10,
        np.where(base.grid // 10 == 1, base.grid - 10, base.grid))
    _, swapped_priors, swapped_values = evaluator.evaluate(env)

    assert np.allclose(priors[0], swapped_priors[1], atol=1e-5)
    assert np.allclose(priors[1], swapped_priors[0], atol=1e-5)
    assert np.allclose(values[[0, 1]], swapped_values[[1, 0]], atol=1e-5)


# ------------------------------------------------------------------- search
def test_search_returns_action_and_policy_target(env, solver):
    env.reset(seed=0)
    action, target = solver.search(env)

    assert len(action) == 3
    assert all(a in env.unwrapped.action_dict for a in action)
    assert target.shape == (3, 3)
    assert np.allclose(target.sum(axis=1), 1.0)
    env.step(action)


def test_dead_snakes_are_retired_from_the_search(env, solver):
    env.reset(seed=0)
    base = env.unwrapped
    base.snakes[0].alive = False
    base.alive_snakes = 2

    action, target = solver.search(env)

    # the dead snake gets no action dimension and an all-zero policy row
    assert np.allclose(target[0], 0.0)
    assert np.allclose(target[1:].sum(axis=1), 1.0)
    assert len(action) == 3


def test_search_does_not_mutate_the_env(env, solver):
    env.reset(seed=0)
    before = solver.extract_state_hash(env)
    solver.search(env)

    assert solver.extract_state_hash(env) == before


def test_joint_priors_are_capped(evaluator, env):
    env.reset(seed=0)
    solver = AlphaZeroSolver(evaluator, num_simulations=4,
                             max_joint_actions=5, seed=0)
    solver.search(env)

    assert len(solver._root.children) <= 5


# ----------------------------------------------------------------- training
def test_value_gradient_reaches_every_living_snake(env, evaluator):
    """The loss only sees the communal value, but must train all heads."""
    env.reset(seed=0)
    planes, _ = observe(env, list(range(3)))
    alive = np.ones((1, 3), dtype=np.float32)
    batch = ReplayBuffer(seed=0)
    batch.add(planes, alive[0], np.full((3, 3), 1 / 3, dtype=np.float32), 2.0)
    sample = batch.sample(1, 'cpu')

    value_loss, _ = compute_losses(evaluator.net, sample, 'sum')
    value_loss.backward()

    grads = [p.grad for p in evaluator.net.value.parameters()]
    assert all(g is not None and torch.any(g != 0) for g in grads)


def test_training_reduces_loss_on_a_fixed_batch(env, evaluator):
    env.reset(seed=0)
    buffer = ReplayBuffer(seed=0)
    for step in range(8):
        planes, _ = observe(env, list(range(3)))
        alive = np.array([s.alive for s in env.unwrapped.snakes],
                         dtype=np.float32)
        buffer.add(planes, alive,
                   np.full((3, 3), 1 / 3, dtype=np.float32), 1.0)
        _, _, term, trunc, _ = env.step([0, 0, 0])
        if all(term) or all(trunc):
            break

    optimizer = torch.optim.Adam(evaluator.net.parameters(), lr=1e-2)
    batch = buffer.sample(8, 'cpu')
    first = train_step(evaluator.net, optimizer, batch, 'sum')['loss']
    for _ in range(30):
        last = train_step(evaluator.net, optimizer, batch, 'sum')['loss']

    assert last < first


def test_self_play_produces_aligned_positions(env, solver):
    env.reset(seed=0)
    positions, stats = self_play_episode(env, solver, 'sum', max_steps=5,
                                         rng=np.random.default_rng(0))

    assert len(positions) == stats['steps']
    for views, alive, policy, value in positions:
        assert views.shape == (3, NUM_CHANNELS, 11, 11)
        assert alive.shape == (3,)
        assert policy.shape == (3, 3)
        assert np.isfinite(value)
