import gymnasium as gym
import marlenv
import pytest

from marlenv.policies import MCTSSolver


@pytest.fixture
def env():
    return gym.make('Snake-v1', num_fruits=4, num_snakes=2,
                    disable_env_checker=True)


@pytest.fixture
def solver():
    return MCTSSolver(communal_reward_fn=sum, num_simulations=30,
                      max_depth=4, rollout_depth=5, seed=0)


def test_solve_returns_a_steppable_joint_action(env, solver):
    env.reset(seed=0)
    action = solver.solve(env)

    assert len(action) == env.unwrapped.num_snakes
    assert all(a in env.unwrapped.action_dict for a in action)
    env.step(action)


def test_solve_does_not_mutate_the_env(env, solver):
    env.reset(seed=0)
    before = solver.extract_state_hash(env)
    solver.solve(env)

    assert solver.extract_state_hash(env) == before


def test_state_hash_tracks_state_changes(env, solver):
    env.reset(seed=0)
    before = solver.extract_state_hash(env)
    env.step([0, 0])

    assert solver.extract_state_hash(env) != before


def test_subtree_is_reused_across_solve_calls(env, solver):
    env.reset(seed=0)
    env.step(solver.solve(env))
    carried = max(edge.visits for edge in solver._root.children.values())

    solver.solve(env)

    # the new root inherited the statistics of the chosen child instead of
    # starting from zero visits
    assert solver._root.visits > solver.num_simulations
    assert carried > 0


def test_reset_drops_the_cached_tree(env, solver):
    env.reset(seed=0)
    solver.solve(env)
    solver.reset()

    assert solver._root is None
    assert solver._transpositions == {}


def test_communal_reward_fn_receives_one_reward_per_snake(env):
    seen = []

    def communal(rewards):
        seen.append(len(rewards))
        return sum(rewards)

    solver = MCTSSolver(communal, num_simulations=10, max_depth=3,
                        rollout_depth=3, seed=0)
    env.reset(seed=0)
    solver.solve(env)

    assert seen
    assert set(seen) == {env.unwrapped.num_snakes}


def test_dead_snakes_are_not_enumerated(env, solver):
    env.reset(seed=0)
    base = env.unwrapped
    base.snakes[0].alive = False

    actions = solver._joint_actions(env)

    assert len(actions) == len(base.action_dict)
    assert all(action[0] == 0 for action in actions)


def test_max_joint_actions_caps_the_branching_factor():
    env = gym.make('Snake-v1', num_snakes=4, disable_env_checker=True)
    env.reset(seed=0)
    solver = MCTSSolver(sum, max_joint_actions=5, seed=0)

    assert len(solver._joint_actions(env)) <= 5


def test_solver_is_deterministic_given_a_seed(env):
    actions = []
    for _ in range(2):
        env.reset(seed=0)
        solver = MCTSSolver(sum, num_simulations=30, max_depth=4,
                            rollout_depth=5, seed=7)
        actions.append(solver.solve(env))

    assert actions[0] == actions[1]


def test_solve_requires_a_reset_env(solver):
    fresh = gym.make('Snake-v1', num_snakes=2, disable_env_checker=True)

    with pytest.raises(RuntimeError, match='reset'):
        solver.solve(fresh)


def test_solver_drives_every_snake_to_a_longer_episode(env):
    """The search should at least outlive a fixed do-nothing policy."""
    solver = MCTSSolver(sum, num_simulations=60, max_depth=6,
                        rollout_depth=8, seed=0)

    def run(policy):
        env.reset(seed=3)
        for step in range(40):
            action = policy(step)
            _, _, term, trunc, _ = env.step(action)
            if all(term) or all(trunc):
                return step + 1
        return 40

    noop = run(lambda _: [0, 0])
    searched = run(lambda _: solver.solve(env))

    assert searched > noop


def test_inert_observations_do_not_change_the_search(env):
    """The stubbed-out encoding must be invisible to the chosen actions."""
    sequences = []
    for inert in (False, True):
        env.reset(seed=1)
        solver = MCTSSolver(sum, num_simulations=40, max_depth=5,
                            rollout_depth=6, seed=0,
                            inert_observations=inert)
        actions = []
        for _ in range(8):
            action = solver.solve(env)
            actions.append(action)
            _, _, term, trunc, _ = env.step(action)
            if all(term) or all(trunc):
                break
        sequences.append(actions)

    assert sequences[0] == sequences[1]


def test_inert_clone_skips_observation_work(env, solver):
    env.reset(seed=0)
    solver.solve(env)
    sim = solver._root.env

    assert sim.observation_space is None
    assert sim._get_obs() == []
    # the real env is untouched
    assert env.unwrapped.observation_space is not None
