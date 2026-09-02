"""Prior and value modules for :class:`AlphaZeroSolver`.

The search only asks its evaluator one thing::

    evaluate(env) -> (indices, priors, values)

``indices`` lists the living snakes, ``priors`` is ``(n, num_actions)`` and
``values`` is ``(n,)`` -- one value *per snake*, not a communal one, because
folding them is the objective's job. Returning a per-snake vector rather than
a single number is what lets every evaluator work with every objective.

Swapping the evaluator changes what guides the search without touching it:

============== ================= ==========================================
evaluator      prior             value
============== ================= ==========================================
Uniform        uniform           zero (search is driven by rewards alone)
Rollout        uniform           random rollout, per snake
Network        learned policy    learned value head
============== ================= ==========================================
"""
import copy

import numpy as np


class UniformEvaluator:
    """Uniform prior and zero value: PUCT with no guidance at all.

    Useful as a control -- any benefit from a learned or rollout evaluator
    should show up as an improvement over this.
    """

    def __init__(self, num_actions=3):
        self.num_actions = num_actions

    def _living(self, env):
        base = env.unwrapped
        return [i for i, snake in enumerate(base.snakes) if snake.alive]

    def evaluate(self, env):
        indices = self._living(env)
        n = len(indices)
        priors = np.full((n, self.num_actions), 1.0 / self.num_actions,
                         dtype=np.float32)
        return indices, priors, np.zeros(n, dtype=np.float32)


class RolloutEvaluator(UniformEvaluator):
    """Uniform prior, value from a random rollout.

    The rollout accumulates each snake's *own* discounted reward, so the
    returned vector folds into the communal value under any objective -- no
    assumption that the communal function is a sum.
    """

    def __init__(self, num_actions=3, rollout_depth=10, discount=0.97,
                 seed=None):
        super().__init__(num_actions)
        self.rollout_depth = rollout_depth
        self.discount = discount
        self.np_random = np.random.default_rng(seed)

    def evaluate(self, env):
        indices = self._living(env)
        n = len(indices)
        priors = np.full((n, self.num_actions), 1.0 / self.num_actions,
                         dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        if n == 0 or self.rollout_depth <= 0:
            return indices, priors, values

        sim = copy.deepcopy(env.unwrapped)
        discount = 1.0
        for _ in range(self.rollout_depth):
            action = self._random_action(sim)
            if action is None:
                break
            _, rews, terminated, truncated, _ = sim.step(action)
            for slot, agent in enumerate(indices):
                values[slot] += discount * float(rews[agent])
            discount *= self.discount
            if all(terminated) or all(truncated):
                break
        return indices, priors, values

    def _random_action(self, env):
        base = env.unwrapped
        action = [0] * base.num_snakes
        alive = False
        for i, snake in enumerate(base.snakes):
            if snake.alive:
                action[i] = int(self.np_random.integers(self.num_actions))
                alive = True
        return action if alive else None
