"""AlphaZero-style solver for the snake envs.

Differences from :class:`~marlenv.policies.mcts.MCTSSolver`:

* leaves are evaluated by the network instead of a random rollout;
* children are ordered by a prior, and selection is PUCT rather than UCT;
* the search reports a policy target (visit counts) for training.

The prior over joint actions is the product of the per-snake policies. That
ignores the correlations a true joint policy would capture, which is
acceptable because the prior only steers PUCT -- the search itself still
evaluates genuine joint actions and can override the prior.

Values are factorised the same way: the network emits one value per living
snake, the communal objective folds them into the value the search backs up,
and dead snakes contribute a padded 0.
"""
import itertools
import math

import numpy as np

from marlenv.policies.mcts import _MinMax, _inert_obs
from marlenv.policies.objectives import get_objective


class _AZNode:
    """A searched state, with arrays indexed by joint-action slot."""

    __slots__ = ('key', 'env', 'terminal', 'value', 'agents',
                 'joint_actions', 'priors', 'rewards',
                 'visits', 'value_sum', 'children', 'total_visits')

    def __init__(self, key, env, terminal):
        self.key = key
        self.env = env
        self.terminal = terminal
        self.value = 0.0
        self.agents = []
        self.joint_actions = np.zeros((0, 0), dtype=np.int64)
        self.priors = np.zeros(0, dtype=np.float64)
        self.rewards = np.zeros(0, dtype=np.float64)
        self.visits = np.zeros(0, dtype=np.int64)
        self.value_sum = np.zeros(0, dtype=np.float64)
        self.children = []
        self.total_visits = 0

    @property
    def expanded(self):
        return len(self.children) > 0


class AlphaZeroSolver:
    """Network-guided joint-action search.

    Parameters
    ----------
    evaluator : NetworkEvaluator
        Supplies per-snake priors and values.
    objective : str or CommunalObjective
        How per-snake rewards and values fold into one number.
    num_simulations, c_puct, discount, max_depth
        Standard search settings.
    max_joint_actions : int or None
        Keep only this many joint actions per node, chosen by prior. The full
        space is ``num_actions ** num_living_snakes``.
    dirichlet_alpha, exploration_fraction : float
        Root exploration noise, mixed into the per-snake priors. Set
        ``exploration_fraction`` to 0 for evaluation.
    reuse_tree : bool
        Reuse the subtree matching the env's state across calls.
    """

    def __init__(
            self,
            evaluator,
            objective='sum',
            num_simulations=64,
            c_puct=1.5,
            discount=0.97,
            max_depth=24,
            max_joint_actions=32,
            dirichlet_alpha=0.6,
            exploration_fraction=0.25,
            reuse_tree=True,
            seed=None,
    ):
        self.evaluator = evaluator
        self.objective = get_objective(objective)
        self.communal_reward_fn = self.objective.fold
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.discount = discount
        self.max_depth = max_depth
        self.max_joint_actions = max_joint_actions
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.reuse_tree = reuse_tree
        self.np_random = np.random.default_rng(seed)

        self._root = None
        self._value_range = _MinMax()
        self.last_stats = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def reset(self):
        self._root = None
        self.last_stats = {}

    def extract_state_hash(self, env):
        """Hashable state key; see :meth:`MCTSSolver.extract_state_hash`."""
        base = env.unwrapped
        if getattr(base, 'snakes', None) is None:
            raise RuntimeError('env must be reset() before it can be solved')
        snakes = tuple(
            (s.idx, s.alive, s.direction.value, s.head_coord, s.tail_coord,
             tuple(d.value for d in s.directions))
            for s in base.snakes
        )
        return base.grid.tobytes(), snakes, base.alive_snakes

    def solve(self, env, add_noise=False):
        """Search and return the most visited joint action."""
        action, _ = self.search(env, add_noise=add_noise)
        return action

    def search(self, env, add_noise=False):
        """Search from ``env`` and return ``(joint_action, policy_target)``.

        ``policy_target`` is ``(num_snakes, num_actions)``: the root visit
        counts marginalised per snake and normalised, with dead snakes left
        as all-zero rows.
        """
        base = env.unwrapped
        key = self.extract_state_hash(env)
        root = self._reuse_root(key)
        if root is None:
            root = _AZNode(key, self._prepare(env), terminal=False)

        self._value_range = _MinMax()
        self._expand(root)
        if add_noise:
            # applied here rather than inside _expand so that a reused root,
            # which was expanded as somebody's child without noise, still
            # gets fresh exploration noise
            self._add_root_noise(root)
        for _ in range(self.num_simulations):
            self._simulate(root)

        self._root = root
        num_actions = self.evaluator.net.num_actions
        target = np.zeros((base.num_snakes, num_actions), dtype=np.float32)
        if not root.expanded or root.total_visits == 0:
            self.last_stats = {'root_visits': 0, 'num_children': 0}
            return [0] * base.num_snakes, target

        for slot, agent in enumerate(root.agents):
            counts = np.zeros(num_actions, dtype=np.float64)
            for j, joint in enumerate(root.joint_actions):
                counts[joint[slot]] += root.visits[j]
            if counts.sum() > 0:
                target[agent] = counts / counts.sum()

        best = int(np.argmax(root.visits))
        action = self._to_full_action(root, root.joint_actions[best])
        self.last_stats = {
            'root_visits': int(root.total_visits),
            'num_children': len(root.children),
            'chosen_visits': int(root.visits[best]),
            'chosen_value': float(root.value_sum[best]
                                  / max(root.visits[best], 1)),
            'root_value': root.value,
        }
        return action, target

    # ------------------------------------------------------------------
    # tree construction
    # ------------------------------------------------------------------
    def _prepare(self, env):
        """A simulation clone with observation encoding stubbed out."""
        import copy
        base = env.unwrapped
        frames = getattr(base, 'frame_buffer', None)
        if frames:
            base.frame_buffer = []
        try:
            sim = copy.deepcopy(base)
        finally:
            if frames:
                base.frame_buffer = frames
        sim._get_obs = _inert_obs
        sim._init_obs = _inert_obs
        obs = getattr(sim, 'obs', None)
        if obs is not None:
            obs.clear()
        # spaces are never read during search and are costly to copy
        sim.observation_space = None
        sim.action_space = None
        return sim

    def _clone(self, sim):
        import copy
        return copy.deepcopy(sim)

    def _reuse_root(self, key):
        if not self.reuse_tree or self._root is None:
            return None
        if self._root.key == key:
            return self._root
        for child in self._root.children:
            if child is not None and child.key == key:
                return child
        return None

    def _to_full_action(self, node, joint):
        """Map a joint action over living snakes to one per snake."""
        full = [0] * node.env.num_snakes
        for slot, agent in enumerate(node.agents):
            full[agent] = int(joint[slot])
        return full

    def _add_root_noise(self, node):
        """Mix Dirichlet noise into the root's joint priors."""
        if not node.expanded or self.exploration_fraction <= 0:
            return
        noise = self.np_random.dirichlet(
            [self.dirichlet_alpha] * len(node.priors))
        node.priors = ((1 - self.exploration_fraction) * node.priors
                       + self.exploration_fraction * noise)

    def _expand(self, node):
        """Evaluate ``node`` with the network and lay out its children."""
        if node.expanded or node.terminal:
            return node.value

        agents, priors, values = self.evaluator.evaluate(node.env)
        if not agents:
            node.terminal = True
            node.value = 0.0
            return 0.0

        # dead snakes are retired: they are absent from `agents`, so they
        # contribute neither a value nor an action dimension
        node.value = float(self.objective.fold(list(values)))
        node.agents = list(agents)

        priors = np.asarray(priors, dtype=np.float64)
        joints, joint_priors = self._joint_priors(priors)
        node.joint_actions = joints
        node.priors = joint_priors
        node.rewards = np.zeros(len(joints))
        node.visits = np.zeros(len(joints), dtype=np.int64)
        node.value_sum = np.zeros(len(joints))
        node.children = [None] * len(joints)
        return node.value

    def _joint_priors(self, priors):
        """Joint actions and their factorised prior, capped by prior mass.

        The prior of a joint action is the product of the per-snake
        probabilities, which is exactly the independence assumption the
        docstring warns about; it is only used to order PUCT.
        """
        num_agents, num_actions = priors.shape
        combos = np.array(
            list(itertools.product(range(num_actions), repeat=num_agents)),
            dtype=np.int64)
        # product of per-agent priors, in log space for stability
        logp = np.log(np.clip(priors, 1e-12, None))
        joint_logp = sum(logp[i, combos[:, i]] for i in range(num_agents))

        cap = self.max_joint_actions
        if cap is not None and len(combos) > cap:
            keep = np.argpartition(joint_logp, -cap)[-cap:]
            combos, joint_logp = combos[keep], joint_logp[keep]

        weights = np.exp(joint_logp - joint_logp.max())
        return combos, weights / weights.sum()

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def _simulate(self, root):
        path = []
        node = root
        depth = 0

        while True:
            if node.terminal:
                value = 0.0
                break
            if depth >= self.max_depth:
                value = node.value
                break

            slot = self._puct_select(node)
            path.append((node, slot))
            child = node.children[slot]

            if child is None:
                child = self._step_child(node, slot)
                node.children[slot] = child
                value = self._expand(child)
                break

            node = child
            depth += 1

        self._backup(path, value)

    def _step_child(self, node, slot):
        """Simulate one joint action, creating the child node."""
        sim = self._clone(node.env)
        action = self._to_full_action(node, node.joint_actions[slot])
        _, rews, terminated, truncated, _ = sim.step(action)
        node.rewards[slot] = float(self.objective.fold(rews))
        terminal = all(terminated) or all(truncated)
        return _AZNode(self.extract_state_hash(sim), sim, terminal)

    def _puct_select(self, node):
        visits = node.visits
        total = max(node.total_visits, 1)
        q = np.where(visits > 0,
                     node.value_sum / np.maximum(visits, 1),
                     node.value)
        q = np.array([self._value_range.normalize(v) for v in q])
        u = self.c_puct * node.priors * math.sqrt(total) / (1 + visits)
        return int(np.argmax(q + u))

    def _backup(self, path, value):
        ret = value
        for node, slot in reversed(path):
            ret = node.rewards[slot] + self.discount * ret
            node.visits[slot] += 1
            node.total_visits += 1
            node.value_sum[slot] += ret
            self._value_range.update(node.value_sum[slot] / node.visits[slot])
