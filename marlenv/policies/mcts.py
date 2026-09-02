"""Search based joint-action solver for the snake envs.

The solver drives *every* snake at once: it searches over joint actions and
scores a step by folding the per-snake reward vector into a single number
with a user supplied communal reward function.
"""
import copy
import itertools
import math

import numpy as np


def _inert_obs():
    """Stand-in for SnakeEnv._get_obs / _init_obs on simulation clones.

    ``step`` builds the full per-snake feature encoding on every call, which
    costs more than the rest of the transition put together. The search only
    ever reads rewards and done flags, so simulation clones skip it.
    """
    return []


class _MinMax:
    """Running value range, used to normalise Q values before UCT.

    The communal reward function is arbitrary, so raw Q values live on an
    unknown scale and a fixed exploration constant would be meaningless.
    """

    def __init__(self):
        self.low = float('inf')
        self.high = -float('inf')

    def update(self, value):
        self.low = min(self.low, value)
        self.high = max(self.high, value)

    def normalize(self, value):
        if self.high > self.low:
            return (value - self.low) / (self.high - self.low)
        return value


class _Edge:
    """A joint action taken from a node, with its own visit statistics."""

    __slots__ = ('child', 'reward', 'visits', 'value_sum')

    def __init__(self, child, reward):
        self.child = child
        self.reward = reward
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class _Node:
    """A search state: a cloned env plus the joint actions tried from it."""

    __slots__ = ('key', 'env', 'children', 'untried', 'visits', 'terminal')

    def __init__(self, key, env, untried, terminal=False):
        self.key = key
        self.env = env
        self.children = {}
        self.untried = untried
        self.visits = 0
        self.terminal = terminal


class MCTSSolver:
    """Monte Carlo tree search over the joint action space of all snakes.

    Two forms of dynamic programming are used:

    * a transposition table keyed by :meth:`extract_state_hash`, so a state
      reached by several different action orderings shares one node and one
      set of statistics;
    * subtree reuse across calls -- after :meth:`solve` returns an action and
      the caller steps the real env, the next call re-roots the tree at the
      matching child instead of searching from scratch.

    Parameters
    ----------
    communal_reward_fn : callable
        Maps the per-snake reward sequence returned by ``env.step`` to a
        single float, e.g. ``sum``, ``min`` or ``np.mean``.
    num_simulations : int
        Simulations run per :meth:`solve` call.
    exploration_weight : float
        UCT exploration constant, applied to normalised Q values.
    max_depth : int
        Maximum tree depth descended before falling back to a rollout.
    rollout_depth : int
        Steps taken by the random rollout at a leaf.
    discount : float
        Discount applied to future communal reward.
    max_joint_actions : int or None
        Cap on joint actions considered per node. ``None`` enumerates the
        full product, which is ``n_actions ** n_alive_snakes``.
    reuse_tree : bool
        Whether to reuse the subtree across :meth:`solve` calls.
    inert_observations : bool
        Strip observation encoding and the gym spaces from simulation clones.
        The search reads only rewards and done flags, so this is a pure
        saving -- it leaves the chosen action unchanged and makes the search
        several times faster. Turn it off if a subclass needs real
        observations during search.
    seed : int or None
        Seed for the solver's own rollout/sampling RNG.
    """

    def __init__(
            self,
            communal_reward_fn,
            num_simulations=100,
            exploration_weight=1.4,
            max_depth=10,
            rollout_depth=10,
            discount=0.95,
            max_joint_actions=None,
            reuse_tree=True,
            inert_observations=True,
            seed=None,
    ):
        if not callable(communal_reward_fn):
            raise TypeError('communal_reward_fn must be callable')
        self.communal_reward_fn = communal_reward_fn
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth
        self.discount = discount
        self.max_joint_actions = max_joint_actions
        self.reuse_tree = reuse_tree
        self.inert_observations = inert_observations
        self.np_random = np.random.default_rng(seed)

        self._root = None
        self._transpositions = {}
        self._value_range = _MinMax()
        # populated by the most recent solve(), for inspection
        self.last_stats = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def reset(self):
        """Drop the cached tree. Call this between episodes."""
        self._root = None
        self._transpositions = {}
        self.last_stats = {}

    def extract_state_hash(self, env):
        """Return a hashable representation of the env's dynamic state.

        The grid encodes walls, fruits and every snake's body, so it only
        needs to be paired with the per-snake heading and liveness, which the
        grid does not record but which determine the effect of an action.
        Episode counters are deliberately excluded so that a state reached at
        different depths maps to a single node.
        """
        base = env.unwrapped
        if getattr(base, 'snakes', None) is None:
            raise RuntimeError('env must be reset() before it can be solved')
        snakes = tuple(
            (snake.idx,
             snake.alive,
             snake.direction.value,
             snake.head_coord,
             snake.tail_coord,
             tuple(d.value for d in snake.directions))
            for snake in base.snakes
        )
        return base.grid.tobytes(), snakes, base.alive_snakes

    def solve(self, env):
        """Search from ``env``'s current state and return a joint action.

        The returned list holds one action per snake and can be passed
        straight to ``env.step``. ``env`` itself is never mutated.
        """
        base = env.unwrapped
        key = self.extract_state_hash(env)
        root = self._reuse_root(key)
        if root is None:
            self._transpositions = {}
            sim = self._clone(env)
            if self.inert_observations:
                self._make_inert(sim)
            root = self._make_node(key, sim, terminal=False)

        self._value_range = _MinMax()
        for _ in range(self.num_simulations):
            self._simulate(root)

        self._root = root
        if not root.children:
            # nothing to search: every snake is already dead
            self.last_stats = {}
            return [0] * base.num_snakes

        # robust child: most visited, ties broken on mean value
        action, edge = max(root.children.items(),
                           key=lambda kv: (kv[1].visits, kv[1].value))
        self.last_stats = {
            'root_visits': root.visits,
            'num_children': len(root.children),
            'chosen_visits': edge.visits,
            'chosen_value': edge.value,
        }
        return list(action)

    # ------------------------------------------------------------------
    # tree construction
    # ------------------------------------------------------------------
    def _make_node(self, key, env, terminal):
        untried = [] if terminal else self._joint_actions(env)
        node = _Node(key, env, untried, terminal=terminal)
        self._transpositions[key] = node
        return node

    def _reuse_root(self, key):
        """Re-root the cached tree on ``key`` if it is reachable from it."""
        if not self.reuse_tree or self._root is None:
            return None
        found = None
        if self._root.key == key:
            # solve() called twice without stepping the env
            found = self._root
        else:
            for edge in self._root.children.values():
                if edge.child is not None and edge.child.key == key:
                    found = edge.child
                    break
        if found is None:
            return None
        self._transpositions = self._reachable(found)
        return found

    @staticmethod
    def _reachable(root):
        """Transposition entries still reachable from the new root."""
        table = {}
        stack = [root]
        while stack:
            node = stack.pop()
            if node.key in table:
                continue
            table[node.key] = node
            for edge in node.children.values():
                if edge.child is not None:
                    stack.append(edge.child)
        return table

    def _clone(self, env):
        """Deep-copy the unwrapped env, minus any buffered render frames."""
        base = env.unwrapped
        frames = getattr(base, 'frame_buffer', None)
        if frames:
            base.frame_buffer = []
        try:
            clone = copy.deepcopy(base)
        finally:
            if frames:
                base.frame_buffer = frames
        return clone

    def _make_inert(self, sim):
        """Strip a simulation clone down to what the search actually reads.

        Only the root clone needs this: it is applied once and every deeper
        clone inherits it through the deepcopy. Three things go:

        * observation encoding, stubbed out via :func:`_inert_obs`;
        * the cached observation deque it fills;
        * the observation/action spaces, which the search never consults and
          which are otherwise the single most expensive part of cloning
          (their bounds are dense arrays the size of an observation).
        """
        sim._get_obs = _inert_obs
        sim._init_obs = _inert_obs
        obs = getattr(sim, 'obs', None)
        if obs is not None:
            obs.clear()
        sim.observation_space = None
        sim.action_space = None
        return sim

    def _joint_actions(self, env):
        """Joint actions worth trying from ``env``'s state.

        Dead snakes' actions are ignored by ``SnakeEnv.step``, so they are
        pinned to 0 rather than enumerated.
        """
        base = env.unwrapped
        n_actions = len(base.action_dict)
        alive = [i for i, snake in enumerate(base.snakes) if snake.alive]
        if not alive:
            return []

        total = n_actions ** len(alive)
        cap = self.max_joint_actions
        if cap is not None and total > cap:
            combos = {
                tuple(self.np_random.integers(0, n_actions, size=len(alive)))
                for _ in range(cap)
            }
        else:
            combos = itertools.product(range(n_actions), repeat=len(alive))

        actions = []
        for combo in combos:
            joint = [0] * base.num_snakes
            for slot, snake_idx in enumerate(alive):
                joint[snake_idx] = int(combo[slot])
            actions.append(tuple(joint))
        return actions

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def _simulate(self, root):
        path = []
        node = root
        visited = {root.key}
        depth = 0

        while True:
            if node.terminal:
                value = 0.0
                break
            if depth >= self.max_depth:
                value = self._rollout(node)
                break
            if node.untried:
                action = node.untried.pop()
                edge = self._expand(node, action)
                path.append((node, action))
                child = edge.child
                value = 0.0 if child.terminal else self._rollout(child)
                break

            action = self._uct_select(node)
            edge = node.children[action]
            path.append((node, action))
            child = edge.child
            if child.key in visited:
                # the search looped back on itself; stop descending
                value = 0.0 if child.terminal else self._rollout(child)
                break
            visited.add(child.key)
            node = child
            depth += 1

        self._backup(path, value)

    def _expand(self, node, action):
        sim = self._clone(node.env)
        _, rews, terminated, truncated, _ = sim.step(list(action))
        reward = float(self.communal_reward_fn(rews))
        terminal = self._is_terminal(terminated, truncated)

        key = self.extract_state_hash(sim)
        child = self._transpositions.get(key)
        if child is None:
            child = self._make_node(key, sim, terminal)
        elif terminal:
            child.terminal = True

        edge = _Edge(child, reward)
        node.children[action] = edge
        return edge

    def _uct_select(self, node):
        log_visits = math.log(max(node.visits, 1))
        best_score = -float('inf')
        best_action = None
        for action, edge in node.children.items():
            if edge.visits == 0:
                score = float('inf')
            else:
                q = self._value_range.normalize(edge.value)
                score = q + self.exploration_weight * math.sqrt(
                    log_visits / edge.visits)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _rollout(self, node):
        """Random rollout, returning the discounted communal return."""
        if self.rollout_depth <= 0:
            return 0.0
        sim = self._clone(node.env)
        total = 0.0
        discount = 1.0
        for _ in range(self.rollout_depth):
            action = self._random_joint_action(sim)
            if action is None:
                break
            _, rews, terminated, truncated, _ = sim.step(action)
            total += discount * float(self.communal_reward_fn(rews))
            discount *= self.discount
            if self._is_terminal(terminated, truncated):
                break
        return total

    def _random_joint_action(self, env):
        """One uniformly random joint action, or None if no snake is alive."""
        base = env.unwrapped
        n_actions = len(base.action_dict)
        joint = [0] * base.num_snakes
        alive = False
        for i, snake in enumerate(base.snakes):
            if snake.alive:
                joint[i] = int(self.np_random.integers(n_actions))
                alive = True
        return joint if alive else None

    def _backup(self, path, value):
        ret = value
        for node, action in reversed(path):
            edge = node.children[action]
            ret = edge.reward + self.discount * ret
            node.visits += 1
            edge.visits += 1
            edge.value_sum += ret
            self._value_range.update(edge.value)

    @staticmethod
    def _is_terminal(terminated, truncated):
        return all(terminated) or all(truncated)
