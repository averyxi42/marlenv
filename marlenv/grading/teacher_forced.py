"""One-step prediction from recorded truth, without a rollout runner or KV cache.

Every query is rebuilt from real history. Only the target views are denoised;
predictions never become conditioning data. The target population is determined
before the action, so a fatal transition receives one aftermath prediction.
"""
from dataclasses import asdict, dataclass, field
from itertools import combinations
import json

import numpy as np
import torch

from marlenv.core.palette import decode_grid
from marlenv.core.snake import Direction
from marlenv.flex_wm.pairs import PairBatch
from marlenv.grading.compare import PALETTE_SNAKES, unrotate_view
from marlenv.wm.data import to_model_input, to_pixels
from marlenv.wm.diffusion import alpha_sigma, from_velocity
from marlenv.wm.multiagent import actions_to_signal

HEADINGS = list(Direction)
MOVES = np.array([heading.value for heading in HEADINGS], dtype=np.int64)


@dataclass
class Recording:
    """North-up views and actual positions, including the first death view.

    observations: (T+1, N, V, V, 3) uint8
    positions: (T+1, N, 2) integers, never sentinel coordinates
    alive: (T+1, N) bool
    actions: (T, N) cardinal indices; -1 for agents already dead
    """
    observations: np.ndarray
    positions: np.ndarray
    alive: np.ndarray
    actions: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        frames, agents, height, width, channels = self.observations.shape
        if self.observations.dtype != np.uint8 or height != width or height % 2 != 1 or channels != 3:
            raise ValueError('observations must be odd square uint8 RGB views')
        if self.positions.shape != (frames, agents, 2) or self.alive.shape != (frames, agents):
            raise ValueError('recording positions/alive shapes disagree')
        if self.actions.shape != (frames - 1, agents) or frames < 2:
            raise ValueError('need one action row per recorded transition')
        if self.alive.dtype != bool or not np.issubdtype(self.positions.dtype, np.integer):
            raise ValueError('alive must be boolean and positions integer')
        if not np.issubdtype(self.actions.dtype, np.integer):
            raise ValueError('actions must be integer cardinal indices')
        if np.any(self.alive[1:] & ~self.alive[:-1]):
            raise ValueError('respawns need explicit new identities; unsupported here')
        acted = self.alive[:-1]
        if np.any((self.actions[acted] < 0) | (self.actions[acted] >= len(HEADINGS))):
            raise ValueError('missing or invalid action for a living agent')
        if np.any(self.actions[~acted] != -1):
            raise ValueError('an already dead agent cannot have an outgoing action')
        delta = self.positions[1:] - self.positions[:-1]
        if not np.array_equal(delta[acted], MOVES[self.actions[acted]]):
            raise ValueError('recorded positions disagree with actions, including fatal moves')
        if np.any(delta[~acted]):
            raise ValueError('a retired simulator viewpoint moved')

    @property
    def steps(self):
        return len(self.actions)

    def save(self, path):
        np.savez_compressed(path, observations=self.observations,
                            positions=self.positions, alive=self.alive,
                            actions=self.actions,
                            metadata=json.dumps(self.metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(*(data[key].copy() for key in
                         ('observations', 'positions', 'alive', 'actions')),
                       metadata=json.loads(str(data['metadata'])))


def collect_recording(env, policy, *, seed, steps, metadata=None):
    """Record the simulator once; policy accepts env and returns ego actions.

    Reads actual snake poses even after death. No dependency on marlenv.data,
    its sentinel poses, or any model/runner is involved in collection.
    """
    if steps < 1:
        raise ValueError('steps must be positive')
    env.reset(seed=seed)
    base = env.unwrapped
    observations, positions, alive, actions = [], [], [], []

    def capture():
        observations.append(np.stack([
            unrotate_view(view, snake.direction)
            for view, snake in zip(base.egocentric_rgb(), base.snakes)]))
        positions.append(np.array([s.head_coord for s in base.snakes], dtype=np.int64))
        alive.append(np.array([s.alive for s in base.snakes], dtype=bool))

    capture()
    for _ in range(steps):
        if not alive[-1].any():
            break
        _, _, terminated, truncated, _ = env.step(list(policy(env)))
        # SnakeEnv updates direction for every agent that acted, including
        # one killed on arrival. It leaves already dead directions alone.
        actions.append(np.array([
            HEADINGS.index(s.direction) if live else -1
            for s, live in zip(base.snakes, alive[-1])], dtype=np.int64))
        capture()
        if np.all(np.asarray(terminated) | np.asarray(truncated)):
            break
    return Recording(np.stack(observations), np.stack(positions),
                     np.stack(alive), np.stack(actions),
                     {**(metadata or {}), 'seed': int(seed), 'format_version': 1})


@dataclass
class Prediction:
    agents: np.ndarray
    positions: np.ndarray
    observations: np.ndarray


def prediction_query(recording, target, window, device):
    """Build a query without reading target/future images, poses or life flags.

    window counts the target frame plus its real history. Each past death
    observation is included once, with its outgoing action marked unknown.
    """
    if not 1 <= target <= recording.steps or window < 2:
        raise ValueError('target must be a recorded transition and window >= 2')
    first = max(0, target - window + 1)
    times, agents = [], []
    for time in range(first, target):
        available = recording.alive[time].copy()
        if time:
            available |= recording.alive[time - 1]
        who = np.flatnonzero(available)
        times.extend([time] * len(who))
        agents.extend(who.tolist())
    times, agents = np.asarray(times, dtype=np.int64), np.asarray(agents, dtype=np.int64)
    targets = np.flatnonzero(recording.alive[target - 1])
    if not len(targets):
        raise ValueError('no living viewpoints before this transition')
    target_positions = (recording.positions[target - 1, targets]
                        + MOVES[recording.actions[target - 1, targets]])
    history = to_model_input(recording.observations[times, agents])
    positions = np.concatenate([recording.positions[times, agents], target_positions])
    positions = positions - positions[0]
    observations = np.concatenate([history, np.zeros((len(targets), *history.shape[1:]), np.float32)])
    known = np.concatenate([recording.alive[times, agents], np.zeros(len(targets), bool)])
    actions = np.concatenate([np.maximum(recording.actions[times, agents], 0),
                              np.zeros(len(targets), np.int64)])
    to = lambda array: torch.from_numpy(array[None]).to(device)
    pairs = PairBatch(
        observations=to(observations), actions=to(actions),
        agent=to(np.concatenate([agents, targets])),
        time=to(np.concatenate([times, np.full(len(targets), target, np.int64)]) - first),
        position=to(positions), acted=to(known))
    return pairs, len(times), targets, target_positions


@torch.inference_mode()
def predict_next(model, recording, target, *, window=48, denoise_steps=12, seed=0):
    """Independently sample joint next views conditioned only on recorded past.

    Full forward passes deliberately re-encode the finite history on each
    denoising step. No generated image, inferred retirement, or cached state
    can survive this call. Callers load the model in eval mode.
    """
    if denoise_steps < 1 or model.training or model.frame != 'world':
        raise ValueError('need eval-mode world-frame model and positive denoise_steps')
    if model.view != recording.observations.shape[2] or model.action_out.out_features != 4:
        raise ValueError('recording and model observation/action spaces disagree')
    device = next(model.parameters()).device
    pairs, past, agents, positions = prediction_query(recording, target, window, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    signal = actions_to_signal(pairs.actions, 4)
    unknown = torch.randn(signal.shape, generator=generator, device=device)
    signal = torch.where(pairs.acted[..., None], signal, unknown)
    action_tau = (~pairs.acted).float()
    frames = pairs.observations.clone()
    frames[:, past:] = torch.randn(frames[:, past:].shape, generator=generator, device=device)
    frame_tau = torch.zeros_like(action_tau)
    levels = torch.linspace(1, 0, denoise_steps + 1, device=device)
    for now, later in zip(levels[:-1], levels[1:]):
        frame_tau[:, past:] = now
        velocity, _ = model(pairs, frames, signal, frame_tau, action_tau, window=window)
        clean, noise = from_velocity(frames[:, past:], velocity[:, past:], frame_tau[:, past:])
        alpha, sigma = alpha_sigma(later)
        frames[:, past:] = alpha * clean.clamp(-1, 1) + sigma * noise
    return Prediction(agents, positions, to_pixels(frames[0, past:].cpu().numpy()))


@dataclass
class CellCounts:
    cells: int = 0
    exact: int = 0
    snake_union: int = 0
    snake_exact: int = 0
    snake_both: int = 0

    def add(self, left, right):
        if left.shape != right.shape:
            raise ValueError('compared grids must have equal shapes')
        a, b = np.isin(left % 10, (3, 4, 5)), np.isin(right % 10, (3, 4, 5))
        union, same = a | b, left == right
        self.cells += int(left.size)
        self.exact += int(same.sum())
        self.snake_union += int(union.sum())
        self.snake_exact += int((union & same).sum())
        self.snake_both += int((a & b).sum())

    def report(self):
        ratio = lambda n, d: n / d if d else None
        return {**asdict(self), 'all_exact_rate': ratio(self.exact, self.cells),
                'snake_exact_rate': ratio(self.snake_exact, self.snake_union),
                'snake_occupancy_iou': ratio(self.snake_both, self.snake_union)}


def overlap(left, right, left_position, right_position):
    """Intersect two north-up square grids, directly in cell coordinates."""
    a = np.asarray(left_position) - np.array(left.shape) // 2
    b = np.asarray(right_position) - np.array(right.shape) // 2
    lo, hi = np.maximum(a, b), np.minimum(a + left.shape, b + right.shape)
    if np.any(hi <= lo):
        return left[:0, :0], right[:0, :0]
    slices = lambda origin: tuple(slice(int(x), int(y)) for x, y in zip(lo - origin, hi - origin))
    return left[slices(a)], right[slices(b)]


@dataclass
class Scores:
    steps: int = 0
    views: int = 0
    death_views: int = 0
    overlapping_pairs: int = 0
    consistency: CellCounts = field(default_factory=CellCounts)
    truth_control: CellCounts = field(default_factory=CellCounts)
    accuracy: CellCounts = field(default_factory=CellCounts)
    death_accuracy: CellCounts = field(default_factory=CellCounts)

    def add(self, recording, target, prediction):
        expected = np.flatnonzero(recording.alive[target - 1])
        if not np.array_equal(np.sort(prediction.agents), expected):
            raise ValueError('predictions must cover every pre-action living viewpoint exactly once')
        truth_positions = recording.positions[target, prediction.agents]
        if not np.array_equal(prediction.positions, truth_positions):
            raise ValueError('prediction geometry differs from recorded transition')
        truth = recording.observations[target, prediction.agents]
        if prediction.observations.shape != truth.shape:
            raise ValueError('prediction views have the wrong shape')
        predicted = [decode_grid(frame, PALETTE_SNAKES) for frame in prediction.observations]
        real = [decode_grid(frame, PALETTE_SNAKES) for frame in truth]
        self.steps += 1
        self.views += len(predicted)
        for slot, agent in enumerate(prediction.agents):
            self.accuracy.add(real[slot], predicted[slot])
            if not recording.alive[target, agent]:
                self.death_views += 1
                self.death_accuracy.add(real[slot], predicted[slot])
        for a, b in combinations(range(len(predicted)), 2):
            pair = overlap(predicted[a], predicted[b], *prediction.positions[[a, b]])
            if not pair[0].size:
                continue
            self.overlapping_pairs += 1
            self.consistency.add(*pair)
            control = overlap(real[a], real[b], *truth_positions[[a, b]])
            if not np.array_equal(*control):
                raise ValueError('recorded views disagree in their overlap; invalid truth control')
            self.truth_control.add(*control)

    def merge(self, other):
        for name in ('steps', 'views', 'death_views', 'overlapping_pairs'):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in ('consistency', 'truth_control', 'accuracy', 'death_accuracy'):
            for key, value in asdict(getattr(other, name)).items():
                counts = getattr(self, name)
                setattr(counts, key, getattr(counts, key) + value)

    def report(self):
        return {name: value.report() if isinstance(value, CellCounts) else value
                for name, value in vars(self).items()}


def evaluate_recording(model, recording, *, window=48, denoise_steps=12,
                       seed=0, first_target=1):
    """Score every requested transition; random streams are local to each query."""
    if not 1 <= first_target <= recording.steps:
        raise ValueError('first_target is outside the recording')
    scores = Scores()
    for target in range(first_target, recording.steps + 1):
        if not recording.alive[target - 1].any():
            break
        query_seed = int(np.random.SeedSequence(
            [seed, recording.metadata['seed'], target]).generate_state(1, dtype=np.uint64)[0])
        prediction = predict_next(model, recording, target, window=window,
                                  denoise_steps=denoise_steps, seed=query_seed)
        scores.add(recording, target, prediction)
    return scores
