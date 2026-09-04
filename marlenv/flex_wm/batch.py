"""Cropping episodes that are held as flat sets of pairs.

The earlier batcher cropped the rectangular arrays and turned the result
into pairs at the last moment, which meant it could only ever handle
episodes that were rectangles to begin with. Anything whose agent count
moves -- an episode rebuilt from one agent's view, where snakes come and go
and each visit is a new identity -- had nowhere to go.

Cropping the pairs themselves removes the distinction. A crop is the pairs
whose time falls in a window, whatever they are and however many there are
at each step, so an omniscient episode and an egocentric one take the same
path.
"""
import numpy as np
import torch

from marlenv.flex_wm.pairs import PairBatch

FIELDS = ('observations', 'actions', 'agent', 'time', 'position', 'visible',
          'acted', 'trained')


def flatten_episode(observations, actions, alive, trained, positions,
                    tokens):
    """A rectangular episode as a flat set of pairs.

    observations ``(T, agents, view, view, 3)``
    actions      ``(T, agents)`` cardinal indices
    alive        ``(T, agents)``
    trained      ``(T, agents)`` the observation is a target
    positions    ``(T, agents, 2)``
    tokens       patches per observation

    Only live entries become pairs, so the padding at the end of a short
    episode never enters the set at all.
    """
    steps, agents = alive.shape
    keep = np.argwhere(alive)
    time = keep[:, 0]
    who = keep[:, 1]
    return {
        'observations': observations[time, who],
        'actions': actions[time, who],
        'agent': who.astype(np.int64),
        'time': time.astype(np.int64),
        'position': positions[time, who],
        'visible': np.ones((len(time), tokens), bool),
        'acted': alive[time, who] & (time < steps - 1),
        'trained': trained[time, who],
    }


class PairSetBatcher:
    """Random fixed-length crops over episodes held as pair sets."""

    def __init__(self, episodes, context, seed=0, device='cpu',
                 weights=None, dropouts=None):
        """
        episodes  list of dicts of flat arrays, one per episode
        context   frames per crop
        weights   per-episode action weight, for mixing components
        dropouts  per-episode action dropout
        """
        self.episodes = episodes
        self.context = context
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.weights = (np.ones(len(episodes), np.float32)
                        if weights is None else np.asarray(weights,
                                                           np.float32))
        self.dropouts = (np.zeros(len(episodes), np.float32)
                         if dropouts is None else np.asarray(dropouts,
                                                             np.float32))
        self.spans = np.array([int(e['time'].max()) + 1 for e in episodes])
        self.usable = np.flatnonzero(self.spans >= 2)

    def new_epoch(self):
        """Nothing to do: these episodes do not change between epochs."""

    def crop(self, index):
        """One episode's pairs inside a randomly placed window."""
        episode = self.episodes[index]
        span = min(int(self.spans[index]), self.context)
        start = int(self.rng.integers(0, self.spans[index] - span + 1))
        inside = (episode['time'] >= start) & (episode['time'] < start + span)
        taken = {name: episode[name][inside] for name in FIELDS}
        taken['time'] = taken['time'] - start
        if len(taken['time']):
            # only differences matter, so any consistent origin will do
            taken['position'] = taken['position'] - taken['position'][0]
        return taken

    def batch(self, size):
        """``(PairBatch, weight, dropout)`` over ``size`` random crops."""
        picks = self.rng.choice(self.usable, size=size, replace=True)
        crops = [self.crop(index) for index in picks]
        width = max(max(len(crop['time']) for crop in crops), 1)

        def stack(name, dtype, fill=0):
            shape = crops[0][name].shape[1:]
            out = np.full((size, width, *shape), fill, dtype)
            for row, crop in enumerate(crops):
                out[row, :len(crop[name])] = crop[name]
            return torch.from_numpy(out).to(self.device)

        valid = np.zeros((size, width), bool)
        for row, crop in enumerate(crops):
            valid[row, :len(crop['time'])] = True

        from marlenv.wm.data import to_model_input
        pairs = PairBatch(
            observations=torch.from_numpy(to_model_input(
                stack('observations', np.uint8).cpu().numpy())
            ).to(self.device),
            actions=stack('actions', np.int64),
            # padding takes an identity no real pair carries
            agent=stack('agent', np.int64, fill=-1),
            time=stack('time', np.int64),
            position=stack('position', np.int64),
            valid=torch.from_numpy(valid).to(self.device),
            trained=stack('trained', bool) & torch.from_numpy(valid).to(
                self.device),
            acted=stack('acted', bool) & torch.from_numpy(valid).to(
                self.device),
            visible=stack('visible', bool))
        return (pairs,
                torch.from_numpy(self.weights[picks]).to(self.device),
                torch.from_numpy(self.dropouts[picks]).to(self.device))


class ResampledBatcher(PairSetBatcher):
    """Rebuilds its episodes each epoch, from a freshly chosen viewpoint.

    An egocentric episode is one agent's account of what happened, so which
    agent is asked is part of the sample. Choosing again every epoch shows
    the model the same events from a different vantage, while holding the
    choice fixed *within* an epoch keeps an epoch meaning what it usually
    means -- every episode seen once, not the same episode seen from three
    sides and counted three times.

    Rebuilding is cheap enough to prefer over keeping every viewpoint in
    memory: a few seconds against a few minutes of training, and it does
    not grow with the number of agents the way storing them all does.
    """

    def __init__(self, sources, build, context, seed=0, device='cpu',
                 weights=None, dropouts=None):
        """
        sources  the episodes to rebuild from, in whatever form ``build``
                 expects
        build    ``(source, rng) -> pair set``
        """
        self.sources = list(sources)
        self.build = build
        self.epochs = 0
        self._seed = seed
        super().__init__(self._make(seed), context, seed=seed,
                         device=device, weights=weights, dropouts=dropouts)

    def _make(self, seed):
        rng = np.random.default_rng(seed)
        return [self.build(source, rng) for source in self.sources]

    def new_epoch(self):
        """Choose fresh viewpoints and rebuild."""
        self.epochs += 1
        self.episodes = self._make(self._seed + self.epochs)
        self.spans = np.array([int(e['time'].max()) + 1
                               for e in self.episodes])
        self.usable = np.flatnonzero(self.spans >= 2)
