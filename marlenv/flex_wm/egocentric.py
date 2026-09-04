"""Rebuilding a multi-agent episode from what one agent could have seen.

The recorded episodes are omniscient: every agent's view at every step. A
real agent has only its own. What it can additionally recover is limited
and specific -- while another snake's head is inside its view, it can see
where that snake is and, from one step to the next, what it did.

Three consequences shape everything here.

**Identity does not survive an absence.** An agent that leaves the view and
returns is, to the observer, a new agent. Nothing carries across the gap:
no re-identification is attempted, and each visit gets a fresh id. This is
the reason for the pair formulation -- ids are compared, never indexed, so
an episode may contain far more of them than are ever live at once.

**An action needs two observations.** It is a difference of positions, so a
visit of length L yields L observations and L-1 actions; the last
observation before the snake leaves cannot say what it did next. A visit of
a single step yields no action at all and is dropped, since a lone glimpse
teaches nothing about behaviour.

**Views are turned north-up first.** The recorded observation is in its
own agent's head frame, so two agents looking at one cell hold different
pictures of it until both are unrotated. Everything here is world frame,
which is also what the model reads.

**Observation is partial.** Another agent's view is centred on its own head
and reaches past what the observer can see. The patches that fall outside
are not dropped from the sequence -- they exist, at a known place and time
-- they are simply unknown, which is what the top of the noise schedule
means. They are also not training targets: there is no truth to compare
against, and regressing on one would be regressing on nothing.
"""
from dataclasses import dataclass

import numpy as np

from marlenv.core.snake import Direction
from marlenv.grading.compare import unrotate_view

HEADINGS = list(Direction)


@dataclass
class EgoEpisode:
    """One episode as a flat set of pairs, from a single agent's vantage.

    observations ``(P, view, view, 3)`` uint8
    actions      ``(P,)`` cardinal indices
    agent        ``(P,)`` identity, fresh for every visit
    time         ``(P,)`` step
    position     ``(P, 2)`` where the observation was taken
    visible      ``(P, tokens_per_frame)`` patch was observable
    acted        ``(P,)`` the action could be recovered
    ego          which snake the episode was rebuilt around
    """

    observations: np.ndarray
    actions: np.ndarray
    agent: np.ndarray
    time: np.ndarray
    position: np.ndarray
    visible: np.ndarray
    acted: np.ndarray
    ego: int

    def __len__(self):
        return len(self.time)


def patch_offsets(view, patch):
    """Cell offset of each patch centre from the centre of a view.

    The same geometry the model uses, without needing one built: the
    reconstruction has to know where patches sit before there is a model to
    ask.
    """
    grid = view // patch
    radius = view // 2
    index = np.arange(grid) * patch + patch // 2 - radius
    rows, cols = np.meshgrid(index, index, indexing='ij')
    return np.stack([rows.reshape(-1), cols.reshape(-1)], axis=-1)


def head_in_view(watcher, target, radius):
    """Is ``target``'s head inside a view of ``radius`` around ``watcher``?"""
    return bool((np.abs(np.asarray(target) - np.asarray(watcher))
                 <= radius).all())


def visible_runs(flags, shortest=2):
    """Contiguous stretches of truth in ``flags``, as ``(start, stop)``.

    Runs shorter than ``shortest`` are left out: a visit of one step yields
    no action, since an action is a difference between two observations.
    """
    runs, start = [], None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= shortest:
                runs.append((start, index))
            start = None
    return runs


def patch_visibility(watcher, target, offsets, radius, patch):
    """Which of ``target``'s patches lie wholly inside ``watcher``'s view.

    watcher ``(2,)`` centre of the observing view
    target  ``(2,)`` centre of the observed view
    offsets ``(tokens, 2)`` cell offset of each patch centre from its own
            view's centre
    radius  half-width of a view, in cells
    patch   patch width, in cells

    A patch is counted only when all of it is inside the observer's view.
    Half a patch is not a patch: its token carries the whole block, so a
    partly seen one is a token whose content is partly unknown, and there
    is nowhere finer than the token to say so.
    """
    reach = patch // 2
    centres = np.asarray(target)[None] + np.asarray(offsets)
    low = centres - reach
    high = centres + reach
    inside = np.asarray(watcher) - radius
    outside = np.asarray(watcher) + radius
    return ((low >= inside[None]).all(axis=1)
            & (high <= outside[None]).all(axis=1))


def cardinal_from_step(before, after):
    """The cardinal action that moves a head from ``before`` to ``after``."""
    delta = (int(after[0] - before[0]), int(after[1] - before[1]))
    for index, heading in enumerate(HEADINGS):
        if heading.value == delta:
            return index
    return None


def egocentric_pairs(episode, offsets, ego=None, rng=None, radius=4,
                     patch=3, others=True):
    """An episode as a flat pair set, ready for a batcher.

    The same reconstruction as :func:`egocentric_episode`, in the shape the
    batcher crops. Every pair is a target -- what is unknown is said per
    patch, by ``visible``, not by dropping the observation.
    """
    seen = egocentric_episode(episode, offsets, ego=ego, rng=rng,
                              radius=radius, patch=patch, others=others)
    return {'observations': seen.observations, 'actions': seen.actions,
            'agent': seen.agent, 'time': seen.time, 'position': seen.position,
            'visible': seen.visible, 'acted': seen.acted,
            'trained': np.ones(len(seen), bool)}


def egocentric_episode(episode, offsets, ego=None, rng=None, radius=4,
                       patch=3, others=True):
    """Rebuild ``episode`` as one agent could have recorded it.

    episode a decoded episode, as :func:`marlenv.data.decode_episode` gives
    offsets ``(tokens_per_frame, 2)`` patch centre offsets, from the model
    ego     which snake observes; chosen at random when not given
    rng     for that choice
    radius  view radius, in cells
    patch   patch width, in cells
    others  keep the agents the observer recovered. Setting it False
            leaves one agent's record and nothing else, which is the
            baseline the egocentric run has to beat: whatever a model
            manages without ever being shown a second agent is not
            something the second agent taught it

    Returns an :class:`EgoEpisode`. The observer's own pairs are complete;
    every other pair comes from a visit and is partial.
    """
    alive = episode['alive_mask']
    poses = episode['poses']
    observations = episode['observations']
    cardinal = episode['cardinal_actions'].argmax(axis=-1)
    frames, agents = alive.shape
    tokens = len(offsets)

    if ego is None:
        rng = rng or np.random.default_rng()
        living = [a for a in range(agents) if alive[:, a].sum() >= 2]
        if not living:
            raise ValueError('no agent lives long enough to observe from')
        ego = int(rng.choice(living))

    def upright(step, who):
        """That agent's view, turned north-up like the model reads it."""
        return unrotate_view(observations[step, who],
                             HEADINGS[int(poses[step, who, 2])])

    rows = []
    next_id = 0

    # the observer's own record: complete, and its own actions are its own
    own = np.flatnonzero(alive[:, ego])
    # the observer knows its own actions; the only one it lacks is at the
    # end of the episode, where none was recorded. A death is not that
    # case -- the move that killed it is a move like any other
    ended = len(own) and own[-1] + 1 >= frames
    if len(own) >= 2:
        for position, step in enumerate(own):
            rows.append(dict(
                observation=upright(step, ego),
                action=cardinal[step, ego],
                agent=next_id,
                time=int(step),
                position=poses[step, ego, :2],
                visible=np.ones(tokens, bool),
                acted=position < len(own) - 1 or not ended))
        next_id += 1

    for other in range(agents) if others else ():
        if other == ego:
            continue
        seen = np.array([
            bool(alive[t, ego] and alive[t, other]
                 and head_in_view(poses[t, ego, :2], poses[t, other, :2],
                                  radius))
            for t in range(frames)])
        for start, stop in visible_runs(seen, shortest=2):
            for step in range(start, stop):
                # the action is a difference, so the last step of a visit
                # cannot supply one
                last = step == stop - 1
                rows.append(dict(
                    observation=upright(step, other),
                    action=(0 if last else
                            cardinal_from_step(poses[step, other, :2],
                                               poses[step + 1, other, :2])
                            or 0),
                    agent=next_id,
                    time=int(step),
                    position=poses[step, other, :2],
                    visible=patch_visibility(poses[step, ego, :2],
                                             poses[step, other, :2],
                                             offsets, radius, patch),
                    acted=not last))
            next_id += 1

    if not rows:
        raise ValueError('nothing observable in this episode')
    order = np.argsort([row['time'] for row in rows], kind='stable')
    take = lambda key: np.stack([rows[i][key] for i in order])
    return EgoEpisode(
        observations=take('observation'),
        actions=take('action').astype(np.int64),
        agent=take('agent').astype(np.int64),
        time=take('time').astype(np.int64),
        position=take('position').astype(np.int64),
        visible=take('visible'),
        acted=take('acted'),
        ego=int(ego))
