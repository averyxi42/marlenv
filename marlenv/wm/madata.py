"""Multi-agent sequences: every agent of an episode, together.

One episode is one sequence, carrying all agents rather than being split
into one sequence per agent. Agents are placed in a single shared frame, so
their starting offsets relative to each other are recorded; only the
differences are used, so nothing about the board's absolute position enters.

When an agent dies, the frame that follows is the ordinary view from the
cell it died entering -- the aftermath, with its own snake gone. That keeps
every observation in distribution and needs no sentinel value, and it is
what the identity-free formulation implies: a view belongs to a position,
not to an agent. Death is read off the centre cell, which is the viewer's
own head while it lives and something else once it does not. The viewpoint
then stops being updated, so later frames are masked out of the loss.
"""
import numpy as np

from marlenv.core.snake import Direction
from marlenv.grading.compare import unrotate_view

HEADINGS = list(Direction)


def episode_sequence(episode):
    """One episode as ``(observations, actions, alive, trained, positions)``.

    ``observations`` is ``(T, agents, S, S, 3)`` north-up, ``actions``
    ``(T - 1, agents)`` cardinal indices, ``alive`` ``(T, agents)`` and
    ``positions`` ``(T, agents, 2)``.

    Positions are kept for every step, not just the first. A crop can start
    anywhere in the episode, and the agents have drifted apart by then --
    their offsets at step 0 say nothing about their offsets at step 40. Only
    differences are ever used, so the absolute values here do not matter.
    """
    alive = episode['alive_mask']
    frames, agents = alive.shape
    poses = episode['poses']

    views = np.stack([
        [unrotate_view(episode['observations'][t, a],
                       HEADINGS[int(poses[t, a, 2])])
         for a in range(agents)]
        for t in range(frames)])
    actions = episode['cardinal_actions'].argmax(axis=-1)

    observations = views
    trained = alive.copy()
    for agent in range(agents):
        living = np.flatnonzero(alive[:, agent])
        if len(living) == 0:
            continue
        last = living[-1]
        if last + 1 < frames:
            # the aftermath is worth predicting; the viewpoint then stops
            trained[last + 1, agent] = True
            trained[last + 2:, agent] = False

    # a dead agent's pose is recorded as -1 once its viewpoint stops being
    # updated; carry the last real one forward so the array stays usable
    positions = poses[:, :, :2].astype(np.int64)
    for agent in range(agents):
        for step in range(1, frames):
            if positions[step, agent, 0] < 0:
                positions[step, agent] = positions[step - 1, agent]
    return observations, actions[:-1], alive, trained, positions


def build_multi_sequences(datasets, limit=None):
    """Padded arrays over one or more HuggingFace datasets."""
    from marlenv.data import decode_episode

    rows = []
    for dataset in datasets:
        for row in dataset:
            if limit is not None and len(rows) >= limit:
                break
            rows.append(episode_sequence(decode_episode(row)))
    if not rows:
        raise ValueError('no episodes')

    length = max(len(r[0]) for r in rows)
    agents = rows[0][0].shape[1]
    view = rows[0][0].shape[2]
    count = len(rows)

    observations = np.zeros((count, length, agents, view, view, 3), np.uint8)
    actions = np.zeros((count, length - 1, agents), np.int64)
    alive = np.zeros((count, length, agents), bool)
    trained = np.zeros((count, length, agents), bool)
    mask = np.zeros((count, length), bool)
    positions = np.zeros((count, length, agents, 2), np.int64)

    for index, (obs, act, liv, tra, pos) in enumerate(rows):
        steps = min(len(obs), length)
        observations[index, :steps] = obs[:steps]
        alive[index, :steps] = liv[:steps]
        trained[index, :steps] = tra[:steps]
        mask[index, :steps] = True
        take = min(len(act), length - 1)
        actions[index, :take] = act[:take]
        positions[index, :steps] = pos[:steps]
        # a padded tail would otherwise read as the board's top-left corner
        positions[index, steps:] = pos[steps - 1]

    return {'observations': observations, 'actions': actions, 'alive': alive,
            'trained': trained, 'mask': mask, 'positions': positions}
