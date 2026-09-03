"""Multi-agent sequences: every agent of an episode, together.

One episode is one sequence, carrying all agents rather than being split
into one sequence per agent. Agents are placed in a single shared frame, so
their starting offsets relative to each other are recorded; only the
differences are used, so nothing about the board's absolute position enters.

A dead agent keeps its slot with a black frame, which is what the model
learns to predict on death, and its loss is masked from the frame after that
so it is not trained to keep restating a fact already established.
"""
import numpy as np

from marlenv.core.snake import Direction
from marlenv.grading.compare import unrotate_view

BLACK = 0
HEADINGS = list(Direction)


def episode_sequence(episode):
    """One episode as ``(observations, actions, alive, origins)``.

    ``observations`` is ``(T, agents, S, S, 3)`` north-up, ``actions``
    ``(T - 1, agents)`` cardinal indices, ``alive`` ``(T, agents)`` and
    ``origins`` ``(agents, 2)`` relative to the first agent.
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

    # a dead agent shows black from its death frame onward
    observations = views.copy()
    trained = alive.copy()
    for agent in range(agents):
        living = np.flatnonzero(alive[:, agent])
        if len(living) == 0:
            continue
        last = living[-1]
        if last + 1 < frames:
            observations[last + 1:, agent] = BLACK
            # the death frame itself is worth predicting; later ones are not
            trained[last + 1, agent] = True
            trained[last + 2:, agent] = False

    origins = poses[0, :, :2].astype(np.int64)
    origins = origins - origins[0]
    return observations, actions[:-1], alive, trained, origins


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
    origins = np.zeros((count, agents, 2), np.int64)

    for index, (obs, act, liv, tra, org) in enumerate(rows):
        steps = min(len(obs), length)
        observations[index, :steps] = obs[:steps]
        alive[index, :steps] = liv[:steps]
        trained[index, :steps] = tra[:steps]
        mask[index, :steps] = True
        take = min(len(act), length - 1)
        actions[index, :take] = act[:take]
        origins[index] = org

    return {'observations': observations, 'actions': actions, 'alive': alive,
            'trained': trained, 'mask': mask, 'origins': origins}
