"""Turning collected episodes into single-agent training sequences.

Each agent's own trajectory is one sequence, so a three-agent episode yields
three. A sequence runs from the first frame to the agent's last living one;
if the agent *died* a single black frame is appended to mark it, which is the
only thing there is to predict past that point. An episode cut short by the
step limit just ends, with no marker, because nothing happened there.
"""
import numpy as np

BLACK = 0


def agent_sequences(episode):
    """Split a decoded episode into one sequence per agent.

    Yields ``(observations, actions, died)`` where ``observations`` is
    ``(L, S, S, 3)`` uint8 and ``actions`` is ``(L - 1,)`` int64, the action
    taken from each frame except the last.
    """
    alive = episode['alive_mask']
    observations = episode['observations']
    ego = episode['ego_actions']
    frames, agents = alive.shape

    for agent in range(agents):
        living = np.flatnonzero(alive[:, agent])
        if len(living) < 2:
            continue
        # frames are contiguous from the start: a snake never comes back
        last = living[-1]
        died = last + 1 < frames and not alive[last + 1, agent]

        obs = observations[:last + 1, agent]
        actions = ego[:last + 1, agent].argmax(axis=-1)
        if died:
            obs = np.concatenate([obs, np.full_like(obs[:1], BLACK)])
        else:
            actions = actions[:-1]
        yield obs, actions.astype(np.int64), bool(died)


def build_sequences(datasets, max_length=None, limit=None):
    """Collect padded arrays over one or more HuggingFace datasets.

    Returns a dict of ``observations (n, L, S, S, 3) uint8``,
    ``actions (n, L - 1) int64``, ``mask (n, L) bool`` and ``died (n,) bool``.
    Padding is masked out of the loss rather than being given a value that
    the model might learn to imitate.
    """
    from marlenv.data import decode_episode

    sequences = []
    for dataset in datasets:
        for index, row in enumerate(dataset):
            if limit is not None and len(sequences) >= limit:
                break
            sequences.extend(agent_sequences(decode_episode(row)))

    if not sequences:
        raise ValueError('no usable agent sequences')

    length = max_length or max(len(obs) for obs, _, _ in sequences)
    view = sequences[0][0].shape[1]

    count = len(sequences)
    observations = np.zeros((count, length, view, view, 3), dtype=np.uint8)
    actions = np.zeros((count, length - 1), dtype=np.int64)
    mask = np.zeros((count, length), dtype=bool)
    died = np.zeros(count, dtype=bool)

    for index, (obs, act, was_dead) in enumerate(sequences):
        keep = min(len(obs), length)
        observations[index, :keep] = obs[:keep]
        mask[index, :keep] = True
        actions[index, :min(len(act), length - 1)] = act[:length - 1]
        died[index] = was_dead and keep == len(obs)

    return {'observations': observations, 'actions': actions, 'mask': mask,
            'died': died}


def to_model_input(observations):
    """uint8 pixels to the [-1, 1] range diffusion works in."""
    return observations.astype(np.float32) / 127.5 - 1.0


def to_pixels(values):
    """Inverse of :func:`to_model_input`, clamped back to uint8."""
    scaled = (np.asarray(values) + 1.0) * 127.5
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
