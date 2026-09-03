"""Recording rollouts and grading a predictor against them.

The intended use is to fix an initial state and an action sequence, run the
simulator with noise off, and run a world model over the same actions. Both
produce head-frame observations; :func:`grade` snaps each to the palette and
counts where the resulting grids disagree, per class pair.
"""
from typing import NamedTuple

import numpy as np

from marlenv.grading.compare import ConfusionMatrix, align_local_obs, align_obs
from marlenv.grading.poses import action_seq_to_pose_seq, pose_from_snake


class Rollout(NamedTuple):
    """One agent's trajectory, with everything needed to grade against it."""

    actions: np.ndarray        # (T, num_snakes) joint actions taken
    poses: list                # (T + 1) actual head poses from the simulator
    dead_reckoned: list        # (T + 1) poses integrated from actions alone
    local_obs: np.ndarray      # (T + 1, S, S, 3) the agent's own views
    global_obs: np.ndarray     # (T + 1, H, W, 3) whole board
    alive: np.ndarray          # (T + 1,) whether the agent was alive
    agent: int

    @property
    def steps(self):
        return len(self.actions)

    def pose_drift(self):
        """Steps where dead reckoning left the simulator's actual path.

        Non-empty means the agent died or was otherwise stopped, so frames
        after that point are not comparable in the agent's own frame.
        """
        return [i for i, (a, b) in enumerate(zip(self.poses,
                                                 self.dead_reckoned))
                if a != b]


def record_rollout(env, actions, agent=0):
    """Play a fixed action sequence and record what the agent saw.

    ``actions`` is ``(T, num_snakes)``. The env should already be reset; it
    is stepped in place.
    """
    base = env.unwrapped
    actions = np.asarray(actions, dtype=int)
    if actions.ndim != 2:
        raise ValueError('actions must be (steps, num_snakes)')

    start = pose_from_snake(base.snakes[agent])
    poses = [start]
    local = [base.egocentric_rgb()[agent]]
    world = [base.render('rgb_array')]
    alive = [base.snakes[agent].alive]

    for joint in actions:
        _, _, terminated, truncated, _ = env.step(list(joint))
        poses.append(pose_from_snake(base.snakes[agent]))
        local.append(base.egocentric_rgb()[agent])
        world.append(base.render('rgb_array'))
        alive.append(base.snakes[agent].alive)
        if all(terminated) or all(truncated):
            break

    taken = actions[:len(poses) - 1]
    return Rollout(taken, poses,
                   action_seq_to_pose_seq(start, taken[:, agent]),
                   np.array(local), np.array(world),
                   np.array(alive, dtype=bool), agent)


def grade(rollout, predicted_local, reference='local', matrix=None,
          poses=None):
    """Score predicted head-frame observations against a rollout.

    ``reference='local'`` compares against the agent's recorded view on the
    cells both cover; ``'global'`` compares against the whole board, which
    also catches a prediction that is right about its surroundings but
    placed wrong. Frames after the agent dies are skipped, since it has no
    view to be right or wrong about.
    """
    matrix = matrix or ConfusionMatrix()
    predicted_local = np.asarray(predicted_local)
    poses = poses or rollout.poses

    for step, prediction in enumerate(predicted_local):
        if step >= len(poses) or not rollout.alive[step]:
            break
        pose = poses[step]
        if reference == 'local':
            alignment = align_local_obs(pose, rollout.local_obs[step],
                                        pose, prediction)
        elif reference == 'global':
            alignment = align_obs(pose, prediction, rollout.global_obs[step])
        else:
            raise ValueError("reference must be 'local' or 'global'")
        matrix.update_from(alignment)
    return matrix


def save_rollout(path, rollout):
    """Store a rollout as a .npz, poses flattened to plain arrays."""
    np.savez_compressed(
        path,
        actions=rollout.actions,
        poses=np.array([(p.row, p.col, p.direction.value[0],
                         p.direction.value[1]) for p in rollout.poses]),
        local_obs=rollout.local_obs,
        global_obs=rollout.global_obs,
        alive=rollout.alive,
        agent=rollout.agent,
    )
