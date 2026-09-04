"""Grading predictions against the simulator.

Note what is *not* re-exported here. ``frames`` and ``ratchet`` score world
models and so import from :mod:`marlenv.wm`, while ``marlenv.wm`` imports
this package for its palette and pose helpers. Pulling them in here closes
that loop, and the failure is not local: ``import marlenv.wm`` stops
working, from the module that did nothing wrong. Import them by name.
"""
from marlenv.grading.compare import (Alignment, ConfusionMatrix,
                                     Disagreement, NUM_CLASSES,
                                     PALETTE_SNAKES, align_local_obs,
                                     align_obs, diff_local_obs, diff_obs,
                                     unrotate_view, view_grid)
from marlenv.grading.rollout import (Rollout, grade, record_rollout,
                                     save_rollout)
from marlenv.grading.poses import (Pose, action_seq_to_pose_seq,
                                   pose_from_snake, step_pose, turn)

__all__ = ['Alignment', 'ConfusionMatrix', 'Disagreement', 'NUM_CLASSES',
           'PALETTE_SNAKES', 'align_obs', 'align_local_obs', 'diff_obs',
           'diff_local_obs', 'unrotate_view', 'view_grid',
           'Pose', 'action_seq_to_pose_seq', 'pose_from_snake', 'step_pose',
           'turn', 'Rollout', 'grade', 'record_rollout', 'save_rollout']
