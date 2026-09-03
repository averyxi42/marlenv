# demo data

Generated, not tracked — everything here rebuilds from a seed.

## Episodes (HuggingFace datasets)

Components are collected separately so training recipes can mix them in
whatever proportion; mixing at collection time bakes in a ratio that cannot
be undone.

```bash
CK=az_obs_latest.pt
python examples/collect_dataset.py --preset expert  --episodes 1200 --workers 20 --checkpoint $CK
python examples/collect_dataset.py --preset explore --episodes 1500 --workers 20 --checkpoint $CK
```

Both on a fixed 15x15 board, 3 agents, view radius 4, 12% obstacles, with
the search policy at 16 simulations and `max_joint_actions=4`.

| component | episodes | transitions | agent frames | steps/ep | deaths/ep | return |
| --- | --- | --- | --- | --- | --- | --- |
| `expert` | 1200 | 95,976 | 287,928 | 80.0 | 0.72 | +18.6 |
| `explore` | 1500 | 102,965 | 308,895 | 70.5 | 2.33 | -0.4 |

`explore` is the same policy with 15% random actions. It is not just noisier
expert data: three times the deaths, so it carries the collision dynamics a
model trained only on expert trajectories would never see.

A `random` preset exists but was not collected -- with obstacles it dies in
about 11 steps with every snake gone, which is mostly initial-state-then-death
and too short to be useful to a sequence model. `explore` covers that ground
with far richer trajectories.

One row per multi-agent episode. Every per-frame column shares the leading
frame axis `N = steps + 1`; the terminal frame is kept because a world model
needs the observation an action leads *to*, so the action and reward rows are
zero-padded at the end and `transition_mask` marks the real transitions.

| column | shape | dtype |
| --- | --- | --- |
| `observations` | `(N, agents, S, S, 3)` | uint8 |
| `ego_actions` | `(N, agents, 3)` | uint8 one-hot |
| `cardinal_actions` | `(N, agents, 4)` | uint8 one-hot |
| `alive_mask` | `(N, agents)` | bool |
| `poses` | `(N, agents, 3)` | int16 — row, col, heading |
| `rewards` | `(N, agents)` | float32 |
| `content` | `(N, H, W)` | int16 — empty/wall/fruit/`3 + agent_id` |
| `body_index` | `(N, H, W)` | int16 — distance from head, `-1` off-snake |
| `transition_mask` | `(N,)` | bool |

Arrays are stored flat and reshaped on read:

```python
from datasets import load_from_disk
from marlenv.data import decode_episode

dataset = load_from_disk('marlenv/demodata/episodes')
episode = decode_episode(dataset[0])   # shapes and dtypes restored
```

## Grading rollouts

```bash
python examples/grade_rollout.py --out marlenv/demodata
```

Writes `.npz` rollouts and scores stand-in predictors, so the grading path
can be checked before a world model exists.

## Checkpoints kept in the repository

The collected datasets stay out of git, since a seed and the collection
script regenerate them. A few trained models are worth keeping, because
they are what the measurements in the history refer to and they take hours
to reproduce.

| file | what it is |
| --- | --- |
| `az_policy.pt` | AlphaZero network. Drives data collection, and the searched prefix a world model rollout is bootstrapped with. |
| `wm_ctx48/model.pt` | Single agent world model, context 48. The baseline every multi-agent number is compared against. |
| `wam_deep/model_step8000.pt` | Multi agent world *action* model, 12 blocks. Diffuses actions alongside frames, so it can be rolled out with several agents. |

Scored on next-frame prediction from clean history, split by cell type,
which is the comparison that matters -- an aggregate over all pixels is
dominated by background and hides the part that is hard:

| | single, 6 blocks | multi, 12 blocks |
| --- | --- | --- |
| centre cell (the viewer's own head) | 1.000 | 0.984 |
| snake cells | 0.815 | 0.810 |
| empty / wall / fruit | 0.981 | 0.995 |
| overall | 0.960 | 0.970 |

```bash
python examples/grade_frames.py \
    --models marlenv/demodata/wm_ctx48/model.pt \
             marlenv/demodata/wam_deep/model_step8000.pt \
    --names single multi
python examples/rollout_wam.py \
    --model marlenv/demodata/wam_deep/model_step8000.pt \
    --checkpoint marlenv/demodata/az_policy.pt
```
