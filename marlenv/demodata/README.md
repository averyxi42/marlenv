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
