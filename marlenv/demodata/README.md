# demo data

Generated, not tracked — everything here rebuilds from a seed.

## Episodes (HuggingFace datasets)

Components are collected separately so training recipes can mix them in
whatever proportion; mixing at collection time bakes in a ratio that cannot
be undone.

```bash
CK=az_obs_latest.pt
python examples/data/collect_dataset.py --preset expert  --episodes 1200 --workers 20 --checkpoint $CK
python examples/data/collect_dataset.py --preset explore --episodes 1500 --workers 20 --checkpoint $CK
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
python examples/analysis/grade_rollout.py --out marlenv/demodata
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
python examples/analysis/grade_frames.py \
    --models marlenv/demodata/wm_ctx48/model.pt \
             marlenv/demodata/wam_deep/model_step8000.pt \
    --names single multi
python examples/play/rollout_wam.py \
    --model marlenv/demodata/wam_deep/model_step8000.pt \
    --checkpoint marlenv/demodata/az_policy.pt
```


## A bug worth knowing about: the move that kills

The cardinal action for a step used to be read off the pose the agent ended
up in. A snake that dies entering a cell has no resulting pose, so its
final move was stored as an all-zero one-hot -- and `argmax` reads all
zeros as index 0, which is **UP**.

Every death in the affected data therefore claimed the snake had been
heading north. A model trained on it learns that going up is what kills
you, and dies more readily against the top wall. That is how it was
eventually noticed, months of work after it was introduced.

It was found once and fixed in the *datasets*, by `patch_aftermath.py`, and
not in the collector that writes them. So it returned the moment fresh data
was collected. Both are fixed now: `marlenv/data/collect.py` derives the
action from the heading the agent had and the turn it chose, which is
correct whether or not it survives arriving.

### Checking a dataset for it

A repaired set has a roughly uniform spread of fatal headings, because no
direction is intrinsically more dangerous. All-north means the bug:

```python
import numpy as np
from datasets import load_from_disk
from marlenv.data import decode_episode

deaths, valid = 0, 0
for row in load_from_disk('marlenv/demodata/expert').select(range(60)):
    episode = decode_episode(row)
    alive, cardinal = episode['alive_mask'], episode['cardinal_actions']
    for agent in range(alive.shape[1]):
        living = np.flatnonzero(alive[:, agent])
        if len(living) < 2:
            continue
        last = int(living[-1])
        if last + 1 < alive.shape[0] and not alive[last + 1, agent]:
            deaths += 1
            valid += int(cardinal[last, agent].sum() == 1)
print(f'fatal actions recorded: {valid / max(deaths, 1):.3f}')
```

A blank action is legitimate in exactly one place: the last frame of an
episode that ran out of steps, where no action was ever taken. It is not
legitimate at a death.

### Repairing an existing set

```bash
python examples/data/patch_aftermath.py \
    --components expert_nogradient explore_nogradient \
    --background-gradient 0
```

Pass the collection settings the set was made with -- the script rebuilds
frames from the recorded seed and refuses to write if its reconstruction
does not match what is stored, so a mismatched gradient or noise level
stops it rather than corrupting it.

Both gradient-free sets have been repaired: 2011 and 4284 deaths, verified
across every episode, with fatal headings spread evenly over the four
directions.
