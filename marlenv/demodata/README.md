# demo data

Generated, not tracked — everything here rebuilds from a seed.

## Episodes (HuggingFace dataset)

```bash
python examples/collect_dataset.py --episodes 64 --out marlenv/demodata/episodes
python examples/collect_dataset.py --episodes 64 --checkpoint az_obs_latest.pt
```

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
