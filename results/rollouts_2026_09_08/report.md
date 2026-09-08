# Corrected three-way autonomous rollouts

Two paired GIFs compare solo, egocentric, and ceiling checkpoints at 24k and
40k total updates. Seed 0 was fixed before generation; no seed or trajectory
was selected based on its outcome.

- [24k comparison](../../diagrams/rollout_corrected_trio24_b12_dp1_seed0.gif)
- [40k comparison](../../diagrams/rollout_corrected_trio40_b12_dp1_seed0.gif)

All six models start from the same verified 12-frame real prefix. Sampling
uses the cached Flex runner, window 48, 12 frame-denoising steps, four
action-denoising steps, seed 0, and up to 240 generated transitions. Retirement
is immediate after a generated centre cell lacks a head (`death_patience=1`).
The environment uses gradient 0 and noise period 3, matching collection;
other settings are recorded in each source GIF's JSON sidecar. The real prefix
is played by the stored AlphaZero policy with 48 simulations, search depth 6,
and up to 32 joint actions, as configured by the rollout command.

| Checkpoint | Generated transitions | Retirements by agent, measured from end of prefix |
| --- | ---: | --- |
| solo24 | 240 | agent 0 survives; agent 1: 138; agent 2: 152 |
| ego24 | 240 | agent 0: 88; agent 1 survives; agent 2: 66 |
| ceiling24 | 71 | agent 0: 71; agent 1: 5; agent 2: 9 |
| solo40 | 135 | agent 0: 46; agent 1: 29; agent 2: 135 |
| ego40 | 149 | agent 0: 15; agent 1: 149; agent 2: 6 |
| ceiling40 | 62 | agent 0: 62; agent 1: 62; agent 2: 20 |

These are qualitative autonomous examples. Survival length alone is not a
quality score: an implausible trajectory can keep a viewpoint alive, while a
plausible fatal collision can retire it. The separate
[teacher-forced report](../teacher_forced_2026_09_08/report.md) measures a
different task. Equal sampling seeds do not guarantee corresponding random
numbers after populations diverge and consume differently sized draws.

## Corrections

- Current action slots are matched to agent identities. Retirement of a
  middle agent cannot shift a survivor's action into the wrong record.
- Supplied actions are recorded before movement and frame generation. Fixed
  actions condition other action predictions; already-decided actions are
  returned without being sampled again.
- A fatal transition contributes one final observation at the reached
  position. Historical observations remain available until normal eviction;
  there are no later pairs, actions, prediction targets, or repeated death
  observations for that agent. Once everyone retires, no further model
  forward pass is performed.
- The final observation appears once on the reconstructed canvas. The retired
  tile is a frozen, dimmed display of it and never contributes new context.
- Both GIF encoders reserve class and caption colours. A palette derived from
  a middle frame previously changed colours that had disappeared by then;
  source GIF quantization also changed some cell classes. Exact RGB lookup
  now preserves viewpoint tile colours. Noisy/faded canvas colours remain
  approximate because GIF palettes contain at most 256 entries.

The final Q pair's unavailable outgoing-action slot is historical bookkeeping
under the existing pair representation; it is never sampled as an action.
This is not an ongoing masked record for the retired agent. The uncached
runner re-encodes retained history; the cached runner reuses completed context.

## Reproduction and raw artifacts

The generating code is committed at `495ab0c`; the runner correction itself
is `9a7b3ff`. The teacher-forced implementation and its complete raw results
were archived independently at `ab3a410`. See
[rollout notes](../../examples/play/rollout_notes.md) for the command and file
formats. Each individual source GIF has adjacent `.json`, `.prefix.npz`, and
`.rollout.npz` sidecars under `diagrams/rollout_corrected_<arm><budget>_b12_dp1_seed0.*`.
The six checkpoint paths and hashes are recorded in those JSON files.

The sparse raw trajectories contain one row per emitted observation, with
identity, time, position, and outgoing action (`-1` at death or recording end).
They are independent of GIF palette conversion. The exact initial model-input
prefixes are also saved, including known-action flags and life flags.

Validation: 300 tests pass, including 17 new rollout tests and two palette
regressions. [Trajectory verification](verification.json) checks identical
prefixes, action ownership and displacement, one contiguous record per agent,
no post-death entries, hashes, and GIF frame counts.
[Rendering verification](rendering_verification.json) compares every displayed
viewpoint cell with the sparse raw observations, including frozen death tiles,
and every active combined panel with its corresponding source GIF frame.
The accompanying verification scripts reproduce these checks.
