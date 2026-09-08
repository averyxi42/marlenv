# 40k autonomous rollouts without interior obstacles

[Paired solo / egocentric / ceiling GIF](../../diagrams/rollout_no_obstacles_trio40_b12_dp1_seed0.gif)

Only `--obstacle-density` changes, from `0.12` to `0`. The outer boundary
remains. This changes the real simulator prefix; subsequent model predictions
are generated freely, without suppressing predicted walls or changing retirement.

All three runs use the same newly collected 12-frame prefix, seed 0, a
240-transition generation limit, cached runner, window 48, 12 frame-denoising
steps, four action-denoising steps, and immediate retirement after a missing
head. Checkpoints, policy, search settings, and noise settings match the prior
40k runs. No training or inference code was changed.

| Model | Previous generated transitions | New generated transitions | Viewpoints alive at new endpoint |
| --- | ---: | ---: | ---: |
| solo | 135 | 131 | 0 |
| ego | 149 | 240 | 1 |
| ceiling | 62 | 240 | 2 |

The egocentric and ceiling recordings reach the generation limit. Solo
ends when its last viewpoint retires. These are the seed-0 recordings,
without searching for a favourable example.

Each source GIF has adjacent `.json`, `.prefix.npz`, and `.rollout.npz`
sidecars under `diagrams/rollout_no_obstacles_<arm>40_b12_dp1_seed0.*`.
They record complete settings and hashes, exact prefixes, and sparse
emitted observations, identities, positions, and outgoing actions.
A death contributes one final observation and no later records.

Generation used committed source `3937b24`. [Trajectory checks](verification.json)
verify matching prefixes, the single changed setting, unchanged model and
policy hashes, correct action ownership and displacement, and no post-death
records. [Rendering checks](rendering_verification.json) verify exact tile
colours and paired frame alignment. The accompanying scripts reproduce
these checks.

## Short consistency comparison

[Slice ending when any rollout first has one snake alive](../../diagrams/rollout_no_obstacles_trio40_b12_dp1_seed0_until_one_alive.gif)

The slice includes generated frames 0–33 (34 frames, 5.44 seconds). At its
last frame, solo falls from two surviving viewpoints to one; egocentric and
ceiling still have three each. This is simulation step 44, after the 11 real
prefix transitions. The original full GIF remains available above.

The slice copies the original GIF blocks through that frame without
re-encoding, changing timing, or adding a final hold. Every decoded pixel
and frame duration was checked against the source. See
[slice metadata](slice_until_one_alive.json) and the accompanying reproduction
script.
