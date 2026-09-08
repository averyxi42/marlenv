# Teacher-forced consistency evaluation

`grade_teacher_forced.py` is independent of the existing rollout runners and
the old consistency evaluator. It has two stages: record simulator truth once,
then evaluate every model on those same saved transitions.

## Record

From the repository root:

```bash
python examples/analysis/grade_teacher_forced.py collect \
  --policy marlenv/demodata/az_policy.pt \
  --episodes 12 --steps 60 --seed 2100 \
  --background-gradient 0 --noise-period 3 \
  --out /tmp/teacher-forced-recordings
```

The defaults match the gradient-free collection settings: a 15x15 board,
three snakes, four fruits, view radius 4, obstacle density 0.12, observation
noise 2, snake noise 8, and noise period **3**. The policy uses 16 search
simulations, maximum depth 24, and four candidate joint actions. These search
settings are recorded explicitly; a policy checkpoint alone does not specify
a collection policy.

Each NPZ contains north-up uint8 views, actual simulator positions including
death positions, life flags, and cardinal outgoing actions. An already dead
agent has action -1. There are T action rows and T+1 observation rows, with
no fabricated terminal action. Geometry is checked against every actual
move. Metadata records environment/search settings, episode seeds, policy
hash, and runtime versions. A manifest lists the recordings and SHA-256 hashes.

The new collector does not import `marlenv.data` or use its pose conventions.
It refuses to overwrite a nonempty recording directory.

## Score

```bash
python examples/analysis/grade_teacher_forced.py score \
  --recordings /tmp/teacher-forced-recordings \
  --models \
    marlenv/demodata/flex_solo/model_step24000.pt \
    marlenv/demodata/flex_ego/model_step24000.pt \
    marlenv/demodata/flex_nograd/model_step24000.pt \
    marlenv/demodata/flex_solo_deaths/model_step16000.pt \
    marlenv/demodata/flex_ego_deaths/model_step16000.pt \
    marlenv/demodata/flex_nograd_warm/model_step16000.pt \
  --names solo24 ego24 ceiling24 solo40 ego40 ceiling40 \
  --window 48 --denoise-steps 12 --device cuda \
  --out /tmp/teacher-forced-scores.json
```

The evaluator supports world-frame flex checkpoints with four cardinal
actions, including all six experiment arms. All arms receive the same real
multi-agent history during evaluation. The solo arm remains the flex model
trained on solo samples; it is not the older single-agent model.

For target frame t, conditioning consists of recorded observations through
t-1 and recorded actions through t-1. `--window 48` means at most **47 real
history frames plus one target frame**. Full forward passes re-encode this
finite context at every denoising step; no cache is used. Every target is a
fresh query, so generated views cannot affect later predictions. Query RNG
seeds depend on the evaluation seed, recording seed, and target index, not
checkpoint order or earlier predictions.

All target views start as noise and are denoised jointly. No real target
image, future action, future pose, or future survival flag is used to build
the query. Target positions are computed from the last real positions and
the supplied outgoing actions. Target action slots are unknown: an action
from frame t is not supplied when predicting frame t.

Targets include every agent alive **before** the action. Thus a fatal
transition receives one aftermath prediction and contributes to consistency
when it overlaps another target. The recording determines subsequent
availability. A real aftermath view can appear once in later conditioning,
without an outgoing action. There is no predicted retirement threshold.

By default every transition is scored, including the cold start from one
real frame. `--first-target K` excludes targets before K from scoring while
retaining their real observations as conditioning; it does not bootstrap
with generated observations.

## Measurements

Every unordered target pair is aligned directly using its world-coordinate
view footprint. Disjoint pairs contribute nothing. In the overlap, let A
and B be the decoded class grids and S denote a snake class:

```
snake consistency = count(A == B and (S(A) or S(B)))
                    / count(S(A) or S(B))
```

Exact agreement requires the same snake colour identity and the same
head/body/tail class. The denominator is a union, not just snake cells from
the first view. Swapping the pair therefore leaves every reported count and
rate unchanged. An empty denominator is JSON `null`, never perfect agreement.

Reports include:

- **Consistency:** exact snake agreement, snake occupancy IoU, and agreement
  across all classes, computed between predictions.
- **Truth control:** the same overlap calculation between recorded views.
  Any class disagreement raises an error rather than producing model scores
  from an invalid reference alignment.
- **Accuracy:** the same cell statistics between each predicted view and its
  real target. This exposes mutually consistent but wrong predictions.
- **Death accuracy:** accuracy restricted to fatal transitions.
- Counts of predictions, fatal transitions, overlapping pairs, and cells.

Totals sum integer numerators and denominators before division; they do not
average pair-level ratios. Per-recording counts are also saved so uncertainty
can be estimated at the recording level rather than treating correlated
overlapping pixels as independent samples. One stochastic sample is drawn
per query; repeating evaluation seeds measures sampling variability.

The JSON includes checkpoint hashes, model schedules, training context/window
metadata when available, evaluation settings, and the recording manifest.
Completed checkpoint results are written after each model.

These are one-step, truth-conditioned results. They do not measure autonomous
rollout survival or accumulation of prediction errors, and are not directly
comparable with the old README's action-conditioned rollout scores.

## Tests

```bash
pytest tests/test_teacher_forced.py
```

Tests cover the symmetric union denominator, class identity, empty unions,
world-cell alignment, valid death poses, recording round trips, future-data
isolation, absence of prediction feedback, clean conditioning throughout
denoising, action ownership after a middle agent dies, window boundaries,
oracle truth controls, empty-prediction accuracy, pair-order invariance,
continued evaluation despite missing predicted heads, and count aggregation.
