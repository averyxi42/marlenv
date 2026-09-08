# Teacher-forced evaluation of the six existing checkpoints

All six checkpoints completed evaluation on the same 12 simulator recordings
(seeds 2100–2111), each with 60 transitions. No generated observation conditions
another prediction. No model-driven retirement or rollout runner is used.

| Checkpoint | Exact snake consistency | 95% recording-bootstrap interval | Exact snake accuracy vs truth | Snake occupancy consistency |
| --- | ---: | ---: | ---: | ---: |
| solo24 | 32.71% | 29.32–35.37% | 59.89% | 53.36% |
| ego24 | 75.62% | 72.98–78.07% | 80.98% | 90.75% |
| ceiling24 | 88.62% | 86.61–90.49% | 93.68% | 96.37% |
| solo40 | 32.42% | 29.22–35.08% | 60.58% | 51.28% |
| ego40 | 75.79% | 73.43–78.13% | 82.18% | 90.12% |
| ceiling40 | 89.11% | 86.66–91.53% | 94.29% | 96.44% |

The principal ordering survives: egocentric training produces substantially
more consistent predictions than solo training, and the ceiling remains ahead.
However, the solo baseline scores approximately 32%, rather than the near-zero
scores reported by the old autoregressive instrument. The old and new numbers
measure different prediction tasks and are not a correction factor for each
other. Continuations change one-step consistency little in this run.

Every arm scored 720 transitions, 2,062 predicted views, 1,609 overlapping
viewpoint pairs, and 38,827 pair-overlap cells. The real-view overlap control
returned exactly 1.0 throughout. Four predicted views were fatal transitions;
that is insufficient to draw comparative conclusions about death behavior.

## Definition

Exact snake consistency counts exact palette-class matches (identity and
head/body/tail included) on the union of cells where either prediction contains
a snake. The operation is symmetric in the two viewpoints. Both-empty unions
are undefined (`null`), not perfect agreement. Snake occupancy consistency is
intersection-over-union of the two binary snake masks.

The accuracy column applies the same exact-class union metric between each
prediction and its true target, over the full view. It is not ordinary all-cell
accuracy. It checks whether predictions that agree also match the simulator.

Integer counts are pooled before division. The intervals above resample the
12 whole recordings with replacement 20,000 times, using seed 731. They describe
recording variability for these fixed checkpoints and this one sampling seed;
they do not capture variation across model training runs or all diffusion seeds.

## Protocol

- Every transition is evaluated, starting with prediction of frame 1 from real
  frame 0. There is no unscored bootstrap prefix.
- Window 48 means up to 47 true past frames plus the jointly generated target.
  Each query re-encodes this finite window; no KV cache is used.
- All models receive all available real agent histories. All target views
  start unknown. Target positions are derived from the last known positions
  and supplied actions. Future observations, poses, actions, and survival flags
  cannot enter the prediction query.
- Every agent alive before the action receives a target, including one
  aftermath view for fatal transitions. Only recorded life flags determine
  availability afterward.
- Twelve DDIM denoising steps; evaluation sampling seed 0, with a separate RNG
  stream derived from recording seed and target index for every prediction.
- Simulator: 15x15, three snakes, four fruits, 9x9 views, obstacle density 0.12,
  background gradient 0, observation noise 2, snake noise 8, noise period 3.
- Collection: stored AlphaZero policy, 16 simulations, maximum search depth 24,
  four candidate joint actions, no exploration noise. The simulator was run
  once and its recordings reused for every checkpoint.
- Evaluation ran on an NVIDIA GeForce RTX 3090, in approximately 6.6 minutes
  across the six checkpoints.

## Files and implementation

- [Full metrics, per-recording counts and checkpoint hashes](scores.json)
- [Recording manifest, configuration and recording hashes](recordings/manifest.json)
- [Bootstrap intervals](intervals.json)

The implementation lives in `marlenv/grading/teacher_forced.py`, with a separate
`examples/analysis/grade_teacher_forced.py` collection/scoring CLI. Usage and
measurement conventions are documented in `examples/analysis/teacher_forced.md`.
No existing training code or rollout evaluator was modified.

Validation: 15 new tests passed. The pre-existing 266 tests also passed (the
GIF-writing test was run separately with the filesystem access it requires).
The new tests check symmetry, empty unions, exact class identity, alignment,
death positions, persistence, isolation from future data, absence of generated
feedback, clean history throughout denoising, action ownership after a middle
agent dies, window semantics, perfect oracle controls, truth accuracy for empty
predictions, and continued evaluation when predicted heads are absent.
