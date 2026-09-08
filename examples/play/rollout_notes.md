# Corrected autonomous rollouts

The corrected FlexRunner and CachedFlexRunner use pair identities to attach
outgoing actions. Removing agent 1 cannot attach agent 2's action to agent 1's
final observation. Both externally supplied actions and sampled actions are
recorded before movement and frame generation. Fixed actions condition the
other action predictions at noise level zero.

A fatal transition emits one final observation at the reached position. The
agent then contributes no subsequent pairs or prediction targets. Its past,
including that observation, remains historical context until window eviction.
The final observation has no outgoing action; its existing Q-pair action slot
is unavailable, as in training. This is not a repeated death observation or a
sequence of masked post-death placeholders. The uncached runner re-encodes
historical context; the cached runner encodes completed context for reuse.

The GIF freezes the retired agent's tile at its final observation, dimmed and
labelled. That tile is display state only. The final observation is pasted onto
the canvas once, at emission; it is not pasted again on later steps.

## Reproduction

For the six stored checkpoints, pass both `--background-gradient 0` and
`--noise-period 3`. The original rollout command used the simulator's default
noise period of 8, unlike collection. The command now exposes that setting.

```bash
python examples/play/rollout_flex.py \
  --model marlenv/demodata/flex_ego/model_step24000.pt \
  --checkpoint marlenv/demodata/az_policy.pt \
  --background-gradient 0 --noise-period 3 \
  --steps 240 --bootstrap 12 --seed 0 --death-patience 1 \
  --window 48 --denoise-steps 12 --action-steps 4 --device cuda \
  --out diagrams/rollout_corrected_ego24_b12_dp1_seed0.gif
```

Each GIF has three sidecars:

- `.json`: settings, actual prefix hash, checkpoint/policy hashes, source
  revision, retirement times and artifact hashes.
- `.prefix.npz`: exact model-input history at the start of autonomous play.
- `.rollout.npz`: sparse emitted observations and their positions/identities/
  times/actions, starting with the retained prefix. No row follows an agent's
  death. Action `-1` means no outgoing action, either death or recording end.
  Images are uint8 north-up RGB before GIF palette quantization.

Matching prefix hashes verify that the models started from identical data.
Matching diffusion seeds give paired initial randomness; after populations
or trajectory lengths diverge, they need not consume corresponding draws.
These are autonomous qualitative examples, not the teacher-forced instrument.
