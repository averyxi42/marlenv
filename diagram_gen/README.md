# diagram_gen

Scripts that draw the pictures used to explain what this repository does.
Generated images go in `diagrams/` at the top level, never next to the code
that made them.

Each script picks its own subject matter rather than taking one on the
command line, because a hand-picked example is an argument and a searched
one is evidence. Say what makes an example worth showing, score every
candidate by it, and draw the winner.

## `ego_timeline.py`

What one agent's account of an episode contains, laid out left to right,
one column per step. The true board sits at the bottom with the observer's
view outlined on it; above that the observer's own record, complete, with
its actions as cardinal arrows; above that every snake it managed to
recover, dimmed wherever it could not see.

Rows are real snakes -- ground truth, which the observer does not have. The
bubbles on a row are what it does have: a fresh identity per visit. Two
bubbles on one row is the same snake counted twice, and that is the point.

```bash
python diagram_gen/ego_timeline.py                 # eight steps
python diagram_gen/ego_timeline.py --steps 12      # more room for re-entries
python diagram_gen/ego_timeline.py --scan 200      # search harder
```

Windows are scored for how much appearing and disappearing they contain, so
`--steps 12` tends to find a snake seen three separate times, while a short
window finds one clean entrance and exit.

## `pair_rollouts.py`

Two rollout gifs tiled side by side under captions. Reading them in separate
windows means holding one in memory while watching the other, which is the
comparison a reader is worst at.

```bash
python diagram_gen/pair_rollouts.py \
    --left diagrams/rollout_ego_step12000.gif \
    --right diagrams/rollout_nograd_step12000.gif \
    --left-label "one agent's record" \
    --right-label "every agent's record" \
    --out diagrams/rollout_pair.gif
```

The sources are left alone; this writes a third file. Where one rollout ends
early its last frame is held, greyed and marked, rather than looping back to
the start -- a loop reads as the rollout continuing, and a bright still
picture beside a moving one reads as one that happens to be sitting quiet.
All frames share a single palette, because per-frame adaptive palettes make
the captions -- identical pixels every frame -- speckle as the text colour
lands on a different entry each time.

### Making the rollouts comparable

Two models can only be read against each other if everything except the
weights is held fixed, and three of those settings are not recorded in a
checkpoint:

```bash
python examples/play/rollout_flex.py \
    --model marlenv/demodata/<run>/model_step12000.pt \
    --checkpoint marlenv/demodata/az_policy.pt \
    --background-gradient 0 \
    --steps 240 --bootstrap 12 --seed 0 \
    --out diagrams/<name>.gif
```

`--background-gradient` defaults to 16.0 and must be `0` for a model trained
on the gradient-free sets, or it is shown a gradient it has never seen.
`--checkpoint` is the search that plays the bootstrap prefix: without it the
prefix comes from random rollouts, which survive but rarely eat, handing the
model a short-snaked history unlike its training data. The same `--seed`
gives both models the same board and the same prefix.
