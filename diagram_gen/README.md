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
