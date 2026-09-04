"""Turning the rectangular training arrays into sets of pairs.

The collected data happens to be neat -- every agent present at every step,
one array per field -- but nothing downstream should depend on that. This
is the one place that knows about the rectangle, and its job is to forget
it: positions are dead reckoned once, here, and travel with the pairs from
then on.
"""
import torch

from marlenv.flex_wm.pairs import PairBatch


def pairs_from_arrays(frames, actions, origins, alive=None, trained=None,
                      model=None, positions=None):
    """A ``PairBatch`` from ``(batch, time, agents, ...)`` arrays.

    ``positions`` may be given directly; otherwise they are dead reckoned
    from the actions the way the older model did it, which is what makes the
    two comparable.
    """
    batch, steps, agents = frames.shape[:3]
    device = frames.device

    if positions is None:
        if model is None:
            raise ValueError('need a model to dead reckon, or positions')
        moves = actions if actions.shape[1] == steps else actions
        positions, _ = model.trajectory(moves, origins, alive,
                                        trailing=moves.shape[1] == steps)

    index = torch.arange(agents, device=device)
    agent = index[None, None].expand(batch, steps, -1)
    stamp = torch.arange(steps, device=device)[None, :, None]
    stamp = stamp.expand(batch, -1, agents)

    flat = lambda x, *rest: x.reshape(batch, steps * agents, *rest)
    valid = (torch.ones(batch, steps, agents, dtype=torch.bool,
                        device=device) if alive is None else alive)
    return PairBatch(
        observations=flat(frames, *frames.shape[3:]),
        actions=flat(actions),
        agent=flat(agent),
        time=flat(stamp),
        position=flat(positions, 2),
        valid=flat(torch.ones_like(valid)),
        trained=flat(valid if trained is None else trained),
        acted=flat(valid))


def unflatten(values, steps, agents):
    """``(batch, steps * agents, ...)`` back to ``(batch, steps, agents,...)``.

    The inverse of how :func:`pairs_from_arrays` laid the rectangle out, for
    comparing against code that still speaks in rectangles.
    """
    return values.reshape(values.shape[0], steps, agents, *values.shape[2:])
