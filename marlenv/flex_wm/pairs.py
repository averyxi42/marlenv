"""The observation/action pair, and a batch of them.

A Q pair is one observation and the action taken from it, by one agent at
one step. An episode is a *set* of these and nothing more -- no agent axis,
no fixed count per step, no requirement that every agent appear at every
time. Whatever structure the data happens to have is expressed as
attributes on the pairs rather than as the shape of an array, which is what
lets the agent count move during an episode: agents dying, respawning, or
being seized from the background policy are all just pairs appearing and
disappearing from the set.

Everything a pair needs travels with the pair. In particular position does:
it is where the observation was taken from, and it is the pair's property,
not a lookup into some per-agent table. Agent identity is an attribute too,
and only ever compared, so ids may be sparse and there may be far more of
them across an episode than are ever live at once.

Padding exists only because a batch is a rectangle. ``valid`` says which
pairs are real, and the masks and the loss both read it.
"""
from dataclasses import dataclass

import torch


@dataclass
class PairBatch:
    """``(batch, pairs)`` of observation/action pairs.

    observations  ``(b, p, view, view, channels)``
    actions       ``(b, p)`` action indices, or ``(b, p, n)`` as a signal
    agent         ``(b, p)`` identity, compared and never indexed by
    time          ``(b, p)`` step the observation was taken at
    position      ``(b, p, 2)`` where it was taken from
    valid         ``(b, p)`` real pair, as opposed to padding
    trained       ``(b, p)`` the frame is a target, defaults to ``valid``
    acted         ``(b, p)`` the action is a target, defaults to ``valid``
    """

    observations: torch.Tensor
    actions: torch.Tensor
    agent: torch.Tensor
    time: torch.Tensor
    position: torch.Tensor
    valid: torch.Tensor = None
    trained: torch.Tensor = None
    acted: torch.Tensor = None

    def __post_init__(self):
        shape = self.observations.shape[:2]
        ones = torch.ones(shape, dtype=torch.bool,
                          device=self.observations.device)
        if self.valid is None:
            self.valid = ones
        if self.trained is None:
            self.trained = self.valid
        if self.acted is None:
            self.acted = self.valid

    @property
    def batch(self):
        return self.observations.shape[0]

    @property
    def pairs(self):
        return self.observations.shape[1]

    def to(self, device):
        moved = {name: getattr(self, name).to(device)
                 for name in ('observations', 'actions', 'agent', 'time',
                              'position', 'valid', 'trained', 'acted')}
        return PairBatch(**moved)


def token_attributes(pairs, tokens_per_frame):
    """Per-token ``(time, agent, is_action, valid)``, in pair-major order.

    Each pair lays down its patch tokens and then its action token, so a
    pair occupies a contiguous run. The order is a convenience for reading
    results back out; it carries no meaning, because every rule downstream
    is written against these attributes rather than against position in the
    sequence.
    """
    width = tokens_per_frame + 1
    repeat = lambda x: x[:, :, None].expand(-1, -1, width).reshape(
        x.shape[0], -1)

    is_action = torch.zeros(width, dtype=torch.bool,
                            device=pairs.time.device)
    is_action[-1] = True
    is_action = is_action[None, None].expand(pairs.batch, pairs.pairs,
                                             -1).reshape(pairs.batch, -1)
    return (repeat(pairs.time), repeat(pairs.agent), is_action,
            repeat(pairs.valid))


def token_coords(pairs, offsets):
    """``(batch, tokens, 3)`` of time, row and column.

    A patch sits at its own offset from the observation's position; the
    action sits at the position itself, which is the same cell the central
    patch is centred on -- an action and the view it was chosen from happen
    in the same place.
    """
    batch, count = pairs.batch, pairs.pairs
    spatial = pairs.position[:, :, None, :] + offsets[None, None]
    spatial = torch.cat([spatial, pairs.position[:, :, None, :]], dim=2)
    stamp = pairs.time[:, :, None, None].expand(-1, -1, spatial.shape[2], -1)
    return torch.cat([stamp, spatial], dim=-1).reshape(batch, -1, 3).long()
