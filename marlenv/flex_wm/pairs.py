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
    visible       ``(b, p, tokens)`` the patch was observed, defaults to
                  all of them

    ``trained`` and ``visible`` say the same thing at two granularities and
    compose by conjunction: ``trained`` rules a whole observation in or out,
    ``visible`` rules single patches of one out. An unseen patch is pinned
    at the top of the noise schedule and kept out of the loss -- it exists,
    at a known place, with unknown content and no truth to regress on.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    agent: torch.Tensor
    time: torch.Tensor
    position: torch.Tensor
    valid: torch.Tensor = None
    trained: torch.Tensor = None
    acted: torch.Tensor = None
    visible: torch.Tensor = None

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
        if self.visible is None:
            # shaped lazily: the token count belongs to the model, not here
            self.visible = None

    @property
    def batch(self):
        return self.observations.shape[0]

    @property
    def pairs(self):
        return self.observations.shape[1]

    def patch_mask(self, tokens):
        """``(b, p, tokens)`` of which patches were observed."""
        if self.visible is None:
            return torch.ones(self.batch, self.pairs, tokens,
                              dtype=torch.bool,
                              device=self.observations.device)
        return self.visible

    def cell_mask(self, tokens, view):
        """``(b, p, view, view)``: a patch's flag spread over its cells."""
        grid = int(round(tokens ** 0.5))
        patch = view // grid
        flags = self.patch_mask(tokens).reshape(self.batch, self.pairs,
                                                grid, grid)
        return flags.repeat_interleave(patch, dim=2).repeat_interleave(
            patch, dim=3)

    def to(self, device):
        names = ('observations', 'actions', 'agent', 'time', 'position',
                 'valid', 'trained', 'acted', 'visible')
        moved = {name: (None if getattr(self, name) is None
                        else getattr(self, name).to(device))
                 for name in names}
        return PairBatch(**moved)


def compact(pairs, keep):
    """Drop the pairs ``keep`` is false for, repacking what is left.

    pairs ``PairBatch`` of ``(b, p)``
    keep  ``(b, p)`` bool

    Returns a ``PairBatch`` of ``(b, p')`` where ``p'`` is the largest
    number surviving in any row; rows with fewer are padded and marked
    invalid.

    This is how an agent leaves. Retiring it by pinning its tokens at the
    top of the noise schedule would keep them in the sequence, and a token
    that is present is a token the model can learn to read -- position and
    count alone say something, even when the content says nothing. Removing
    them closes that channel, and it is the same thing a rollout does when
    an agent stops being simulated, so the two agree by construction rather
    than by careful arrangement.
    """
    import torch as _torch

    keep = keep & pairs.valid
    counts = keep.sum(dim=1)
    width = int(counts.max().clamp(min=1))
    batch = pairs.batch
    device = pairs.observations.device

    # a stable gather: surviving pairs keep their order, padding goes last
    order = _torch.argsort(~keep, dim=1, stable=True)[:, :width]
    slot = _torch.arange(width, device=device)[None]
    live = slot < counts[:, None]

    def take(values, fill=0):
        index = order.reshape(batch, width, *([1] * (values.dim() - 2)))
        gathered = _torch.gather(
            values, 1, index.expand(-1, -1, *values.shape[2:]))
        spread = live.reshape(batch, width, *([1] * (values.dim() - 2)))
        return _torch.where(spread, gathered,
                            _torch.full_like(gathered, fill))

    return PairBatch(
        observations=take(pairs.observations),
        actions=take(pairs.actions),
        # an identity no real pair carries, so padding matches nothing
        agent=take(pairs.agent, fill=-1),
        time=take(pairs.time),
        position=take(pairs.position),
        valid=live,
        trained=take(pairs.trained) & live,
        acted=take(pairs.acted) & live)


def token_attributes(pairs, tokens_per_frame):
    """Per-token ``(time, agent, is_action, valid)``, in pair-major order.

    pairs            ``PairBatch`` of ``(b, p)``
    tokens_per_frame patches an observation is cut into

    Each returned tensor is ``(b, p * (tokens_per_frame + 1))``; ``time``
    and ``agent`` are long, the rest bool.

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
    """Rotary coordinates per token.

    pairs   ``PairBatch`` of ``(b, p)``
    offsets ``(tokens_per_frame, 2)`` cell offset of each patch centre from
            the observation's own position

    Returns ``(b, p * (tokens_per_frame + 1), 3)`` long, of time, row, col.

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
