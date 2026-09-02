"""Communal reward functions, paired with a differentiable counterpart.

The search folds a step's per-snake reward vector into one number, and the
network folds its per-snake value predictions the same way. Both have to be
the *same* function or the value head is not learning what the search scores,
so they are defined together here.
"""


class CommunalObjective:
    """A communal reward function and its differentiable twin.

    Parameters
    ----------
    name : str
        Identifier, used in checkpoints and logs.
    fold : callable
        ``fold(sequence_of_floats) -> float``, applied to env rewards.
    torch_fold : callable
        ``torch_fold(values, alive_mask) -> tensor`` of shape ``(batch,)``,
        applied to per-agent value predictions of shape ``(batch, n_agents)``.
        Dead agents are padded with 0 by the caller and must be excluded from
        aggregations where 0 is not the identity element.
    """

    def __init__(self, name, fold, torch_fold):
        self.name = name
        self.fold = fold
        self.torch_fold = torch_fold

    def __call__(self, rewards):
        return self.fold(rewards)

    def __repr__(self):
        return f'CommunalObjective({self.name!r})'


def _torch_sum(values, alive):
    return (values * alive).sum(dim=-1)


def _torch_mean(values, alive):
    count = alive.sum(dim=-1).clamp(min=1.0)
    return (values * alive).sum(dim=-1) / count


def _torch_min(values, alive):
    # dead agents must not win the min, so push them out of the way
    masked = values.masked_fill(alive <= 0, float('inf'))
    out = masked.min(dim=-1).values
    return out.masked_fill(alive.sum(dim=-1) <= 0, 0.0)


def _torch_max(values, alive):
    masked = values.masked_fill(alive <= 0, -float('inf'))
    out = masked.max(dim=-1).values
    return out.masked_fill(alive.sum(dim=-1) <= 0, 0.0)


def _mean(rewards):
    rewards = list(rewards)
    return sum(rewards) / len(rewards) if rewards else 0.0


OBJECTIVES = {
    'sum': CommunalObjective('sum', sum, _torch_sum),
    'mean': CommunalObjective('mean', _mean, _torch_mean),
    'min': CommunalObjective('min', min, _torch_min),
    'max': CommunalObjective('max', max, _torch_max),
}


def get_objective(name_or_objective):
    """Look up a built-in objective by name, or pass one through."""
    if isinstance(name_or_objective, CommunalObjective):
        return name_or_objective
    try:
        return OBJECTIVES[name_or_objective]
    except KeyError:
        raise KeyError(f'unknown objective {name_or_objective!r}; '
                       f'choose from {sorted(OBJECTIVES)} or pass a '
                       f'CommunalObjective') from None
