"""Attention scopes, as masks over token attributes rather than positions.

The world model's rule has always been the same: a token may look at what
came before it, the patches of one observation may look at each other
freely because they are one picture being denoised jointly, and an action
may not be seen by the observation it was taken from. What changes here is
*how far* that rule reaches, and it changes per layer.

    frame   only within one observation -- the same agent at the same step
    agent   only within one agent's own history
    global  everywhere the base rule allows, which is the older behaviour

Each adds a conjunct to the one below it. A conjunct on the predicate is an
intersection on the set of permitted pairs, so more conjuncts permit less:
global is the base rule, agent is that and same-identity, frame is that and
same-step. The permitted sets therefore nest the other way round from the
list of conditions -- frame within agent within global. Alternating them
gives a
model that can consolidate a single view, follow one agent through time
without another agent's tokens sitting in the way, and still share what
everyone has seen -- without an agent-identity embedding, because a mask
built from identity *equality* is permutation equivariant in a way a
learned per-agent vector is not.

That last property is also why nothing here indexes by agent id. Ids are
compared, never used as offsets, so an episode may contain far more of them
than are ever active at once -- respawns, teams, agents seized mid-episode.
"""
import torch

FRAME, AGENT, GLOBAL = 'F', 'A', 'G'
SCOPES = (FRAME, AGENT, GLOBAL)


def parse_schedule(spec, depth):
    """One scope per block, from a string like ``FAGFAGAAGAAG`` or a list.

    A short schedule repeats, so ``AG`` covers any even depth and ``G``
    reproduces the older uniform behaviour at any depth.
    """
    if isinstance(spec, str):
        letters = [c for c in spec.upper() if not c.isspace()]
    else:
        letters = [str(s).upper()[:1] for s in spec]
    if not letters:
        raise ValueError('empty attention schedule')
    bad = sorted({c for c in letters if c not in SCOPES})
    if bad:
        raise ValueError(f'unknown attention scope(s) {bad}; '
                         f'expected any of {list(SCOPES)}')
    if depth % len(letters):
        raise ValueError(f'schedule of {len(letters)} does not tile a depth '
                         f'of {depth}')
    return [letters[i % len(letters)] for i in range(depth)]


def base_rule(time, is_action, window=None):
    """Causality, plus the two exceptions the model has always had.

    time       ``(batch, tokens)`` long, the step each token belongs to
    is_action  ``(batch, tokens)`` bool, action token rather than a patch
    window     frames of history a token may reach back over, or ``None``

    Returns ``(batch, 1, tokens, tokens)`` bool, broadcast over heads, where
    entry ``[b, 0, q, k]`` says whether query ``q`` may attend to key ``k``.

    The window counts *frames*, not tokens, and it is the only conjunct here
    that varies with scope in effect rather than in form. Under frame scope
    query and key share a step, so the difference is always zero and the
    window never bites. Under agent scope a window of W admits W of that
    agent's own observations; under global scope the same W admits W steps
    of every agent at once, which is many more tokens for the same number.
    """
    query, key = time[:, :, None], time[:, None, :]
    allowed = key <= query
    # an action conditions the *next* observation, so the observation it was
    # taken from must not see it
    allowed &= ~((key == query) & is_action[:, None, :]
                 & ~is_action[:, :, None])
    if window is not None:
        allowed &= (query - key) < window
    return allowed[:, None]


def scope_mask(scope, time, agent, is_action, window=None, valid=None):
    """The base rule, narrowed to one scope.

    scope      one of ``FRAME``, ``AGENT``, ``GLOBAL``
    time       ``(batch, tokens)`` long
    agent      ``(batch, tokens)`` long, compared for equality only
    is_action  ``(batch, tokens)`` bool
    window     frames of history, or ``None`` for unlimited
    valid      ``(batch, tokens)`` bool, real token rather than padding

    Returns ``(batch, 1, tokens, tokens)`` bool.

    Padding is never attended to, and a padded query keeps its own diagonal
    so that no row is entirely empty -- an all-false row would send softmax
    through a division by zero.
    """
    if scope not in SCOPES:
        raise ValueError(f'unknown attention scope {scope!r}')
    allowed = base_rule(time, is_action, window)

    if scope in (FRAME, AGENT):
        same_agent = agent[:, :, None] == agent[:, None, :]
        allowed = allowed & same_agent[:, None]
    if scope == FRAME:
        same_time = time[:, :, None] == time[:, None, :]
        allowed = allowed & same_time[:, None]

    if valid is not None:
        allowed = allowed & valid[:, None, None, :]
        rows = torch.eye(time.shape[1], dtype=torch.bool,
                         device=time.device)[None, None]
        allowed = allowed | (~valid[:, None, :, None] & rows)
    return allowed


def build_masks(schedule, time, agent, is_action, window=None, valid=None):
    """One mask per distinct scope in ``schedule``, keyed by scope.

    Built once and shared by every block that uses it: a twelve block model
    needs three masks, not twelve.
    """
    return {scope: scope_mask(scope, time, agent, is_action, window, valid)
            for scope in dict.fromkeys(schedule)}
