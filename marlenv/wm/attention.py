"""Attention backends and the mask the world model needs.

The mask is block-causal with one exception: the patch tokens of a single
frame attend to each other freely, because they are one observation being
denoised jointly and imposing an order on them would be arbitrary. An action
token conditions the *next* frame, so its own frame's patches may not see it.

Two paths exist and must agree.

``full``
    the whole sequence at once, used for training. FlexAttention expresses
    the mask directly as a predicate rather than materialising an N-by-N
    boolean, and falls back to SDPA with a dense mask where FlexAttention is
    unavailable.
``incremental``
    one frame at a time against a KV cache, used for play. Here no mask is
    needed at all: everything in the cache is strictly earlier in time, and
    the new tokens are patches of one frame, which may see each other. That
    is a property worth stating, because it is what makes the fast path both
    simple and provably equal to the slow one.
"""
import functools

import torch
import torch.nn.functional as F

try:  # FlexAttention landed in torch 2.5
    from torch.nn.attention.flex_attention import (create_block_mask,
                                                   flex_attention)
    FLEX_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed torch
    FLEX_AVAILABLE = False


def mask_predicate(time, is_action, window=None):
    """The rule both paths implement, as a function of token indices.

    ``window`` limits how far back a token may look, in frames; ``None``
    means the whole sequence.
    """
    def predicate(batch, head, q_index, kv_index):
        q_time, kv_time = time[q_index], time[kv_index]
        allowed = kv_time <= q_time
        # an action conditions the next frame, so its own frame cannot see it
        same_frame = kv_time == q_time
        allowed = allowed & ~(same_frame & is_action[kv_index]
                              & ~is_action[q_index])
        if window is not None:
            allowed = allowed & (q_time - kv_time < window)
        return allowed
    return predicate


def dense_mask(time, is_action, window=None):
    """The same rule as an explicit boolean matrix, for the SDPA fallback."""
    allowed = time[None, :] <= time[:, None]
    same_frame = time[None, :] == time[:, None]
    allowed = allowed & ~(same_frame & is_action[None, :]
                          & ~is_action[:, None])
    if window is not None:
        allowed = allowed & ((time[:, None] - time[None, :]) < window)
    return allowed[None, None]


#: FlexAttention only pays off once compiled; uncompiled it materialises the
#: whole score matrix and measured 1.5x slower than SDPA at our lengths, so it
#: is opt-in rather than the default.
USE_FLEX = False

_compiled_flex = None


def _flex():
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex


@functools.lru_cache(maxsize=32)
def _cached_mask(key, tokens, window, device, flex):
    time, is_action = _MASK_INPUTS[key]
    if flex:
        predicate = mask_predicate(time, is_action, window)
        return create_block_mask(predicate, B=None, H=None, Q_LEN=tokens,
                                 KV_LEN=tokens, device=device)
    return dense_mask(time, is_action, window)


_MASK_INPUTS = {}


def build_mask(time, is_action, window=None, prefer_flex=None):
    """A mask object suitable for :func:`attend`, cached by shape.

    The mask depends only on the sequence length and window, never on the
    batch, so rebuilding it every forward was pure overhead -- and building a
    BlockMask is not cheap.
    """
    tokens = int(time.shape[0])
    flex = (USE_FLEX if prefer_flex is None else prefer_flex)
    flex = flex and FLEX_AVAILABLE and time.device.type == 'cuda'
    key = (tokens, bool(flex))
    _MASK_INPUTS[key] = (time, is_action)
    return _cached_mask(key, tokens, window, time.device, flex)


def attend(query, key, value, mask):
    """Dispatch to FlexAttention or SDPA depending on the mask type."""
    if FLEX_AVAILABLE and mask is not None and not torch.is_tensor(mask):
        return _flex()(query, key, value, block_mask=mask)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
