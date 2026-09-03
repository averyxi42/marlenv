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


def build_mask(time, is_action, window=None, prefer_flex=True):
    """A mask object suitable for :func:`attend`.

    Returns a FlexAttention ``BlockMask`` when available, otherwise a dense
    boolean tensor. Both describe the same rule.
    """
    tokens = time.shape[0]
    if prefer_flex and FLEX_AVAILABLE and time.device.type == 'cuda':
        predicate = mask_predicate(time, is_action, window)
        return create_block_mask(predicate, B=None, H=None, Q_LEN=tokens,
                                 KV_LEN=tokens, device=time.device)
    return dense_mask(time, is_action, window)


def attend(query, key, value, mask):
    """Dispatch to FlexAttention or SDPA depending on the mask type."""
    if FLEX_AVAILABLE and not torch.is_tensor(mask) and mask is not None:
        return flex_attention(query, key, value, block_mask=mask)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
