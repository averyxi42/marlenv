"""Keys and values kept per block, with the attributes to mask them by.

A uniform-attention model needs no mask against its cache: everything in it
is strictly earlier, which is exactly what causality permits. Scoped
attention breaks that convenience. An agent-scope block may look only at
its own past, so it has to be able to tell which cached token belongs to
whom, and a frame-scope block reaches no further than the observation it is
part of, so for those blocks the cache holds nothing worth reading at all.

So the cache stores the attributes beside the keys. The saving is real: a
frame-scope block never reads, an agent-scope block reads a third of a
three-agent sequence, and only the global blocks pay in full.
"""
import torch

from marlenv.flex_wm.attention import AGENT, FRAME


class ScopedCache:
    """Per-block keys and values, plus who and when each one was."""

    def __init__(self, layers, device=None):
        self.layers = layers
        self.device = device
        self.keys = [None] * layers
        self.values = [None] * layers
        self.time = None
        self.agent = None
        self.is_action = None

    def __len__(self):
        return 0 if self.time is None else int(self.time.shape[1])

    @property
    def empty(self):
        return len(self) == 0

    def reset(self):
        self.keys = [None] * self.layers
        self.values = [None] * self.layers
        self.time = self.agent = self.is_action = None

    def read(self, layer, key, value, scope):
        """Cached keys and values a block of this scope may see, plus new.

        layer  block index
        key    ``(1, heads, tokens, head_dim)`` for the step being computed
        value  the same
        scope  the block's attention scope

        Returns ``(key, value)`` with the cache prepended, or the arguments
        unchanged for a frame-scope block, which cannot reach the past.
        """
        if scope == FRAME or self.keys[layer] is None:
            return key, value
        return (torch.cat([self.keys[layer], key], dim=2),
                torch.cat([self.values[layer], value], dim=2))

    def write(self, layer, key, value):
        """Append a committed step's keys and values for one block."""
        if self.keys[layer] is None:
            self.keys[layer] = key.detach()
            self.values[layer] = value.detach()
        else:
            self.keys[layer] = torch.cat([self.keys[layer], key.detach()],
                                         dim=2)
            self.values[layer] = torch.cat(
                [self.values[layer], value.detach()], dim=2)

    def commit(self, time, agent, is_action):
        """Record the attributes of the tokens just written."""
        if self.time is None:
            self.time, self.agent = time, agent
            self.is_action = is_action
            return
        self.time = torch.cat([self.time, time], dim=1)
        self.agent = torch.cat([self.agent, agent], dim=1)
        self.is_action = torch.cat([self.is_action, is_action], dim=1)

    def trim(self, oldest):
        """Drop everything older than ``oldest``, in frames."""
        if self.time is None:
            return
        keep = (self.time >= oldest)[0]
        if bool(keep.all()):
            return
        index = torch.nonzero(keep, as_tuple=True)[0]
        for layer in range(self.layers):
            if self.keys[layer] is not None:
                self.keys[layer] = self.keys[layer][:, :, index]
                self.values[layer] = self.values[layer][:, :, index]
        self.time = self.time[:, index]
        self.agent = self.agent[:, index]
        self.is_action = self.is_action[:, index]

    def mask_for(self, scope, time, agent, is_action, window=None):
        """The mask for one block: new tokens against cache plus new.

        scope     the block's attention scope
        time      ``(1, q)`` long, of the tokens being computed
        agent     ``(1, q)`` long
        is_action ``(1, q)`` bool
        window    frames of history, or ``None``

        Returns ``(1, 1, q, k)`` bool, where ``k`` counts the cached tokens
        this scope can see followed by the new ones.
        """
        if scope == FRAME or self.time is None:
            keys = (time, agent, is_action)
        else:
            keys = (torch.cat([self.time, time], dim=1),
                    torch.cat([self.agent, agent], dim=1),
                    torch.cat([self.is_action, is_action], dim=1))
        key_time, key_agent, key_action = keys

        query = time[:, :, None]
        allowed = key_time[:, None, :] <= query
        allowed &= ~((key_time[:, None, :] == query)
                     & key_action[:, None, :] & ~is_action[:, :, None])
        if window is not None:
            allowed &= (query - key_time[:, None, :]) < window
        if scope in (FRAME, AGENT):
            allowed &= key_agent[:, None, :] == agent[:, :, None]
        if scope == FRAME:
            allowed &= key_time[:, None, :] == query
        return allowed[:, None]
