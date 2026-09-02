"""Factorised policy/value network shared across snakes.

One network is applied independently to every living snake's view, which is
what makes it identity equivariant and cardinality flexible: there is no
snake-index input and no per-snake parameter, so the same weights serve any
number of snakes in any order.

The per-snake value heads are combined into the communal value by the
:class:`~marlenv.policies.objectives.CommunalObjective`, and that combined
value is the only thing the value loss sees -- individual heads are never
supervised directly, so the decomposition is learned.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marlenv.policies.features import NUM_CHANNELS, head_positions, observe


class _ResidualBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class SnakeNet(nn.Module):
    """Per-snake policy logits and value from a canonicalised grid view.

    Normalisation is GroupNorm rather than BatchNorm because search evaluates
    one position at a time: the batch is the number of living snakes, often
    one, which BatchNorm statistics handle badly.

    The trunk output is read out two ways -- the feature at the snake's own
    head, and the mean over the board -- so the head sees both local and
    global context without a flatten, keeping the network independent of
    grid size.
    """

    def __init__(self, channels=32, blocks=2, hidden=64, num_actions=3,
                 groups=4):
        super().__init__()
        self.num_actions = num_actions
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(channels, groups) for _ in range(blocks)])
        self.readout = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, planes, heads):
        """``(n, C, H, W)`` views plus ``(n, 2)`` head positions.

        Returns policy logits ``(n, num_actions)`` and values ``(n,)``.
        """
        x = self.blocks(self.stem(planes))
        rows, cols = heads[:, 0], heads[:, 1]
        at_head = x[torch.arange(x.shape[0], device=x.device), :, rows, cols]
        pooled = x.mean(dim=(2, 3))
        z = self.readout(torch.cat([at_head, pooled], dim=1))
        return self.policy(z), self.value(z).squeeze(-1)


class NetworkEvaluator:
    """Runs :class:`SnakeNet` on env states for the search.

    Keeps the network in eval mode and off the autograd tape; the search only
    ever needs numbers back.
    """

    def __init__(self, net, device=None):
        self.num_actions = net.num_actions
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.net = net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def evaluate(self, env):
        """Priors and values for every living snake in ``env``.

        Returns ``(indices, priors, values)`` where ``indices`` lists the
        living snakes in order, ``priors`` is ``(n, num_actions)`` and
        ``values`` is ``(n,)``. All three are empty when no snake is alive.
        """
        planes, indices = observe(env)
        if not indices:
            return [], np.zeros((0, self.num_actions), dtype=np.float32), \
                np.zeros((0,), dtype=np.float32)

        heads = head_positions(planes)
        planes_t = torch.from_numpy(planes).to(self.device)
        heads_t = torch.from_numpy(heads).to(self.device)
        logits, values = self.net(planes_t, heads_t)
        priors = torch.softmax(logits, dim=-1)
        return (indices,
                priors.cpu().numpy().astype(np.float32),
                values.cpu().numpy().astype(np.float32))


def batch_from_views(planes, device):
    """Move a stack of views and their head positions onto ``device``."""
    heads = head_positions(planes)
    return (torch.from_numpy(planes).to(device),
            torch.from_numpy(heads).to(device))
