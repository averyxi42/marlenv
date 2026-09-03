"""Causal Diffusion Forcing Transformer over head-frame observations.

Token layout per step is ``[patch tokens of frame t] [action token t]``,
repeated. Attention is causal across steps, with the patch tokens of one
frame free to attend to each other: they are a single observation being
denoised jointly, and forcing an order on them would be arbitrary.

Positions use axial RoPE over (time, height, width). Action tokens sit at
the spatial origin and carry a learned type embedding, so the model can tell
conditioning from content without a separate stream.

Each frame carries its own diffusion level, which is what makes this
diffusion *forcing* rather than plain video diffusion: with independent
levels per frame, the model learns to denoise the present given a history at
any noise level, including the clean history it gets during rollout.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def axial_rope_frequencies(dim, positions, base=10_000.0):
    """``(..., dim)`` cos/sin pair for one axis of RoPE."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(base) * torch.arange(half, device=positions.device) / half)
    angles = positions[..., None].float() * freqs
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Rotate pairs of channels; ``x`` is ``(batch, heads, tokens, dim)``."""
    a, b = x[..., 0::2], x[..., 1::2]
    cos = cos[..., : a.shape[-1]]
    sin = sin[..., : a.shape[-1]]
    out = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return out.flatten(-2)


class AxialRope(nn.Module):
    """RoPE split across time, height and width."""

    def __init__(self, head_dim):
        super().__init__()
        # split the head into three axes; time gets the remainder
        per_axis = head_dim // 3
        self.splits = (head_dim - 2 * (per_axis // 2 * 2),
                       per_axis // 2 * 2, per_axis // 2 * 2)

    def forward(self, coords):
        """``coords`` is ``(tokens, 3)`` of time, row, col."""
        parts = []
        for axis, size in enumerate(self.splits):
            cos, sin = axial_rope_frequencies(size, coords[:, axis])
            parts.append((cos, sin, size))
        cos = torch.cat([c for c, _, _ in parts], dim=-1)
        sin = torch.cat([s for _, s, _ in parts], dim=-1)
        return cos[None, None], sin[None, None]


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, mask):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(out.transpose(1, 2).reshape(batch, tokens, -1))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        return x + self.mlp(self.norm2(x))


def timestep_embedding(values, dim, max_period=10_000.0):
    """Sinusoidal embedding of a continuous noise level."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device).float() / half)
    args = values.float()[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class WorldModel(nn.Module):
    """Predicts the noise on each frame's patches, given the past."""

    def __init__(self, view=9, patch=3, channels=3, num_actions=3,
                 dim=256, depth=6, heads=8):
        super().__init__()
        if view % patch:
            raise ValueError('view size must divide into whole patches')
        self.view, self.patch = view, patch
        self.grid = view // patch
        self.tokens_per_frame = self.grid ** 2
        self.patch_dim = patch * patch * channels
        self.dim = dim

        self.to_tokens = nn.Linear(self.patch_dim, dim)
        self.action_embedding = nn.Embedding(num_actions, dim)
        self.type_embedding = nn.Embedding(2, dim)
        self.tau_embedding = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.rope = AxialRope(dim // heads)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.to_noise = nn.Linear(dim, self.patch_dim)

    # ---------------------------------------------------------------- tokens
    def patchify(self, frames):
        """``(b, t, v, v, c)`` to ``(b, t, tokens, patch_dim)``."""
        b, t = frames.shape[:2]
        x = frames.permute(0, 1, 4, 2, 3)
        x = x.reshape(b, t, -1, self.grid, self.patch, self.grid, self.patch)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        return x.reshape(b, t, self.tokens_per_frame, self.patch_dim)

    def unpatchify(self, tokens):
        """Inverse of :meth:`patchify`."""
        b, t = tokens.shape[:2]
        x = tokens.reshape(b, t, self.grid, self.grid, -1, self.patch,
                           self.patch)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        x = x.reshape(b, t, -1, self.view, self.view)
        return x.permute(0, 1, 3, 4, 2)

    def token_coords(self, steps, device):
        """``(tokens, 3)`` of time, row, col for one interleaved sequence."""
        rows = torch.arange(self.grid, device=device)
        grid_r, grid_c = torch.meshgrid(rows, rows, indexing='ij')
        spatial = torch.stack([grid_r.reshape(-1), grid_c.reshape(-1)], -1)

        coords = []
        for step in range(steps):
            time = torch.full((self.tokens_per_frame, 1), step, device=device)
            coords.append(torch.cat([time, spatial], dim=-1))
            if step < steps - 1:
                # the action sits at the spatial origin of its own step
                coords.append(torch.tensor([[step, 0, 0]], device=device))
        return torch.cat(coords).long()

    def attention_mask(self, steps, device):
        """Causal across steps, bidirectional inside one observation."""
        coords = self.token_coords(steps, device)
        time = coords[:, 0]
        is_action = self.token_types(steps, device) == 1

        # a token may see anything from an earlier step
        allowed = time[None, :] <= time[:, None]
        # ... but an action token is conditioning for the *next* frame, so
        # tokens of the same step may see it only if they are the action
        same_step = time[None, :] == time[:, None]
        allowed &= ~(same_step & is_action[None, :] & ~is_action[:, None])
        return allowed[None, None]

    def token_types(self, steps, device):
        types = []
        for step in range(steps):
            types.append(torch.zeros(self.tokens_per_frame, device=device))
            if step < steps - 1:
                types.append(torch.ones(1, device=device))
        return torch.cat(types).long()

    # --------------------------------------------------------------- forward
    def forward(self, noisy_frames, actions, tau):
        """Predict the noise on every frame.

        ``noisy_frames`` is ``(b, t, v, v, c)``, ``actions`` ``(b, t - 1)``
        and ``tau`` ``(b, t)``, one noise level per frame.
        """
        batch, steps = noisy_frames.shape[:2]
        device = noisy_frames.device

        patches = self.to_tokens(self.patchify(noisy_frames))
        level = self.tau_embedding(timestep_embedding(tau, self.dim))
        patches = patches + level[:, :, None, :]

        action_tokens = self.action_embedding(actions)

        pieces = []
        for step in range(steps):
            pieces.append(patches[:, step])
            if step < steps - 1:
                pieces.append(action_tokens[:, step:step + 1])
        x = torch.cat(pieces, dim=1)

        types = self.token_types(steps, device)
        x = x + self.type_embedding(types)[None]

        cos, sin = self.rope(self.token_coords(steps, device))
        mask = self.attention_mask(steps, device)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        x = self.norm(x)

        observation = types == 0
        tokens = x[:, observation].reshape(batch, steps,
                                           self.tokens_per_frame, self.dim)
        return self.unpatchify(self.to_noise(tokens))
