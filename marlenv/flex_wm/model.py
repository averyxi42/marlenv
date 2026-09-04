"""A world action model over sets of observation/action pairs.

Same weights as :class:`~marlenv.wm.multiagent.MultiAgentWorldModel`, and
deliberately the same module names, so a checkpoint trained by either loads
into the other. Two things differ.

The input is a set of pairs rather than a ``(time, agent)`` rectangle, so
nothing here has a fixed agent count and nothing indexes by agent id. That
is what allows the number of agents to change during an episode, and what
allows an episode to contain more identities than are ever live at once.

Attention scope varies by block. With every block global this is exactly
the older model -- that equality is pinned by a test, since the whole point
is that this generalises rather than replaces.
"""
import torch
import torch.nn as nn

from marlenv.flex_wm.attention import build_masks, parse_schedule
from marlenv.flex_wm.pairs import token_attributes, token_coords
from marlenv.wm.attention import attend  # noqa: F401  (kept in one place)
from marlenv.wm.model import timestep_embedding
from marlenv.wm.multiagent import MultiAgentWorldModel


class FlexWorldModel(MultiAgentWorldModel):
    """Frames and actions diffused together, over a set of pairs."""

    def __init__(self, schedule='G', **kwargs):
        # the parent's agent count is meaningless here and is kept only so
        # its module shapes are identical; nothing below reads it
        super().__init__(num_agents=1, **kwargs)
        self.schedule = parse_schedule(schedule, len(self.blocks))

    @property
    def tokens_per_pair(self):
        return self.tokens_per_frame + 1

    # --------------------------------------------------------------- tokens
    def pair_tokens(self, pairs, frames, actions, frame_tau, action_tau):
        """Embed every pair into ``tokens_per_pair`` tokens, in pair order."""
        batch, count = pairs.batch, pairs.pairs
        patches = self.frame_tokens(frames, frame_tau)

        actions = self.action_in(actions)
        actions = actions + self.action_tau(
            timestep_embedding(action_tau, self.dim))

        tokens = torch.cat([patches, actions[:, :, None]], dim=2)
        return tokens.reshape(batch, count * self.tokens_per_pair, self.dim)

    def token_types(self, batch, count, device):
        types = torch.zeros(self.tokens_per_pair, dtype=torch.long,
                            device=device)
        types[-1] = 1
        return types[None, None].expand(batch, count, -1).reshape(batch, -1)

    # -------------------------------------------------------------- forward
    def forward(self, pairs, noisy_frames, noisy_actions, frame_tau,
                action_tau, window=None):
        """Predict the noise on every observation and every action.

        ``pairs`` carries the attributes -- who, when, where, and what is
        real -- and the two noisy tensors carry the content being denoised.
        Keeping them apart is what lets a rollout hand over a set of pairs
        whose actions are still only a guess: the guess lays out the
        coordinates, and the content it is denoising is separate.
        """
        device = pairs.observations.device
        batch, count = pairs.batch, pairs.pairs

        x = self.pair_tokens(pairs, noisy_frames, noisy_actions, frame_tau,
                             action_tau)
        types = self.token_types(batch, count, device)
        x = x + self.type_embedding(types)

        time, agent, is_action, valid = token_attributes(
            pairs, self.tokens_per_frame)
        cos, sin = self.rope(token_coords(pairs, self.patch_offsets(device)))
        masks = build_masks(self.schedule, time, agent, is_action, window,
                            valid)

        for block, scope in zip(self.blocks, self.schedule):
            x = block(x, cos, sin, masks[scope])
        x = self.norm(x)

        grouped = x.reshape(batch, count, self.tokens_per_pair, self.dim)
        frames = self.unpatchify(self.to_noise(grouped[:, :, :-1]))
        actions = self.action_out(grouped[:, :, -1])
        return frames, actions
