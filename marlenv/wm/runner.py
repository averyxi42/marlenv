"""Generating frames one at a time against a KV cache.

Each frame takes many denoising passes that all see the same history, so the
history is encoded once and reused. Provisional frames are never written to
the cache; only the frame that is finally accepted, followed by the action
that leads away from it, which keeps the cache exactly equal to the prefix
the full forward would have built.

Because the spatial coordinates are absolute and RoPE reads differences, the
cache stays valid as the window slides: dropping old frames changes which
differences exist, not what any of them mean. That is also why a window
longer than the one trained on is coherent -- the offsets it produces are
larger, but they are the same kind of thing.
"""
import torch

from marlenv.wm.cache import KVCache, recording
from marlenv.wm.diffusion import alpha_sigma, from_velocity


class CachedRunner:
    """Rolls a world model forward with a sliding-window KV cache."""

    def __init__(self, model, window=None, device=None):
        """``window`` counts frames the model may attend to, itself included.

        The cache therefore keeps ``window - 1`` committed frames, so that
        with the frame being generated the total matches what the full
        forward's mask permits at the same setting.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.window = window
        self.cache = KVCache(len(model.blocks),
                             model.tokens_per_frame + 1)
        self.time = 0
        self.displacement = torch.zeros(2, dtype=torch.long,
                                        device=self.device)
        self.heading = 0

    # ------------------------------------------------------------- geometry
    def _patch_coords(self, time, displacement, heading):
        """Shared-frame coordinates for one frame's patch tokens."""
        model = self.model
        offsets = model.patch_offsets(self.device)
        if model.frame == 'world':
            world = offsets
        else:
            from marlenv.wm.model import (HEADINGS, _FORWARD, _RIGHTWARD)
            forward = torch.tensor(_FORWARD[HEADINGS[heading]],
                                   device=self.device)
            right = torch.tensor(_RIGHTWARD[HEADINGS[heading]],
                                 device=self.device)
            world = (-offsets[:, :1] * forward + offsets[:, 1:] * right)
        spatial = world + displacement
        stamp = torch.full((spatial.shape[0], 1), time, device=self.device)
        return torch.cat([stamp, spatial], dim=-1).long()[None]

    def _action_coords(self, time, displacement):
        """The action sits where the agent is."""
        stamp = torch.tensor([[time]], device=self.device)
        return torch.cat([stamp, displacement[None]], dim=-1).long()[None]

    def _advance(self, action):
        """Update heading and displacement the way the model does."""
        from marlenv.wm.model import HEADINGS
        from marlenv.grading.poses import LEFT_TURN, RIGHT_TURN
        if self.model.frame == 'world':
            self.heading = int(action)
        else:
            current = HEADINGS[self.heading]
            table = (current, LEFT_TURN[current], RIGHT_TURN[current])
            self.heading = HEADINGS.index(table[int(action)])
        move = HEADINGS[self.heading].value
        self.displacement = self.displacement + torch.tensor(
            move, device=self.device)

    # ---------------------------------------------------------------- public
    def reset(self, frame, heading=0):
        """Start from a known first frame."""
        self.cache.reset()
        self.time = 0
        self.displacement = torch.zeros(2, dtype=torch.long,
                                        device=self.device)
        self.heading = heading
        self._commit_frame(frame)

    def _commit_frame(self, frame):
        """Write one accepted frame's keys and values into the cache."""
        clean = torch.zeros(1, 1, device=self.device)
        coords = self._patch_coords(self.time, self.displacement,
                                    self.heading)
        with recording(self.cache):
            self.model.forward_cached(frame, clean, coords, self.cache)
        self.cache.open_step(self.model.tokens_per_frame)
        self.cache.close_step(1)

    def _commit_action(self, action):
        coords = self._action_coords(self.time, self.displacement)
        token = torch.tensor([[int(action)]], device=self.device)
        with recording(self.cache):
            self.model.push_action(token, coords, self.cache)

    @torch.no_grad()
    def step(self, action, denoise_steps=16, generator=None):
        """Commit ``action`` and generate the frame it leads to."""
        self._commit_action(action)
        self._advance(action)
        self.time += 1
        self.cache.trim(None if self.window is None else self.window - 1)

        coords = self._patch_coords(self.time, self.displacement,
                                    self.heading)
        shape = (1, 1, self.model.view, self.model.view, 3)
        frame = torch.randn(shape, device=self.device, generator=generator)
        levels = torch.linspace(1.0, 0.0, denoise_steps + 1,
                                device=self.device)
        for index in range(denoise_steps):
            tau = levels[index].view(1, 1)
            predicted = self.model.forward_cached(frame, tau, coords,
                                                  self.cache)
            clean, noise = from_velocity(frame, predicted, tau)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(levels[index + 1])
            frame = alpha * clean + sigma * noise

        self._commit_frame(frame)
        return frame
