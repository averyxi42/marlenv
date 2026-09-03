"""A multi-agent world *action* model.

Two changes to the single-agent model, both small.

**Agents are told apart by where they are.** With shared spatial
coordinates every token already sits at its owner's world position, and two
snakes are never in the same cell, so identity is carried geometrically and
no agent-id embedding is needed. That also makes the model permutation
equivariant in the agents: reordering them reorders the outputs and changes
nothing else. The one thing actions cannot supply is where the agents start
relative to each other, so the initial offsets are given; they are
differences, so no absolute position on the board leaks in.

**Actions are generated, not just conditioned on.** A world model alone
cannot be rolled out with more than one agent, because the other agents'
actions are not available to condition on -- they are policies. Diffusing
the action tokens under the same objective as the frames makes the model a
policy and a dynamics model at once, so a rollout can sample what the
others do while a human's own action is held fixed.

The attention rule is untouched. An action token at time t already attends
to frame t and everything earlier, which is exactly what a policy needs,
while the frame's own patches still cannot see it.
"""
import torch
import torch.nn as nn

from marlenv.wm.model import (HEADINGS, WorldModel, _FORWARD, _RIGHTWARD,
                              timestep_embedding)


def actions_to_signal(indices, num_actions):
    """Discrete actions as vectors in ``[-1, 1]``, the range diffusion uses."""
    one_hot = nn.functional.one_hot(indices.long(), num_actions).float()
    return one_hot * 2.0 - 1.0


def signal_to_actions(signal):
    """The most likely action from a generated vector."""
    return signal.argmax(dim=-1)


class MultiAgentWorldModel(WorldModel):
    """Frames and actions for several agents, all diffused together."""

    def __init__(self, num_agents=3, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = num_agents
        # actions become content to be generated, so they need a projection
        # in and a noise head out, exactly as the patches do
        self.action_in = nn.Linear(self.action_embedding.num_embeddings,
                                   self.dim)
        self.action_out = nn.Linear(self.dim,
                                    self.action_embedding.num_embeddings)
        self.action_tau = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(),
            nn.Linear(self.dim, self.dim))

    @property
    def tokens_per_step(self):
        return self.num_agents * (self.tokens_per_frame + 1)

    # ------------------------------------------------------------- geometry
    def trajectory(self, actions, origins=None, alive=None):
        """Per-agent displacement and heading, in one shared frame.

        ``actions`` is ``(b, t - 1, agents)`` of action indices and
        ``origins`` ``(b, agents, 2)`` of starting offsets relative to each
        other. Without origins the agents would each dead-reckon from their
        own zero and their coordinates would not be comparable.

        A dead agent stops moving. Letting it keep dead-reckoning would walk
        its tokens across the board and eventually onto a living agent's
        cells, which is precisely the confusion the positional identity is
        supposed to prevent. ``alive`` is ``(b, t, agents)``; a rollout that
        does not know it can read it off the black frame the model predicts
        on death.
        """
        batch, transitions, agents = actions.shape
        device = actions.device
        steps = transitions + 1

        if origins is None:
            origins = torch.zeros(batch, agents, 2, dtype=torch.long,
                                  device=device)
        displacement = torch.zeros(batch, steps, agents, 2, dtype=torch.long,
                                   device=device)
        displacement[:, 0] = origins
        heading = torch.zeros(batch, steps, agents, dtype=torch.long,
                              device=device)
        moves = torch.tensor([h.value for h in HEADINGS], device=device)

        for step in range(transitions):
            nxt = actions[:, step]                 # cardinal index per agent
            step_move = moves[nxt]
            if alive is not None:
                step_move = step_move * alive[:, step + 1, :, None].long()
                heading[:, step + 1] = torch.where(
                    alive[:, step + 1], nxt, heading[:, step])
            else:
                heading[:, step + 1] = nxt
            displacement[:, step + 1] = displacement[:, step] + step_move
        return displacement, heading

    def token_coords(self, steps, device, actions=None, origins=None,
                     alive=None):
        """``(batch, tokens, 3)``; every token sits where its agent is."""
        offsets = self.patch_offsets(device)
        displacement, _ = self.trajectory(actions, origins, alive)
        batch = displacement.shape[0]

        pieces = []
        for step in range(steps):
            for agent in range(self.num_agents):
                shift = displacement[:, step, agent]           # (b, 2)
                spatial = offsets[None].expand(batch, -1, -1) + shift[:, None]
                stamp = torch.full((batch, self.tokens_per_frame, 1), step,
                                   device=device)
                pieces.append(torch.cat([stamp, spatial], dim=-1))
            if step < steps - 1:
                for agent in range(self.num_agents):
                    shift = displacement[:, step, agent]
                    stamp = torch.full((batch, 1, 1), step, device=device)
                    pieces.append(torch.cat([stamp, shift[:, None]], dim=-1))
        return torch.cat(pieces, dim=1).long()

    def token_types(self, steps, device):
        types = []
        for step in range(steps):
            types.append(torch.zeros(self.num_agents * self.tokens_per_frame,
                                     device=device))
            if step < steps - 1:
                types.append(torch.ones(self.num_agents, device=device))
        return torch.cat(types).long()

    # --------------------------------------------------------------- forward
    def forward(self, noisy_frames, noisy_actions, frame_tau, action_tau,
                origins=None, window=None, action_indices=None, alive=None):
        """Predict the noise on every frame and every action.

        ``noisy_frames`` is ``(b, t, agents, v, v, c)`` and ``noisy_actions``
        ``(b, t - 1, agents, num_actions)``. ``action_indices`` is only used
        to lay out the shared coordinates, so pass the clean actions there
        even when the action content is noised; a rollout that does not know
        them yet passes its best current estimate.
        """
        batch, steps, agents = noisy_frames.shape[:3]
        device = noisy_frames.device

        flat = noisy_frames.reshape(batch, steps * agents,
                                    *noisy_frames.shape[3:])
        patches = self.frame_tokens(flat, frame_tau.reshape(batch,
                                                            steps * agents))
        patches = patches.reshape(batch, steps, agents,
                                  self.tokens_per_frame, self.dim)

        action_tokens = self.action_in(noisy_actions)
        action_tokens = action_tokens + self.action_tau(
            timestep_embedding(action_tau, self.dim))

        pieces = []
        for step in range(steps):
            for agent in range(agents):
                pieces.append(patches[:, step, agent])
            if step < steps - 1:
                for agent in range(agents):
                    pieces.append(action_tokens[:, step, agent:agent + 1])
        x = torch.cat(pieces, dim=1)

        types = self.token_types(steps, device)
        x = x + self.type_embedding(types)[None]

        indices = action_indices if action_indices is not None else \
            signal_to_actions(noisy_actions)
        cos, sin = self.rope(self.token_coords(steps, device, indices,
                                               origins, alive))
        mask = self.attention_mask(steps, device, window, indices, origins,
                                   alive)
        x = self.run_blocks(x, cos, sin, mask)

        observed = x[:, types == 0].reshape(batch, steps, agents,
                                            self.tokens_per_frame, self.dim)
        frame_noise = self.unpatchify(
            self.to_noise(observed.reshape(batch, steps * agents,
                                           self.tokens_per_frame, self.dim)))
        frame_noise = frame_noise.reshape(batch, steps, agents,
                                          *frame_noise.shape[2:])
        action_noise = self.action_out(x[:, types == 1]).reshape(
            batch, steps - 1, agents, -1)
        return frame_noise, action_noise

    def attention_mask(self, steps, device, window=None, actions=None,
                       origins=None, alive=None):
        """Unchanged in rule; only the token layout differs."""
        from marlenv.wm.attention import build_mask
        time = self.token_coords(steps, device, actions, origins,
                                 alive)[0][:, 0]
        is_action = self.token_types(steps, device) == 1
        return build_mask(time, is_action, window)
