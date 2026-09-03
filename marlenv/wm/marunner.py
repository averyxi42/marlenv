"""Rolling the multi-agent world action model forward.

Generation is two-stage per step, and the order is forced rather than
chosen. The shared coordinates of frame ``t + 1`` depend on where the agents
move, which depends on the actions at ``t`` -- so the actions must be
decided before the frames they lead to can even be placed. Sampling actions
first breaks that circularity; denoising both at once could not.

A human's action is held fixed by overwriting it at every denoising step,
which conditions the sample on it while the others are free.
"""
import torch

from marlenv.wm.diffusion import alpha_sigma, from_velocity
from marlenv.wm.model import HEADINGS
from marlenv.wm.multiagent import actions_to_signal, signal_to_actions


class MultiAgentRunner:
    """Generates joint actions and the frames they lead to."""

    def __init__(self, model, origins, window=None, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.window = window
        self.num_agents = model.num_agents
        self.num_actions = model.action_out.out_features
        self.origins = origins.to(self.device)
        self.frames = None                 # (1, t, agents, v, v, c)
        self.actions = None                # (1, t - 1, agents) indices
        self.alive = None                  # (1, t, agents)

    def reset(self, frame):
        self.frames = frame.to(self.device)
        self.actions = torch.zeros(1, 0, self.num_agents, dtype=torch.long,
                                   device=self.device)
        self.alive = torch.ones(1, 1, self.num_agents, dtype=torch.bool,
                                device=self.device)

    def _clip(self):
        """Keep the window, in frames."""
        if self.window is None or self.frames.shape[1] <= self.window:
            return
        drop = self.frames.shape[1] - self.window
        self.frames = self.frames[:, drop:]
        self.actions = self.actions[:, drop:]
        self.alive = self.alive[:, drop:]
        # re-base the origins on the new first frame; only differences matter
        displacement, _ = self.model.trajectory(
            torch.zeros(1, 0, self.num_agents, dtype=torch.long,
                        device=self.device), self.origins)
        self.origins = displacement[:, 0]

    def _pad_actions(self, extra):
        """Actions for the step being generated, appended to the history."""
        return torch.cat([self.actions, extra], dim=1)

    @torch.no_grad()
    def sample_actions(self, fixed=None, denoise_steps=6, generator=None):
        """Sample the joint action from the current state.

        ``fixed`` maps agent index to an action to hold; those are written
        back at every step so the rest are sampled conditioned on them.
        """
        steps = self.frames.shape[1]
        shape = (1, 1, self.num_agents, self.num_actions)
        signal = torch.randn(shape, device=self.device, generator=generator)
        clean_history = actions_to_signal(self.actions, self.num_actions)

        levels = torch.linspace(1.0, 0.0, denoise_steps + 1,
                                device=self.device)
        for index in range(denoise_steps):
            if fixed:
                for agent, action in fixed.items():
                    signal[0, 0, agent] = actions_to_signal(
                        torch.tensor(action, device=self.device),
                        self.num_actions)

            # a trailing action slot: one action per frame, the last being
            # the one asked for. It moves nothing yet, so the coordinates of
            # every existing token are unaffected by its value.
            noisy = torch.cat([clean_history, signal], dim=1)
            frame_tau = torch.zeros(1, steps, self.num_agents,
                                    device=self.device)
            action_tau = torch.zeros(1, steps, self.num_agents,
                                     device=self.device)
            action_tau[:, -1] = levels[index]
            indices = self._pad_actions(signal_to_actions(signal).long())
            _, predicted = self.model(
                self.frames, noisy, frame_tau, action_tau,
                origins=self.origins, action_indices=indices,
                alive=self.alive)

            tau = action_tau[:, -1:]
            clean, noise = from_velocity(signal, predicted[:, -1:], tau)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(levels[index + 1])
            signal = alpha * clean + sigma * noise

        if fixed:
            for agent, action in fixed.items():
                signal[0, 0, agent] = actions_to_signal(
                    torch.tensor(action, device=self.device),
                    self.num_actions)
        return signal_to_actions(signal)[0, 0].long()

    @torch.no_grad()
    def generate_frame(self, actions, denoise_steps=16, generator=None):
        """The frames that follow a decided joint action."""
        steps = self.frames.shape[1]
        history = self._pad_actions(actions.view(1, 1, self.num_agents))
        signal = actions_to_signal(history, self.num_actions)

        shape = (1, 1, self.num_agents, *self.frames.shape[3:])
        frame = torch.randn(shape, device=self.device, generator=generator)
        alive = torch.cat([self.alive, self.alive[:, -1:]], dim=1)

        levels = torch.linspace(1.0, 0.0, denoise_steps + 1,
                                device=self.device)
        for index in range(denoise_steps):
            frames = torch.cat([self.frames, frame], dim=1)
            frame_tau = torch.zeros(1, steps + 1, self.num_agents,
                                    device=self.device)
            frame_tau[:, -1] = levels[index]
            action_tau = torch.zeros(1, steps, self.num_agents,
                                     device=self.device)

            predicted, _ = self.model(
                frames, signal, frame_tau, action_tau, origins=self.origins,
                action_indices=history, alive=alive)

            tau = frame_tau[:, -1:]
            clean, noise = from_velocity(frame, predicted[:, -1:], tau)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(levels[index + 1])
            frame = alpha * clean + sigma * noise
        return frame

    def commit(self, actions, frame, alive=None):
        self.actions = self._pad_actions(actions.view(1, 1, self.num_agents))
        self.frames = torch.cat([self.frames, frame], dim=1)
        if alive is None:
            alive = self.alive[:, -1:]
        self.alive = torch.cat([self.alive, alive.view(1, 1, -1)], dim=1)
        self._clip()

    def step(self, fixed=None, denoise_steps=16, action_steps=6,
             generator=None):
        """One full step: decide actions, then generate the frames."""
        actions = self.sample_actions(fixed, action_steps, generator)
        frame = self.generate_frame(actions, denoise_steps, generator)
        self.commit(actions, frame)
        return actions, frame


class CachedMultiRunner(MultiAgentRunner):
    """The same rollout, against a sliding-window KV cache.

    A multi-agent step adds ``agents * (patches + 1)`` tokens, so the context
    grows several times faster than in the single-agent case and recomputing
    it every denoising pass dominates. Only accepted tokens are written: the
    actions once sampled, then the frames they lead to, in that order, which
    is the order the full sequence has them in.

    Neither stage needs a mask. When sampling actions everything cached is at
    the current step or earlier, and same-step actions may see each other;
    when generating frames everything cached is strictly earlier, and
    same-step patches may see each other. Both are what the full rule allows.
    """

    def __init__(self, model, origins, window=None, device=None):
        super().__init__(model, origins, window=window, device=device)
        from marlenv.wm.cache import KVCache
        tokens = model.num_agents * (model.tokens_per_frame + 1)
        self.cache = KVCache(len(model.blocks), tokens)
        self.time = 0
        self.displacement = None

    def reset(self, frame):
        super().reset(frame)
        self.cache.reset()
        self.time = 0
        self.displacement = self.origins[0].clone()
        self._commit_frames(frame)

    def _commit_frames(self, frame):
        from marlenv.wm.cache import recording
        coords = self.model.step_frame_coords(self.displacement, self.time,
                                              self.device)
        tau = torch.zeros(1, 1, self.num_agents, device=self.device)
        with recording(self.cache):
            self.model.frames_cached(frame, tau, coords, self.cache)

    def _commit_actions(self, actions):
        from marlenv.wm.cache import recording
        coords = self.model.step_action_coords(self.displacement, self.time,
                                               self.device)
        signal = actions_to_signal(actions.view(1, 1, self.num_agents),
                                   self.num_actions)
        tau = torch.zeros(1, 1, self.num_agents, device=self.device)
        with recording(self.cache):
            self.model.actions_cached(signal, tau, coords, self.cache)
        self.cache.frames += 1

    @torch.no_grad()
    def sample_actions(self, fixed=None, denoise_steps=6, generator=None):
        shape = (1, 1, self.num_agents, self.num_actions)
        signal = torch.randn(shape, device=self.device, generator=generator)
        coords = self.model.step_action_coords(self.displacement, self.time,
                                               self.device)
        levels = torch.linspace(1.0, 0.0, denoise_steps + 1,
                                device=self.device)

        def hold():
            if not fixed:
                return
            for agent, action in fixed.items():
                signal[0, 0, agent] = actions_to_signal(
                    torch.tensor(action, device=self.device),
                    self.num_actions)

        for index in range(denoise_steps):
            hold()
            tau = torch.full((1, 1, self.num_agents), float(levels[index]),
                             device=self.device)
            predicted = self.model.actions_cached(signal, tau, coords,
                                                  self.cache)
            clean, noise = from_velocity(signal, predicted, tau)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(levels[index + 1])
            signal = alpha * clean + sigma * noise
        hold()
        return signal_to_actions(signal)[0, 0].long()

    @torch.no_grad()
    def generate_frame(self, actions, denoise_steps=16, generator=None):
        moves = torch.tensor([h.value for h in HEADINGS], device=self.device)
        nxt = self.displacement + moves[actions]
        coords = self.model.step_frame_coords(nxt, self.time + 1,
                                              self.device)
        shape = (1, 1, self.num_agents, *self.frames.shape[3:])
        frame = torch.randn(shape, device=self.device, generator=generator)
        levels = torch.linspace(1.0, 0.0, denoise_steps + 1,
                                device=self.device)

        for index in range(denoise_steps):
            tau = torch.full((1, 1, self.num_agents), float(levels[index]),
                             device=self.device)
            predicted = self.model.frames_cached(frame, tau, coords,
                                                 self.cache)
            clean, noise = from_velocity(frame, predicted, tau)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(levels[index + 1])
            frame = alpha * clean + sigma * noise
        return frame

    def step(self, fixed=None, denoise_steps=16, action_steps=6,
             generator=None):
        actions = self.sample_actions(fixed, action_steps, generator)
        self._commit_actions(actions)
        frame = self.generate_frame(actions, denoise_steps, generator)

        moves = torch.tensor([h.value for h in HEADINGS], device=self.device)
        self.displacement = self.displacement + moves[actions]
        self.time += 1
        self.cache.trim(None if self.window is None else self.window - 1)
        self._commit_frames(frame)

        self.actions = self._pad_actions(actions.view(1, 1, self.num_agents))
        self.frames = torch.cat([self.frames, frame], dim=1)
        self.alive = torch.cat([self.alive, self.alive[:, -1:]], dim=1)
        if self.window is not None and self.frames.shape[1] > self.window:
            drop = self.frames.shape[1] - self.window
            self.frames = self.frames[:, drop:]
            self.actions = self.actions[:, drop:]
            self.alive = self.alive[:, drop:]
        return actions, frame
