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

from marlenv.core.palette import decode_grid
from marlenv.core.snake import Cell
from marlenv.grading.compare import PALETTE_SNAKES
from marlenv.wm.data import to_pixels
from marlenv.wm.diffusion import alpha_sigma, from_velocity
from marlenv.wm.model import HEADINGS
from marlenv.wm.multiagent import actions_to_signal, signal_to_actions


class MultiAgentRunner:
    """Generates joint actions and the frames they lead to."""

    def __init__(self, model, origins, window=None, device=None,
                 num_agents=None, immortal_agents=None, death_patience=3):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.window = window
        self.num_agents = (model.num_agents if num_agents is None
                           else num_agents)
        self.num_actions = model.action_out.out_features
        self.origins = origins.to(self.device)
        self.frames = None                 # (1, t, agents, v, v, c)
        self.actions = None                # (1, t - 1, agents) indices
        self.alive = None                  # (1, t, agents)
        self.immortal_agents = immortal_agents
        self.death_patience = death_patience
        self.misses = [0] * self.num_agents
        # the agents actually in play, which may be fewer than the model
        # was trained with -- identity is positional, so that is legal
        self.all_agents = list(range(self.num_agents))

    def reset(self, frame):
        self.misses = [0] * self.num_agents
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

        frames, past_tau = self.retire(self.frames, self.alive, generator)
        history, past_action_tau = self.retire(
            clean_history, self.alive[:, :clean_history.shape[1]], generator)
        here = self.alive[:, -1:]

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
            noisy = torch.cat([history, signal], dim=1)
            frame_tau = past_tau
            action_tau = torch.cat(
                [past_action_tau,
                 torch.where(here, torch.full_like(past_tau[:, :1],
                                                   float(levels[index])),
                             torch.ones_like(past_tau[:, :1]))], dim=1)
            indices = self._pad_actions(signal_to_actions(signal).long())
            _, predicted = self.model(
                frames, noisy, frame_tau, action_tau,
                origins=self.origins, action_indices=indices,
                alive=self.alive)

            tau = torch.full_like(action_tau[:, -1:], float(levels[index]))
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

        # retired viewpoints are drawn once and held, as a training sample
        # holds its noise, rather than redrawn every denoising pass
        past, past_tau = self.retire(self.frames, self.alive, generator)
        signal, action_tau = self.retire(signal, self.alive, generator)
        here = alive[:, -1:]

        levels = torch.linspace(1.0, 0.0, denoise_steps + 1,
                                device=self.device)
        for index in range(denoise_steps):
            frames = torch.cat([past, frame], dim=1)
            level = torch.where(here, torch.full_like(past_tau[:, :1],
                                                      float(levels[index])),
                                torch.ones_like(past_tau[:, :1]))
            frame_tau = torch.cat([past_tau, level], dim=1)

            predicted, _ = self.model(
                frames, signal, frame_tau, action_tau, origins=self.origins,
                action_indices=history, alive=alive)

            tau = torch.full_like(frame_tau[:, -1:], float(levels[index]))
            clean, noise = from_velocity(frame, predicted[:, -1:], tau)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(levels[index + 1])
            frame = alpha * clean + sigma * noise
        return frame

    def retire(self, content, alive, generator=None):
        """Present retired viewpoints the way training did.

        Training said "this agent contributes nothing" by pinning its tokens
        at the maximum noise level, where alpha is zero and the input is
        pure noise whatever was written there. A rollout has to say it the
        same way, which means both halves: tau of one *and* noise in place
        of the content. Passing a dead agent's last frame at tau zero claims
        it is known, and dropping its tokens changes the token layout -- the
        model never saw either, so the whole context goes out of
        distribution the moment anyone dies.

        Returns the content to feed and the noise levels to feed it at.
        """
        tau = (~alive).float()
        if not bool(tau.any()):
            return content, tau
        noise = torch.randn(content.shape, device=content.device,
                            generator=generator)
        spread = alive.reshape(*alive.shape, *([1] * (content.dim()
                                                      - alive.dim())))
        return torch.where(spread, content, noise), tau

    def commit(self, actions, frame, alive=None):
        self.actions = self._pad_actions(actions.view(1, 1, self.num_agents))
        self.frames = torch.cat([self.frames, frame], dim=1)
        if alive is None:
            alive = self.alive[:, -1:]
        self.alive = torch.cat([self.alive, alive.view(1, 1, -1)], dim=1)
        self._clip()

    def filtered_looks_dead(self, frame):
        """Which viewpoints to retire, given what their centres show.

        A single reading is not enough to go on. The centre is its owner's
        head while it lives, but the model only paints that head about
        seven times in ten, and a miss reads as empty -- so retiring on one
        look kills a living agent within a few steps, and retiring is
        permanent. Waiting for the reading to persist costs a few steps of
        delay on a real death, which is cheap: a dead viewpoint is frozen,
        so it keeps showing the same thing, while a rendering miss does not
        survive being asked again.
        """
        seen = looks_dead(frame[0, 0]).to(self.alive.device)
        died = torch.zeros_like(seen)
        for agent in range(self.num_agents):
            if bool(seen[agent]):
                self.misses[agent] += 1
            else:
                self.misses[agent] = 0
            died[agent] = self.misses[agent] >= self.death_patience
        if self.immortal_agents is not None:
            died[self.immortal_agents] = 0
        return died
    def step(self, fixed=None, denoise_steps=16, action_steps=6,
             generator=None):
        """One full step: decide actions, then generate the frames."""
        actions = self.sample_actions(fixed, action_steps, generator)
        frame = self.generate_frame(actions, denoise_steps, generator)
        # a viewpoint whose centre is no longer its own head has died; it is
        # masked out of the conditioning from here on, as in the cached path
        died = self.filtered_looks_dead(frame)
        self.commit(actions, frame, self.alive[:, -1].reshape(-1) & ~died)
        return actions, frame

    @torch.no_grad()
    def observe(self, actions, frame, alive=None):
        """Absorb a real transition instead of a generated one.

        This is how a rollout is given a prefix of real history: the frames
        and actions come from the simulator rather than the model, but they
        enter the context by exactly the same route.
        """
        self.commit(actions.view(1, 1, self.num_agents), frame, alive)
        return actions, frame


SNAKE_KINDS = (Cell.HEAD.value, Cell.BODY.value, Cell.TAIL.value)


def looks_dead(frame, num_snakes=PALETTE_SNAKES):
    """Has this viewpoint stopped being a living snake's head?

    While an agent lives, the centre of its view is its own head, by
    construction. Once it dies the view is taken from the cell it died
    entering, so the centre shows whatever is there instead. That makes
    death a property of the observation rather than a sentinel value, and
    needs no threshold.
    """
    pixels = to_pixels(frame.detach().cpu().numpy())
    middle = pixels.shape[-2] // 2
    centre = pixels[..., middle, middle, :]
    grid = decode_grid(centre.reshape(-1, 1, 3), num_snakes).reshape(-1)
    return torch.tensor([int(v) % 10 != Cell.HEAD.value for v in grid],
                        device=frame.device)


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

    def __init__(self, model, origins, window=None, device=None,
                 num_agents=None, immortal_agents=None, death_patience=3):
        super().__init__(model, origins, window=window, device=device,
                         num_agents=num_agents,
                         immortal_agents=immortal_agents,
                         death_patience=death_patience)
        from marlenv.wm.cache import KVCache
        tokens = self.num_agents * (model.tokens_per_frame + 1)
        self.cache = KVCache(len(model.blocks), tokens)
        self.time = 0
        self.displacement = None
        self.live = None

    def reset(self, frame):
        super().reset(frame)
        self.cache.reset()
        self.time = 0
        self.displacement = self.origins[0].clone()
        self.live = [True] * self.num_agents
        self.misses = [0] * self.num_agents
        self._commit_frames(frame)

    @property
    def living(self):
        return [i for i, alive in enumerate(self.live) if alive]

    @property
    def live_flags(self):
        return torch.tensor(self.live, device=self.device).view(1, 1, -1)

    def _commit_frames(self, frame):
        """Write every agent's patches, retired ones as noise at tau one."""
        from marlenv.wm.cache import recording
        content, tau = self.retire(frame, self.live_flags)
        coords = self.model.step_frame_coords(self.displacement, self.time,
                                              self.device, self.all_agents)
        with recording(self.cache):
            self.model.frames_cached(content, tau, coords, self.cache)
        tokens = self.num_agents * self.model.tokens_per_frame
        self.cache.open_step(tokens)
        return tokens

    def _commit_actions(self, actions):
        from marlenv.wm.cache import recording
        signal = actions_to_signal(actions.view(1, 1, self.num_agents),
                                   self.num_actions)
        content, tau = self.retire(signal, self.live_flags)
        coords = self.model.step_action_coords(self.displacement, self.time,
                                               self.device, self.all_agents)
        with recording(self.cache):
            self.model.actions_cached(content, tau, coords, self.cache)
        self.cache.close_step(self.num_agents)
        return self.num_agents

    @torch.no_grad()
    def sample_actions(self, fixed=None, denoise_steps=6, generator=None):
        agents = self.living
        full = torch.zeros(self.num_agents, dtype=torch.long,
                           device=self.device)
        if not agents:
            return full
        # every agent contributes a token, retired ones as fixed noise held
        # at tau one -- the same thing training fed them
        live = self.live_flags
        shape = (1, 1, self.num_agents, self.num_actions)
        signal = torch.randn(shape, device=self.device, generator=generator)
        retired = signal.clone()
        coords = self.model.step_action_coords(self.displacement, self.time,
                                               self.device, self.all_agents)
        levels = torch.linspace(1.0, 0.0, denoise_steps + 1,
                                device=self.device)
        spread = live[..., None]

        def hold():
            if not fixed:
                return
            for agent, action in fixed.items():
                if self.live[agent]:
                    signal[0, 0, agent] = actions_to_signal(
                        torch.tensor(action, device=self.device),
                        self.num_actions)

        for index in range(denoise_steps):
            hold()
            level = float(levels[index])
            tau = torch.where(live, torch.full_like(live, level,
                                                    dtype=torch.float),
                              torch.ones_like(live, dtype=torch.float))
            predicted = self.model.actions_cached(signal, tau, coords,
                                                  self.cache)
            flat = torch.full_like(tau, level)
            clean, noise = from_velocity(signal, predicted, flat)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(levels[index + 1])
            signal = torch.where(spread, alpha * clean + sigma * noise,
                                 retired)
        hold()
        chosen = signal_to_actions(signal)[0, 0].long()
        full[agents] = chosen[agents]
        return full

    @torch.no_grad()
    def generate_frame(self, actions, denoise_steps=16, generator=None):
        agents = self.living
        shape = (1, 1, self.num_agents, *self.frames.shape[3:])
        out = torch.full(shape, -1.0, device=self.device)   # dead: a filler
        if not agents:
            return out

        live = self.live_flags
        moves = torch.tensor([h.value for h in HEADINGS], device=self.device)
        step_move = moves[actions] * live.reshape(-1, 1).long()
        coords = self.model.step_frame_coords(self.displacement + step_move,
                                              self.time + 1, self.device,
                                              self.all_agents)
        frame = torch.randn(shape, device=self.device, generator=generator)
        retired = frame.clone()
        spread = live[..., None, None, None]
        levels = torch.linspace(1.0, 0.0, denoise_steps + 1,
                                device=self.device)

        for index in range(denoise_steps):
            level = float(levels[index])
            tau = torch.where(live, torch.full_like(live, level,
                                                    dtype=torch.float),
                              torch.ones_like(live, dtype=torch.float))
            predicted = self.model.frames_cached(frame, tau, coords,
                                                 self.cache)
            flat = torch.full_like(tau, level)
            clean, noise = from_velocity(frame, predicted, flat)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(levels[index + 1])
            frame = torch.where(spread, alpha * clean + sigma * noise,
                                retired)
        out[:, :, agents] = frame[:, :, agents]
        return out

    def step(self, fixed=None, denoise_steps=16, action_steps=6,
             generator=None):
        actions = self.sample_actions(fixed, action_steps, generator)
        self._commit_actions(actions)
        frame = self.generate_frame(actions, denoise_steps, generator)
        self._absorb(actions, frame)
        return actions, frame

    @torch.no_grad()
    def observe(self, actions, frame, alive=None):
        """Absorb a real transition instead of a generated one.

        Same route into the cache as a generated step takes, so a prefix of
        real history and the rollout that follows it are indistinguishable
        to the model. Aliveness is known here rather than read off the
        frame, so it is passed in.
        """
        self._commit_actions(actions)
        self._absorb(actions, frame, alive)
        return actions, frame

    def _absorb(self, actions, frame, alive=None):
        """Advance the geometry and commit one step's frames."""
        moves = torch.tensor([h.value for h in HEADINGS], device=self.device)
        step_move = moves[actions]
        for agent in range(self.num_agents):
            if not self.live[agent]:
                step_move[agent] = 0
        self.displacement = self.displacement + step_move
        self.time += 1

        # death is a property of the observation: read it off the centre
        # cell, unless the caller already knows who is alive
        died = (self.filtered_looks_dead(frame) if alive is None
                else ~torch.as_tensor(alive, device=self.device).reshape(-1))
        for agent in range(self.num_agents):
            if self.live[agent] and bool(died[agent]):
                self.live[agent] = False

        self.cache.trim(None if self.window is None else self.window - 1)
        self._commit_frames(frame)

        self.actions = self._pad_actions(actions.view(1, 1, self.num_agents))
        self.frames = torch.cat([self.frames, frame], dim=1)
        live = torch.tensor(self.live, device=self.device).view(1, 1, -1)
        self.alive = torch.cat([self.alive, live], dim=1)
        if self.window is not None and self.frames.shape[1] > self.window:
            drop = self.frames.shape[1] - self.window
            self.frames = self.frames[:, drop:]
            self.actions = self.actions[:, drop:]
            self.alive = self.alive[:, drop:]
