"""Rolling a flex model forward, one step at a time.

The history is a set of pairs and stays one; agents join it and leave it
without the set having a shape to disagree with. A step is two denoisings
against that set:

    actions   every live agent's pair has its observation and wants its
              action. Only the action tokens are noisy.
    frames    a new pair per live agent, at the position its action just
              moved it to, wanting its observation. Its own action is not
              yet decided, which costs nothing: an observation may never
              attend to the action taken from it, so whatever sits in that
              slot cannot reach it.

Both are exactly what training presents, which is the point -- the noise
levels say which tokens are known, and nothing else about the arrangement
changes between the two.

A rectangular view of the history is offered for the tools that want one,
but it is a report rather than the representation.
"""
import torch

from marlenv.flex_wm.pairs import PairBatch, compact
from marlenv.wm.diffusion import alpha_sigma, from_velocity
from marlenv.wm.marunner import looks_dead
from marlenv.wm.model import HEADINGS
from marlenv.wm.multiagent import actions_to_signal, signal_to_actions


class FlexRunner:
    """Generates joint actions and the observations that follow them."""

    def __init__(self, model, agents, positions, window=None, device=None,
                 death_patience=3, immortal=None):
        """
        model          a ``FlexWorldModel``
        agents         ``(n,)`` long, the identities in play
        positions      ``(n, 2)`` long, where each one starts
        window         frames of history kept, and of attention
        death_patience consecutive frames whose centre is not a head before
                       a viewpoint is retired
        immortal       identities never retired, by index into ``agents``
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.window = window
        self.death_patience = death_patience
        self.immortal = immortal or []
        self.agents = torch.as_tensor(agents, dtype=torch.long,
                                      device=self.device)
        self.start = torch.as_tensor(positions, dtype=torch.long,
                                     device=self.device)
        self.num_agents = len(self.agents)
        if len(torch.unique(self.agents)) != self.num_agents:
            raise ValueError("agent identities must be unique")
        if window is not None and window < 1:
            raise ValueError("window must be positive")
        self.reset_state()

    def reset_state(self):
        self.pairs = None
        self.time = 0
        self.live = [True] * self.num_agents
        self.misses = [0] * self.num_agents
        self.position = self.start.clone()

    @property
    def living(self):
        return [i for i, alive in enumerate(self.live) if alive]

    # ------------------------------------------------------------- history
    def reset(self, observations):
        """Seed with a real first observation, ``(1, 1, n, v, v, c)``."""
        self.reset_state()
        self.latest = observations.clone()
        self.pairs = self._new_pairs(observations[:, 0], self.living,
                                     actions=None)
        self.actions_known = torch.zeros(1, self.pairs.pairs,
                                         dtype=torch.bool,
                                         device=self.device)

    def _new_pairs(self, frames, agents, actions=None):
        """One pair per given agent, at this runner's current time."""
        count = len(agents)
        index = torch.tensor(agents, dtype=torch.long, device=self.device)
        act = (torch.zeros(count, dtype=torch.long, device=self.device)
               if actions is None else actions[index])
        return PairBatch(
            observations=frames[:, index],
            actions=act[None],
            agent=self.agents[index][None],
            time=torch.full((1, count), self.time, dtype=torch.long,
                            device=self.device),
            position=self.position[index][None],
            acted=torch.tensor([[self.live[i] for i in agents]],
                               dtype=torch.bool, device=self.device))

    def _remember(self, frames, agents):
        """Hold the newest observation per agent, ready to report.

        Scanning the history for it costs a Python pass over every pair
        with a device synchronisation each; measured against a full step it
        was five times the price of the work itself.
        """
        index = torch.tensor(agents, dtype=torch.long, device=self.device)
        self.latest[0, 0, index] = frames[0]

    def _append(self, extra, known, agents=None):
        if agents is not None:
            self._remember(extra.observations, agents)
        join = lambda a, b: torch.cat([a, b], dim=1)
        self.pairs = PairBatch(
            observations=join(self.pairs.observations, extra.observations),
            actions=join(self.pairs.actions, extra.actions),
            agent=join(self.pairs.agent, extra.agent),
            time=join(self.pairs.time, extra.time),
            position=join(self.pairs.position, extra.position),
            valid=join(self.pairs.valid, extra.valid),
            trained=join(self.pairs.trained, extra.trained),
            acted=join(self.pairs.acted, extra.acted),
            visible=(None if self.pairs.visible is None and extra.visible is None
                     else join(self.pairs.patch_mask(self.model.tokens_per_frame),
                               extra.patch_mask(self.model.tokens_per_frame))))
        self.actions_known = join(self.actions_known, known)
        self._trim()

    def _trim(self):
        if self.window is None:
            return
        keep = self.pairs.time > self.time - self.window
        if keep.all():
            return
        self.actions_known = self.actions_known[keep][None]
        self.pairs = compact(self.pairs, keep)

    # ------------------------------------------------------------ stepping
    def _levels(self, steps):
        return torch.linspace(1.0, 0.0, steps + 1, device=self.device)

    def _denoise(self, target, steps, generator, frame_slot):
        """DDIM over targets with clean history and unavailable actions at tau=1."""
        pairs = self.pairs
        clean_actions = actions_to_signal(
            pairs.actions, self.model.action_out.out_features)

        # Observations and decided actions stay clean. An unavailable action
        # stays at maximum noise and is never a prediction target.
        if frame_slot:
            # Current actions cannot reach current observations; historical
            # unavailable actions can, and must carry noise at tau=1.
            unknown = ~self.actions_known & (pairs.time < self.time)
            if unknown.any():
                clean_actions = torch.where(
                    unknown[..., None],
                    torch.randn(clean_actions.shape, device=self.device,
                                generator=generator), clean_actions)
        known = pairs.observations if frame_slot else clean_actions
        spread = target.reshape(*target.shape,
                                *([1] * (known.dim() - target.dim())))
        noisy = spread if frame_slot else (~self.actions_known)[..., None]
        content = torch.where(
            noisy, torch.randn(known.shape, device=self.device,
                               generator=generator), known)

        zero = torch.zeros(pairs.batch, pairs.pairs, device=self.device)
        # an action nobody has decided yet is unknown, whatever is written
        # in its slot; saying so is what keeps it out of the conditioning
        action_known = torch.where(self.actions_known, zero,
                                   torch.ones_like(zero))

        for index in range(steps):
            level = float(self._levels(steps)[index])
            tau = torch.where(target, torch.full_like(zero, level),
                              torch.zeros_like(zero))
            frame_tau = tau if frame_slot else zero
            action_tau = action_known if frame_slot else torch.where(
                target, torch.full_like(zero, level), action_known)

            frames = content if frame_slot else pairs.observations
            actions = clean_actions if frame_slot else content
            with torch.no_grad():
                predicted = self.model(pairs, frames, actions, frame_tau,
                                       action_tau, window=self.window)
            predicted = predicted[0] if frame_slot else predicted[1]

            flat = torch.full_like(tau, level)
            clean, noise = from_velocity(content, predicted, flat)
            clean = clean.clamp(-1.0, 1.0)
            alpha, sigma = alpha_sigma(self._levels(steps)[index + 1])
            content = torch.where(spread, alpha * clean + sigma * noise,
                                  content)
        return content

    def _current_slots(self, pairs=None):
        """Map current pair slots to runner indices through their identities."""
        pairs = self.pairs if pairs is None else pairs
        for agent in self.living:
            slots = torch.nonzero(
                pairs.valid[0] & (pairs.time[0] == self.time)
                & (pairs.agent[0] == self.agents[agent]), as_tuple=True)[0]
            if len(slots) != 1:
                raise RuntimeError("each live agent needs exactly one current pair")
            yield int(slots[0]), agent

    def _mark_retired_actions(self):
        # Keep the death observation, but it has no outgoing action. Do not
        # change valid/trained: the observation remains real context.
        for pairs in (self.pairs, getattr(self, 'frontier', None)):
            if pairs is None:
                continue
            dead = torch.zeros_like(pairs.valid)
            for agent, live in enumerate(self.live):
                if not live:
                    dead |= ((pairs.time == self.time)
                             & (pairs.agent == self.agents[agent]))
            pairs.acted = pairs.acted & ~dead

    def _record_actions(self, actions):
        self._mark_retired_actions()
        for slot, agent in self._current_slots():
            self.pairs.actions[0, slot] = int(actions[agent])
            self.actions_known[0, slot] = True
        frontier = getattr(self, 'frontier', None)
        if frontier is not None:
            for slot, agent in self._current_slots(frontier):
                frontier.actions[0, slot] = int(actions[agent])

    @torch.no_grad()
    def sample_actions(self, fixed=None, steps=6, generator=None):
        """Decide live agents' current actions; fixed actions condition peers.

        ``fixed`` keys are runner indices, as in the returned full vector.
        A repeated call returns already decided actions without resampling.
        """
        self._mark_retired_actions()
        slots = list(self._current_slots())
        for slot, agent in slots:
            if fixed and agent in fixed:
                action = int(fixed[agent])
                if not 0 <= action < self.model.action_out.out_features:
                    raise ValueError("fixed action out of range")
                self.pairs.actions[0, slot] = action
                self.actions_known[0, slot] = True
        frontier = getattr(self, 'frontier', None)
        if frontier is not None:
            frontier.actions[:] = self.pairs.actions[:, -frontier.pairs:]
        target = torch.zeros_like(self.actions_known)
        for slot, _ in slots:
            target[0, slot] = ~self.actions_known[0, slot]
        if target.any():
            signal = self._denoise(target, steps, generator, frame_slot=False)
            chosen = signal_to_actions(signal)
            self.pairs.actions[target] = chosen[target]
        full = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
        for slot, agent in slots:
            full[agent] = self.pairs.actions[0, slot]
        self._record_actions(full)
        return full

    @torch.no_grad()
    def generate_frames(self, actions, steps=12, generator=None):
        """The observations that follow a decided joint action."""
        self._record_actions(actions)
        moves = torch.tensor([h.value for h in HEADINGS], device=self.device)
        alive = self.living
        if not alive:
            # nobody left to look from, so there is nothing to generate
            self.time += 1
            empty = self.pairs.observations[:, :0]
            return empty, alive
        for agent in alive:
            self.position[agent] = self.position[agent] + moves[
                actions[agent]]
        self.time += 1

        blank = torch.zeros(1, self.num_agents,
                            *self.pairs.observations.shape[2:],
                            device=self.device)
        extra = self._new_pairs(blank, alive)
        known = torch.zeros(1, len(alive), dtype=torch.bool,
                            device=self.device)
        self._append(extra, known, alive)

        target = torch.zeros(1, self.pairs.pairs, dtype=torch.bool,
                             device=self.device)
        target[0, -len(alive):] = True
        frames = self._denoise(target, steps, generator, frame_slot=True)
        self.pairs.observations[0, -len(alive):] = frames[0, -len(alive):]
        self._remember(frames[:, -len(alive):], alive)
        return frames[:, -len(alive):], alive

    def retire(self, frames, alive):
        """Read death off the centre cell, once it has persisted.

        frames ``(1, len(alive), v, v, c)`` just generated
        alive   the agent indices those frames belong to, in order

        A single reading is not enough: the model paints the head it stands
        on most of the time but not always, and retiring is permanent.
        """
        if not alive:
            return
        seen = looks_dead(frames[0]).to(self.device)
        for slot, agent in enumerate(alive):
            if agent in self.immortal:
                continue
            self.misses[agent] = (self.misses[agent] + 1
                                  if bool(seen[slot]) else 0)
            if self.misses[agent] >= self.death_patience:
                self.live[agent] = False
        self._mark_retired_actions()

    # ---------------------------------------------------- rectangular view
    @property
    def alive(self):
        """``(1, 1, n)`` bool, for tools that expect an agent axis."""
        return torch.tensor(self.live, device=self.device)[None, None]

    @property
    def frames(self):
        """``(1, 1, n, v, v, c)`` of the newest observation per agent.

        A display snapshot, not context: retired slots retain the single
        final observation. No new pairs are made from these stale slots.
        """
        return self.latest

    @torch.no_grad()
    def observe(self, actions, frames, live=None):
        """Absorb a real transition instead of a generated one.

        actions ``(n,)`` long, what each agent actually did
        frames  ``(1, 1, n, v, v, c)`` what each actually saw next
        live    ``(n,)`` bool, who is still playing; read from the
                simulator rather than off the picture

        The same route into the history a generated step takes, so a prefix
        of real play and the rollout after it are indistinguishable.
        """
        moving = self.living
        if not moving:
            self.time += 1
            return
        self._record_actions(actions)
        self._commit_observed_actions()
        moves = torch.tensor([h.value for h in HEADINGS], device=self.device)
        for agent in moving:
            self.position[agent] += moves[int(actions[agent])]
        self.time += 1
        if live is not None:
            self.live = [was_live and bool(live[i])
                         for i, was_live in enumerate(self.live)]
        self.misses = [0] * self.num_agents
        # Everyone who took the action has an aftermath observation, even
        # if that action killed them. Only subsequent pairs are omitted.
        extra = self._new_pairs(frames[:, 0], moving)
        self._append(extra, torch.zeros(1, len(moving), dtype=torch.bool,
                                       device=self.device), moving)
        self._set_frontier(extra)

    def _commit_observed_actions(self):
        pass

    def _set_frontier(self, pairs):
        pass

    def step(self, fixed=None, denoise_steps=12, action_steps=4,
             generator=None):
        actions = self.sample_actions(fixed, action_steps, generator)
        frames, alive = self.generate_frames(actions, denoise_steps,
                                             generator)
        if alive:
            self.retire(frames, alive)
        return actions, frames


class CachedFlexRunner(FlexRunner):
    """The same rollout, with each committed step encoded once.

    The uncached runner recomputes the whole window on every denoising
    pass, which is most of the work and all of it repeated. Here a pair is
    encoded once, when both halves of it are decided, and read from the
    cache thereafter. What is being denoised is only ever the frontier --
    one pair per live agent -- so the passes that repeat are small.

    The frontier is deliberately not committed early. A pair is only
    finished once its action is chosen as well as its observation, and
    writing it before then would cache keys for content that is about to
    change.
    """

    def __init__(self, *args, **kwargs):
        from marlenv.flex_wm.cache import ScopedCache

        super().__init__(*args, **kwargs)
        self.cache = ScopedCache(len(self.model.blocks), self.device)
        self.frontier = None

    def reset(self, observations):
        super().reset(observations)
        self.cache.reset()
        self.frontier = self.pairs

    def _zero(self, count):
        return torch.zeros(1, count, device=self.device)

    @torch.no_grad()
    def _commit_frontier(self, generator=None):
        """Encode the finished frontier into the cache, once."""
        if self.frontier is None or self.frontier.pairs == 0:
            return
        pairs = self.frontier
        signal = actions_to_signal(pairs.actions,
                                   self.model.action_out.out_features)
        zero = self._zero(pairs.pairs)
        known = self.actions_known[:, -pairs.pairs:]
        if not known.all():
            signal = torch.where(known[..., None], signal,
                                 torch.randn(signal.shape, device=self.device,
                                             generator=generator))
        self.model.forward_cached(pairs, pairs.observations, signal, zero,
                                  (~known).float(), self.cache, window=self.window,
                                  record=True)
        if self.window is not None:
            self.cache.trim(self.time - self.window + 1)
        self.frontier = None

    def _commit_observed_actions(self):
        self._commit_frontier()

    def _set_frontier(self, pairs):
        self.frontier = pairs

    def _denoise(self, target, steps, generator, frame_slot):
        # Action sampling shares the base runner's identity bookkeeping.
        # Only the current frontier needs repeated model evaluation here.
        assert not frame_slot
        pairs = self.frontier
        count = pairs.pairs
        target = target[:, -count:]
        known = self.actions_known[:, -count:]
        clean_signal = actions_to_signal(pairs.actions,
                                         self.model.action_out.out_features)
        signal = torch.where(known[..., None], clean_signal,
                             torch.randn(clean_signal.shape, device=self.device,
                                         generator=generator))
        zero = self._zero(count)
        levels = self._levels(steps)
        for index in range(steps):
            level = float(levels[index])
            tau = torch.where(target, torch.full_like(zero, level),
                              (~known).float())
            _, predicted = self.model.forward_cached(
                pairs, pairs.observations, signal, zero, tau, self.cache,
                window=self.window)
            clean, noise = from_velocity(signal, predicted,
                                         torch.full_like(zero, level))
            alpha, sigma = alpha_sigma(levels[index + 1])
            signal = torch.where(target[..., None],
                                 alpha * clean.clamp(-1.0, 1.0) + sigma * noise,
                                 signal)
        full = actions_to_signal(self.pairs.actions,
                                 self.model.action_out.out_features)
        full[:, -count:] = signal
        return full

    @torch.no_grad()
    def generate_frames(self, actions, steps=12, generator=None):
        alive = self.living
        if not alive:
            self.time += 1
            return self.pairs.observations[:, :0], alive
        self._record_actions(actions)
        self._commit_frontier(generator)

        moves = torch.tensor([h.value for h in HEADINGS], device=self.device)
        for agent in alive:
            self.position[agent] = self.position[agent] + moves[
                actions[agent]]
        self.time += 1

        blank = torch.zeros(1, self.num_agents,
                            *self.pairs.observations.shape[2:],
                            device=self.device)
        fresh = self._new_pairs(blank, alive)
        zero = self._zero(fresh.pairs)
        # the action of a pair being generated is nobody's decision yet, and
        # an observation may never attend to its own action, so what sits in
        # that slot cannot reach the frames either way
        signal = torch.zeros(1, fresh.pairs,
                             self.model.action_out.out_features,
                             device=self.device)
        content = torch.randn(fresh.observations.shape, device=self.device,
                              generator=generator)
        levels = self._levels(steps)

        for index in range(steps):
            level = float(levels[index])
            predicted, _ = self.model.forward_cached(
                fresh, content, signal, torch.full_like(zero, level),
                torch.ones_like(zero), self.cache, window=self.window)
            clean, noise = from_velocity(content, predicted,
                                         torch.full_like(zero, level))
            alpha, sigma = alpha_sigma(levels[index + 1])
            content = alpha * clean.clamp(-1.0, 1.0) + sigma * noise

        fresh.observations[:] = content
        self._append(fresh, torch.zeros(1, len(alive), dtype=torch.bool,
                                        device=self.device), alive)
        self.frontier = fresh
        return content, alive
