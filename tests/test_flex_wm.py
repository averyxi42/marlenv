"""The flexible formulation must generalise the old one, not replace it."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from marlenv.core.snake import Direction  # noqa: E402
from marlenv.flex_wm.attention import (AGENT, FRAME, GLOBAL,  # noqa: E402
                                       parse_schedule, scope_mask)
from marlenv.flex_wm.data import pairs_from_arrays, unflatten  # noqa: E402
from marlenv.flex_wm.model import FlexWorldModel  # noqa: E402
from marlenv.wm.multiagent import (MultiAgentWorldModel,  # noqa: E402
                                   actions_to_signal)


def shapes(agents=3, steps=5, dim=64, depth=3, heads=4, schedule='G'):
    torch.manual_seed(0)
    old = MultiAgentWorldModel(num_agents=agents, view=9, num_actions=4,
                               frame='world', dim=dim, depth=depth,
                               heads=heads)
    new = FlexWorldModel(schedule=schedule, view=9, num_actions=4,
                         frame='world', dim=dim, depth=depth, heads=heads)
    new.load_state_dict(old.state_dict())
    return old.eval(), new.eval()


def sample(batch=2, steps=5, agents=3):
    torch.manual_seed(1)
    frames = torch.randn(batch, steps, agents, 9, 9, 3).clamp(-1, 1)
    # trailing layout: every observation has an action, which is the pair
    actions = torch.randint(0, 4, (batch, steps, agents))
    origins = torch.zeros(batch, agents, 2, dtype=torch.long)
    origins[:, 1, 0] = 4
    origins[:, 2, 1] = -3
    return frames, actions, origins


# ------------------------------------------------------------ equivalence
def test_all_global_reproduces_the_old_model_exactly():
    """The generalisation has to contain what it generalises."""
    old, new = shapes()
    frames, actions, origins = sample()
    signal = actions_to_signal(actions, 4)
    alive = torch.ones(*actions.shape, dtype=torch.bool)
    frame_tau = torch.rand(*actions.shape)
    action_tau = torch.rand(*actions.shape)

    with torch.no_grad():
        want_frames, want_actions = old(
            frames, signal, frame_tau, action_tau, origins=origins,
            action_indices=actions, alive=alive)

        pairs = pairs_from_arrays(frames, actions, origins, alive,
                                  model=old)
        flat = lambda x: x.reshape(x.shape[0], -1, *x.shape[3:])
        got_frames, got_actions = new(
            pairs, pairs.observations, flat(signal), flat(frame_tau),
            flat(action_tau))

    steps, agents = actions.shape[1], actions.shape[2]
    got_frames = unflatten(got_frames, steps, agents)
    got_actions = unflatten(got_actions, steps, agents)

    assert torch.allclose(got_frames, want_frames, atol=1e-5), (
        (got_frames - want_frames).abs().max().item())
    assert torch.allclose(got_actions, want_actions, atol=1e-5)


def test_a_checkpoint_moves_between_the_two():
    """Same module names, so the weights are interchangeable both ways."""
    old, new = shapes(schedule='FAG')
    assert set(old.state_dict()) == set(new.state_dict())
    old.load_state_dict(new.state_dict())


# ----------------------------------------------------------------- scopes
def attrs(steps=3, agents=2, per_frame=4):
    """Token attributes for a tiny sequence, laid out pair by pair."""
    time, agent, is_action = [], [], []
    for step in range(steps):
        for who in range(agents):
            time += [step] * (per_frame + 1)
            agent += [who] * (per_frame + 1)
            is_action += [False] * per_frame + [True]
    to = lambda x, d=torch.long: torch.tensor([x], dtype=d)
    return to(time), to(agent), to(is_action, torch.bool)


def test_the_scopes_nest():
    """Frame is inside agent is inside global, by construction."""
    time, agent, is_action = attrs()
    frame = scope_mask(FRAME, time, agent, is_action)
    within = scope_mask(AGENT, time, agent, is_action)
    everywhere = scope_mask(GLOBAL, time, agent, is_action)

    assert (frame <= within).all(), 'frame reaches outside agent'
    assert (within <= everywhere).all(), 'agent reaches outside global'
    assert not torch.equal(frame, within), 'the scopes are not distinct'
    assert not torch.equal(within, everywhere)


def test_frame_scope_is_one_observation():
    """And an action sees its own observation, but never the reverse."""
    time, agent, is_action = attrs(steps=2, agents=2, per_frame=4)
    mask = scope_mask(FRAME, time, agent, is_action)[0, 0]

    # the first pair occupies tokens 0..4, the second 5..9
    assert mask[0, :4].all(), 'patches cannot see their own frame'
    assert not mask[0, 4], 'an observation saw the action taken from it'
    assert mask[4, :4].all(), 'the action cannot see its own observation'
    assert not mask[0, 5:].any(), 'frame scope reached another pair'


def test_agent_scope_ignores_the_other_agent():
    time, agent, is_action = attrs(steps=3, agents=2, per_frame=4)
    mask = scope_mask(AGENT, time, agent, is_action)[0, 0]
    other = (agent[0][None, :] != agent[0][:, None])

    assert not (mask & other).any(), 'agent scope crossed identities'
    # but it does reach back in time within its own
    assert mask[10, 0], 'agent scope lost its own history'


def test_ids_are_compared_not_indexed():
    """Sparse, arbitrarily large identities behave like small ones."""
    time, agent, is_action = attrs(steps=2, agents=2, per_frame=4)
    huge = torch.where(agent == 1, torch.full_like(agent, 10 ** 9), agent)

    small = scope_mask(AGENT, time, agent, is_action)
    large = scope_mask(AGENT, time, huge, is_action)
    assert torch.equal(small, large)


def test_a_schedule_must_tile_the_depth():
    assert parse_schedule('AG', 4) == ['A', 'G', 'A', 'G']
    with pytest.raises(ValueError, match='does not tile'):
        parse_schedule('FAG', 4)
    with pytest.raises(ValueError, match='unknown attention scope'):
        parse_schedule('FXG', 3)


# ----------------------------------------------------------------- window
def test_the_training_window_matches_what_a_rollout_sees():
    """A crop wider than the window trains the same computation it plays.

    Without it, a token late in a long crop is trained seeing more history
    than the sliding cache will ever give it, which is the gap the option
    exists to close.
    """
    from marlenv.flex_wm.attention import scope_mask

    steps, agents, per_frame = 8, 1, 4
    time, agent, is_action = attrs(steps, agents, per_frame)
    window = 3
    mask = scope_mask(GLOBAL, time, agent, is_action, window=window)[0, 0]

    last = time[0, -1].item()
    for query in range(mask.shape[0]):
        reach = time[0][mask[query]]
        if reach.numel():
            span = time[0][query] - reach.min()
            assert span < window, 'a token saw past the window'

    # and the deepest token really does have the full window available
    deepest = mask.shape[0] - 1
    seen = time[0][mask[deepest]].unique()
    assert len(seen) == window, 'the window was not filled'
    assert seen.max() == last


def test_the_window_is_inert_when_the_crop_is_the_window():
    time, agent, is_action = attrs(steps=4, agents=2, per_frame=4)
    unlimited = scope_mask(GLOBAL, time, agent, is_action)
    windowed = scope_mask(GLOBAL, time, agent, is_action, window=4)
    assert torch.equal(unlimited, windowed)


def test_padding_never_attracts_attention():
    """And a padded row still has itself, so attention cannot go NaN."""
    from marlenv.flex_wm.attention import scope_mask

    time, agent, is_action = attrs(steps=2, agents=2, per_frame=4)
    valid = torch.ones_like(is_action)
    valid[0, -5:] = False                     # the last pair is padding
    mask = scope_mask(GLOBAL, time, agent, is_action, valid=valid)[0, 0]

    assert not mask[:-5, -5:].any(), 'a real token attended to padding'
    assert mask.any(dim=-1).all(), 'a row with nothing to attend to'


def test_a_loss_runs_over_a_set_of_pairs():
    from marlenv.flex_wm.train import flex_training_loss

    _, model = shapes(schedule='FAG')
    frames, actions, origins = sample()
    alive = torch.ones(*actions.shape, dtype=torch.bool)
    pairs = pairs_from_arrays(frames, actions, origins, alive, model=model)

    generator = torch.Generator().manual_seed(0)
    loss, frame_loss, action_loss = flex_training_loss(
        model, pairs, window=3, generator=generator)
    assert torch.isfinite(loss) and float(frame_loss.detach()) > 0
    loss.backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None


def test_the_agent_count_may_change_between_steps():
    """A set of pairs has no agent axis to be inconsistent about."""
    from marlenv.flex_wm.pairs import PairBatch
    from marlenv.flex_wm.train import flex_training_loss

    _, model = shapes(schedule='FAG')
    torch.manual_seed(2)
    # step 0 has two agents, step 1 has three, one of them brand new
    time = torch.tensor([[0, 0, 1, 1, 1]])
    agent = torch.tensor([[7, 9, 7, 9, 400]])
    pairs = PairBatch(observations=torch.randn(1, 5, 9, 9, 3).clamp(-1, 1),
                      actions=torch.randint(0, 4, (1, 5)),
                      agent=agent, time=time,
                      position=torch.zeros(1, 5, 2, dtype=torch.long))

    loss, _, _ = flex_training_loss(model, pairs,
                                    generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)


# ---------------------------------------------------------------- removal
def test_removal_closes_the_channel_a_retired_agent_would_leave():
    """What was in a removed pair cannot reach the loss, at all.

    Pinning a dead agent's tokens at maximum noise leaves them in the
    sequence, where their position and their number still say something.
    Removing them means there is nothing to say it with.
    """
    from marlenv.flex_wm.pairs import PairBatch, compact
    from marlenv.flex_wm.train import flex_training_loss

    _, model = shapes(schedule='FAG')
    torch.manual_seed(3)
    observations = torch.randn(1, 6, 9, 9, 3).clamp(-1, 1)
    keep = torch.tensor([[True, True, False, True, False, True]])
    build = lambda obs: PairBatch(
        observations=obs, actions=torch.arange(6)[None] % 4,
        agent=torch.tensor([[0, 1, 2, 0, 2, 1]]),
        time=torch.tensor([[0, 0, 0, 1, 1, 1]]),
        position=torch.zeros(1, 6, 2, dtype=torch.long))

    tampered = observations.clone()
    tampered[0, 2] = 5.0                       # both are removed pairs
    tampered[0, 4] = -5.0

    losses = []
    for obs in (observations, tampered):
        pairs = compact(build(obs), keep)
        losses.append(float(flex_training_loss(
            model, pairs,
            generator=torch.Generator().manual_seed(0))[0].detach()))

    assert losses[0] == pytest.approx(losses[1], rel=1e-6), (
        'a removed pair still reached the loss')


def test_removed_pairs_leave_no_tokens_behind():
    from marlenv.flex_wm.pairs import PairBatch, compact, token_attributes

    pairs = PairBatch(observations=torch.zeros(1, 4, 9, 9, 3),
                      actions=torch.zeros(1, 4, dtype=torch.long),
                      agent=torch.tensor([[3, 4, 5, 6]]),
                      time=torch.zeros(1, 4, dtype=torch.long),
                      position=torch.zeros(1, 4, 2, dtype=torch.long))
    kept = compact(pairs, torch.tensor([[True, False, True, False]]))

    assert kept.pairs == 2, 'the set was not repacked'
    assert kept.agent.tolist() == [[3, 5]]
    _, agent, _, valid = token_attributes(kept, tokens_per_frame=9)
    assert valid.all(), 'padding survived a full row'
    assert set(agent[0].tolist()) == {3, 5}


# ----------------------------------------------------------------- loading
def test_the_schedule_survives_a_checkpoint(tmp_path):
    """It is not a tensor, so it has to be carried deliberately."""
    from marlenv.flex_wm.model import load_flex_model

    _, model = shapes(schedule='FAG')
    path = tmp_path / 'model.pt'
    torch.save({'model': model.state_dict(), 'view': 9, 'dim': 64,
                'depth': 3, 'heads': 4, 'num_actions': 4, 'frame': 'world',
                'schedule': 'FAG'}, path)

    loaded, state = load_flex_model(path)
    assert loaded.schedule == ['F', 'A', 'G'], loaded.schedule
    assert state['schedule'] == 'FAG'
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def test_a_checkpoint_without_a_schedule_reads_as_global(tmp_path):
    """An older model is a global one, so the default reproduces it."""
    from marlenv.flex_wm.model import load_flex_model

    old, _ = shapes()
    path = tmp_path / 'old.pt'
    torch.save({'model': old.state_dict(), 'view': 9, 'dim': 64, 'depth': 3,
                'heads': 4, 'num_actions': 4, 'frame': 'world'}, path)

    loaded, _ = load_flex_model(path)
    assert loaded.schedule == ['G', 'G', 'G']


def test_the_recorded_schedule_drives_the_masks(tmp_path):
    """Loading must change what the model computes, not just an attribute."""
    from marlenv.flex_wm.model import load_flex_model

    _, model = shapes(schedule='FAG')
    frames, actions, origins = sample()
    signal = actions_to_signal(actions, 4)
    alive = torch.ones(*actions.shape, dtype=torch.bool)
    pairs = pairs_from_arrays(frames, actions, origins, alive, model=model)
    flat = lambda x: x.reshape(x.shape[0], -1, *x.shape[3:])
    tau = flat(torch.rand(*actions.shape))

    def run(schedule):
        path = tmp_path / f'{schedule}.pt'
        torch.save({'model': model.state_dict(), 'view': 9, 'dim': 64,
                    'depth': 3, 'heads': 4, 'num_actions': 4,
                    'frame': 'world', 'schedule': schedule}, path)
        loaded, _ = load_flex_model(path)
        with torch.no_grad():
            return loaded(pairs, pairs.observations, flat(signal), tau,
                          tau)[0]

    assert not torch.allclose(run('FAG'), run('G')), (
        'the schedule was recorded but not used')


# ----------------------------------------------------------------- runner
def runner_for(agents=3, schedule='G'):
    from marlenv.flex_wm.runner import FlexRunner

    _, model = shapes(agents=agents, schedule=schedule)
    positions = [[0, 0], [4, 0], [0, 4]][:agents]
    return model, FlexRunner(model, list(range(agents)), positions,
                             window=8, device='cpu')


def test_the_runner_keeps_the_history_it_was_given():
    """Bookkeeping first: what went in must be what comes back out."""
    torch.manual_seed(4)
    model, runner = runner_for()
    first = torch.randn(1, 1, 3, 9, 9, 3).clamp(-1, 1)
    runner.reset(first)
    assert torch.equal(runner.frames[0, -1], first[0, 0])

    second = torch.randn(1, 1, 3, 9, 9, 3).clamp(-1, 1)
    runner.observe(torch.tensor([0, 1, 2]), second,
                   torch.tensor([True, True, True]))
    assert torch.equal(runner.frames[0, -1], second[0, 0]), (
        'the newest observation was not the one supplied')
    assert runner.pairs.pairs == 6, 'a pair went missing'


def test_denoising_leaves_the_history_alone():
    """The step being generated must not overwrite what conditions it.

    Filling the whole content tensor with noise instead of only the slot
    being denoised erases the context, which shows up as a model that
    generates confidently from nothing.
    """
    torch.manual_seed(5)
    model, runner = runner_for()
    first = torch.randn(1, 1, 3, 9, 9, 3).clamp(-1, 1)
    runner.reset(first)
    before = runner.pairs.observations.clone()

    generator = torch.Generator().manual_seed(0)
    runner.step(denoise_steps=2, action_steps=2, generator=generator)

    assert torch.equal(runner.pairs.observations[:, :3], before), (
        'generating a step disturbed the history it was conditioned on')


def test_positions_dead_reckon_from_the_actions_taken():
    from marlenv.wm.model import HEADINGS

    torch.manual_seed(6)
    model, runner = runner_for(agents=1)
    runner.reset(torch.randn(1, 1, 1, 9, 9, 3).clamp(-1, 1))
    start = runner.position[0].clone()

    heading = HEADINGS.index(Direction.DOWN)
    runner.observe(torch.tensor([heading]),
                   torch.randn(1, 1, 1, 9, 9, 3).clamp(-1, 1))
    moved = torch.tensor(Direction.DOWN.value)
    assert torch.equal(runner.position[0], start + moved)


def test_a_retired_agent_stops_contributing_pairs():
    torch.manual_seed(7)
    model, runner = runner_for()
    runner.reset(torch.randn(1, 1, 3, 9, 9, 3).clamp(-1, 1))
    runner.live[1] = False

    before = runner.pairs.pairs
    runner.observe(torch.tensor([0, 1, 2]),
                   torch.randn(1, 1, 3, 9, 9, 3).clamp(-1, 1))
    assert runner.pairs.pairs == before + 2, (
        'a retired agent still added a pair')
    assert runner.living == [0, 2]


def test_the_window_bounds_the_history():
    torch.manual_seed(8)
    model, runner = runner_for(agents=1)
    runner.reset(torch.randn(1, 1, 1, 9, 9, 3).clamp(-1, 1))
    for _ in range(12):
        runner.observe(torch.tensor([0]),
                       torch.randn(1, 1, 1, 9, 9, 3).clamp(-1, 1))
    assert runner.pairs.pairs <= 8, 'the window did not trim'
    span = runner.pairs.time.max() - runner.pairs.time.min()
    assert span < 8


# ------------------------------------------------------------------ cache
def slice_pairs(pairs, start, stop):
    from marlenv.flex_wm.pairs import PairBatch

    cut = lambda x: x[:, start:stop]
    return PairBatch(observations=cut(pairs.observations),
                     actions=cut(pairs.actions), agent=cut(pairs.agent),
                     time=cut(pairs.time), position=cut(pairs.position))


@pytest.mark.parametrize('schedule,depth',
                         [('G', 3), ('AG', 4), ('FAG', 3),
                          ('FAGFAGAAGAAG', 12)])
def test_the_cache_agrees_with_recomputing_everything(schedule, depth):
    """Two paths, one answer -- at every scope, not just the global one.

    A frame-scope block reads nothing from the cache and an agent-scope
    block reads only its own past, so this is where getting the scoping
    wrong would show.
    """
    from marlenv.flex_wm.cache import ScopedCache

    _, model = shapes(agents=3, depth=depth, schedule=schedule)
    frames, actions, origins = sample(batch=1, steps=4, agents=3)
    signal = actions_to_signal(actions, 4)
    alive = torch.ones(*actions.shape, dtype=torch.bool)
    pairs = pairs_from_arrays(frames, actions, origins, alive, model=model)
    flat = lambda x: x.reshape(1, -1, *x.shape[3:])
    frame_tau = torch.zeros(1, pairs.pairs)
    action_tau = torch.zeros(1, pairs.pairs)

    with torch.no_grad():
        want_frames, want_actions = model(pairs, pairs.observations,
                                          flat(signal), frame_tau,
                                          action_tau)

        cache = ScopedCache(len(model.blocks))
        agents, got_frames, got_actions = 3, None, None
        for step in range(4):
            lo, hi = step * agents, (step + 1) * agents
            got_frames, got_actions = model.forward_cached(
                slice_pairs(pairs, lo, hi), pairs.observations[:, lo:hi],
                flat(signal)[:, lo:hi], frame_tau[:, lo:hi],
                action_tau[:, lo:hi], cache, record=True)

    assert torch.allclose(got_frames, want_frames[:, -3:], atol=1e-5), (
        (got_frames - want_frames[:, -3:]).abs().max().item())
    assert torch.allclose(got_actions, want_actions[:, -3:], atol=1e-5)


def test_a_frame_scope_block_never_reads_the_cache():
    """Its reach is the observation it belongs to, so there is nothing to
    read -- and that is where the saving comes from."""
    from marlenv.flex_wm.cache import ScopedCache

    cache = ScopedCache(layers=1)
    key = torch.randn(1, 2, 5, 8)
    cache.write(0, key, key)
    cache.commit(torch.zeros(1, 5, dtype=torch.long),
                 torch.zeros(1, 5, dtype=torch.long),
                 torch.zeros(1, 5, dtype=torch.bool))

    fresh = torch.randn(1, 2, 3, 8)
    kept, _ = cache.read(0, fresh, fresh, FRAME)
    assert kept.shape[2] == 3, 'frame scope pulled history in'
    wider, _ = cache.read(0, fresh, fresh, GLOBAL)
    assert wider.shape[2] == 8, 'global scope lost the cache'


def test_the_cache_trims_by_frame():
    from marlenv.flex_wm.cache import ScopedCache

    cache = ScopedCache(layers=1)
    for step in range(4):
        key = torch.randn(1, 2, 2, 8)
        cache.write(0, key, key)
        cache.commit(torch.full((1, 2), step, dtype=torch.long),
                     torch.zeros(1, 2, dtype=torch.long),
                     torch.zeros(1, 2, dtype=torch.bool))
    assert len(cache) == 8
    cache.trim(oldest=2)
    assert len(cache) == 4, 'trimming kept the wrong tokens'
    assert int(cache.time.min()) == 2


def test_both_runners_keep_the_same_books():
    """Fed the same real play, the two must agree on state.

    Not on generated pixels -- they draw different shaped noise, so their
    random streams diverge and comparing samples would be comparing seeds.
    What has to match is everything the cache is not allowed to change:
    who is alive, where they are, and what history is held.
    """
    from marlenv.flex_wm.runner import CachedFlexRunner, FlexRunner

    _, model = shapes(agents=3, depth=3, schedule='FAG')
    torch.manual_seed(9)
    start = torch.randn(1, 1, 3, 9, 9, 3).clamp(-1, 1)
    steps = [(torch.tensor([0, 1, 2]),
              torch.randn(1, 1, 3, 9, 9, 3).clamp(-1, 1)) for _ in range(4)]

    books = []
    for cls in (FlexRunner, CachedFlexRunner):
        runner = cls(model, [0, 1, 2], [[0, 0], [4, 0], [0, 4]], window=8,
                     device='cpu')
        runner.reset(start.clone())
        for actions, frames in steps:
            runner.observe(actions, frames.clone())
        books.append((runner.pairs.pairs, runner.living,
                      runner.position.clone(), runner.time,
                      runner.pairs.time.clone(),
                      runner.pairs.agent.clone(),
                      runner.pairs.actions.clone()))

    plain, cached = books
    assert plain[0] == cached[0], 'different amounts of history'
    assert plain[1] == cached[1], 'different agents alive'
    assert torch.equal(plain[2], cached[2]), 'positions drifted apart'
    assert plain[3] == cached[3]
    for left, right in zip(plain[4:], cached[4:]):
        assert torch.equal(left, right), 'the recorded pairs differ'


def test_the_cache_holds_what_was_committed():
    """One entry per token of every finished pair, and none of the frontier."""
    from marlenv.flex_wm.runner import CachedFlexRunner

    _, model = shapes(agents=2, depth=3, schedule='FAG')
    torch.manual_seed(10)
    runner = CachedFlexRunner(model, [0, 1], [[0, 0], [3, 0]], window=8,
                              device='cpu')
    runner.reset(torch.randn(1, 1, 2, 9, 9, 3).clamp(-1, 1))
    assert len(runner.cache) == 0, 'the frontier was cached before it closed'

    runner.observe(torch.tensor([0, 1]),
                   torch.randn(1, 1, 2, 9, 9, 3).clamp(-1, 1))
    per_pair = model.tokens_per_pair
    assert len(runner.cache) == 2 * per_pair, (
        'a finished pair did not reach the cache')


def test_reporting_the_newest_frames_does_not_scan_the_history():
    """It is read far more often than the step it reports on.

    Reconstructing it from the pair set means a Python pass over every pair
    with a device synchronisation each, which cost five times the step
    itself and grew with the window. Holding it instead keeps the cost
    flat, so this pins that the history is not being walked.
    """
    from marlenv.flex_wm.runner import CachedFlexRunner

    _, model = shapes(agents=2, depth=3, schedule='FAG')
    torch.manual_seed(11)
    runner = CachedFlexRunner(model, [0, 1], [[0, 0], [3, 0]], window=16,
                              device='cpu')
    runner.reset(torch.randn(1, 1, 2, 9, 9, 3).clamp(-1, 1))

    newest = None
    for _ in range(6):
        newest = torch.randn(1, 1, 2, 9, 9, 3).clamp(-1, 1)
        runner.observe(torch.tensor([0, 1]), newest.clone())

    assert torch.equal(runner.frames, newest), (
        'the reported frames are not the newest ones')
    # held rather than derived: no dependence on how much history there is
    assert runner.frames is runner.latest
