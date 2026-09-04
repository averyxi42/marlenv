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


# ---------------------------------------------------------------- ratchet
def test_the_ratchet_tally_separates_losing_from_gaining():
    """The asymmetry is the measurement; a total would hide it.

    Two models can drop the same number of cells and behave completely
    differently in a rollout: one that also adds cells wanders around the
    right length, one that never does can only decay.
    """
    from marlenv.core.snake import Cell
    from marlenv.grading.ratchet import Tally, own_length, snake_cells

    truth = np.zeros((5, 5), dtype=np.int64)
    truth[2, 2] = Cell.HEAD.value            # the viewer's own head
    truth[2, 3] = Cell.BODY.value
    truth[2, 4] = Cell.TAIL.value
    assert own_length(truth) == 3
    assert snake_cells(truth).sum() == 3

    shed = truth.copy(); shed[2, 4] = 0      # lost the tail
    grew = truth.copy(); grew[3, 3] = Cell.BODY.value   # invented one

    tally = Tally(steps=2)
    tally.add(0, truth, shed)
    tally.add(1, truth, grew)
    assert tally.lost.tolist() == [1.0, 0.0]
    assert tally.gained.tolist() == [0.0, 1.0]
    assert tally.dreamt.tolist() == [2.0, 4.0]


def test_own_length_counts_only_the_viewer_s_snake():
    """The centre is the viewer's head, so its colour says which is which."""
    from marlenv.core.snake import Cell
    from marlenv.grading.ratchet import own_length

    grid = np.zeros((5, 5), dtype=np.int64)
    grid[2, 2] = Cell.HEAD.value                  # snake 0, the viewer
    grid[2, 3] = Cell.BODY.value
    grid[0, 0] = 10 + Cell.HEAD.value             # snake 1, someone else
    grid[0, 1] = 10 + Cell.BODY.value
    assert own_length(grid) == 2


def test_the_single_agent_adapter_needs_to_be_told_its_action():
    """It has no policy head, so silently sampling one would be a lie."""
    from marlenv.wm.model import WorldModel
    from marlenv.wm.runner import SingleAgentAdapter

    torch.manual_seed(0)
    model = WorldModel(view=9, dim=64, depth=2, heads=4, num_actions=4,
                       frame='world', align_coords=True)
    adapter = SingleAgentAdapter(model, agent=1, window=8, device='cpu')
    adapter.reset(torch.randn(1, 1, 3, 9, 9, 3).clamp(-1, 1))

    assert adapter.frames.shape[2] == 3, 'the report lost its agent axis'
    assert adapter.living == [1]
    with pytest.raises(ValueError, match='does not choose one'):
        adapter.step(fixed={0: 2}, denoise_steps=1)

    adapter.step(fixed={1: 2}, denoise_steps=1)
    assert adapter.frames.shape == (1, 1, 3, 9, 9, 3)


def test_every_package_imports_on_its_own():
    """In a fresh interpreter, with nothing imported first.

    A cycle between packages only shows up when the wrong one is imported
    first, so a suite that has already pulled in half the tree will not
    notice it. Each of these has to stand up unaided.
    """
    import subprocess
    import sys

    for module in ('marlenv', 'marlenv.wm', 'marlenv.flex_wm',
                   'marlenv.grading', 'marlenv.grading.ratchet',
                   'marlenv.grading.frames', 'marlenv.data',
                   'marlenv.policies'):
        done = subprocess.run([sys.executable, '-c', f'import {module}'],
                              capture_output=True, text=True)
        assert done.returncode == 0, (
            f'{module} does not import on its own:\n'
            + done.stderr.strip().splitlines()[-1])


# ------------------------------------------------------------- visibility
def test_partial_observation_changes_nothing_when_all_is_visible():
    """The refinement has to be invisible until it is used.

    ``visible`` says per patch what ``trained`` says per observation, so
    with every patch visible the loss must be the number it was before --
    identically, not approximately. Patches partition the view evenly, so
    averaging within patches and then across them is the same average.
    """
    from marlenv.flex_wm.pairs import PairBatch
    from marlenv.flex_wm.train import flex_training_loss

    _, model = shapes(schedule='FAG')
    frames, actions, origins = sample()
    alive = torch.ones(*actions.shape, dtype=torch.bool)
    pairs = pairs_from_arrays(frames, actions, origins, alive, model=model)

    tokens = model.tokens_per_frame
    spelled = PairBatch(
        observations=pairs.observations, actions=pairs.actions,
        agent=pairs.agent, time=pairs.time, position=pairs.position,
        valid=pairs.valid, trained=pairs.trained, acted=pairs.acted,
        visible=torch.ones(pairs.batch, pairs.pairs, tokens,
                           dtype=torch.bool))

    losses = []
    for batch in (pairs, spelled):
        generator = torch.Generator().manual_seed(0)
        losses.append([float(v.detach()) for v in
                       flex_training_loss(model, batch, generator=generator)])
    assert losses[0] == pytest.approx(losses[1], rel=1e-9), (
        'spelling out full visibility changed the answer')


def test_an_unseen_patch_cannot_reach_the_loss():
    """Not merely down-weighted: its content must not matter at all."""
    from marlenv.flex_wm.pairs import PairBatch
    from marlenv.flex_wm.train import flex_training_loss

    _, model = shapes(schedule='FAG')
    frames, actions, origins = sample()
    alive = torch.ones(*actions.shape, dtype=torch.bool)
    base = pairs_from_arrays(frames, actions, origins, alive, model=model)

    tokens = model.tokens_per_frame
    visible = torch.ones(base.batch, base.pairs, tokens, dtype=torch.bool)
    visible[:, :, 0] = False                   # the first patch is unseen

    def build(observations):
        return PairBatch(
            observations=observations, actions=base.actions,
            agent=base.agent, time=base.time, position=base.position,
            valid=base.valid, trained=base.trained, acted=base.acted,
            visible=visible)

    view = base.observations.shape[2]
    grid = int(round(tokens ** 0.5))
    patch = view // grid
    tampered = base.observations.clone()
    tampered[:, :, :patch, :patch] = 0.9        # only the unseen patch

    losses = []
    for observations in (base.observations, tampered):
        generator = torch.Generator().manual_seed(1)
        losses.append(float(flex_training_loss(
            model, build(observations), generator=generator)[1].detach()))
    assert losses[0] == pytest.approx(losses[1], rel=1e-6), (
        'an unseen patch reached the frame loss')


def test_cell_mask_spreads_a_patch_over_its_cells():
    from marlenv.flex_wm.pairs import PairBatch

    pairs = PairBatch(observations=torch.zeros(1, 1, 9, 9, 3),
                      actions=torch.zeros(1, 1, dtype=torch.long),
                      agent=torch.zeros(1, 1, dtype=torch.long),
                      time=torch.zeros(1, 1, dtype=torch.long),
                      position=torch.zeros(1, 1, 2, dtype=torch.long),
                      visible=torch.tensor([[[True, False, True,
                                              True, True, True,
                                              True, True, True]]]))
    cells = pairs.cell_mask(9, 9)[0, 0]
    assert cells.shape == (9, 9)
    assert cells[:3, :3].all(), 'the first patch should be visible'
    assert not cells[:3, 3:6].any(), 'the second patch should not be'
    assert cells[:3, 6:].all()
    assert cells[3:].all(), 'later rows were untouched'


# ------------------------------------------------------------- egocentric
def test_a_visit_yields_one_fewer_action_than_observations():
    """An action is a difference of positions, so the last one is missing."""
    from marlenv.flex_wm.egocentric import visible_runs

    assert visible_runs([0, 1, 1, 1, 0, 1, 1, 0]) == [(1, 4), (5, 7)]
    # a single glimpse teaches nothing about behaviour and is dropped
    assert visible_runs([1, 0, 1, 0, 1]) == []
    assert visible_runs([1, 1]) == [(0, 2)]
    assert visible_runs([0, 0, 1, 1, 1]) == [(2, 5)]


def test_a_patch_counts_only_when_all_of_it_was_seen():
    """Half a patch is not a patch: the token carries the whole block."""
    from marlenv.flex_wm.egocentric import patch_visibility

    offsets = np.array([[-3, 0], [0, 0], [3, 0]])
    # watcher and target on the same cell: everything is inside
    assert patch_visibility((0, 0), (0, 0), offsets, radius=4,
                            patch=3).all()
    # slide the target away and the far patch leaves the watcher's view
    seen = patch_visibility((0, 0), (2, 0), offsets, radius=4, patch=3)
    assert seen.tolist() == [True, True, False]


def test_each_visit_gets_an_identity_of_its_own():
    """Nothing survives an absence; a returning snake is a new agent."""
    from marlenv.flex_wm.egocentric import egocentric_episode

    frames, agents, view = 12, 2, 9
    poses = np.zeros((frames, agents, 3), np.int64)
    alive = np.ones((frames, agents), bool)
    cardinal = np.zeros((frames, agents, 4), np.int64)
    cardinal[..., 1] = 1
    poses[:, 0] = (5, 5, 1)
    # the other snake is near, then far, then near again
    for t in range(frames):
        poses[t, 1] = (5, 6, 1) if t < 4 or t >= 8 else (5, 40, 1)
    episode = {'alive_mask': alive, 'poses': poses,
               'observations': np.zeros((frames, agents, view, view, 3),
                                        np.uint8),
               'cardinal_actions': cardinal}

    offsets = np.array([[r, c] for r in (-3, 0, 3) for c in (-3, 0, 3)])
    ego = egocentric_episode(episode, offsets, ego=0, radius=4)

    others = sorted(set(ego.agent[ego.agent != ego.agent[0]].tolist()))
    assert len(others) == 2, 'the two visits were treated as one agent'
    for identity in others:
        rows = ego.agent == identity
        assert ego.acted[rows].sum() == rows.sum() - 1, (
            'a visit should give one fewer action than observations')


def test_the_observer_sees_all_of_its_own_view():
    from marlenv.flex_wm.egocentric import egocentric_episode

    frames, agents, view = 6, 2, 9
    poses = np.zeros((frames, agents, 3), np.int64)
    poses[:, 0] = (5, 5, 1)
    poses[:, 1] = (40, 40, 1)          # never in view
    alive = np.ones((frames, agents), bool)
    cardinal = np.zeros((frames, agents, 4), np.int64)
    cardinal[..., 1] = 1
    episode = {'alive_mask': alive, 'poses': poses,
               'observations': np.zeros((frames, agents, view, view, 3),
                                        np.uint8),
               'cardinal_actions': cardinal}
    offsets = np.array([[r, c] for r in (-3, 0, 3) for c in (-3, 0, 3)])
    ego = egocentric_episode(episode, offsets, ego=0, radius=4)

    assert ego.visible.all(), 'the observer should see all of its own view'
    assert len(set(ego.agent.tolist())) == 1, 'nobody else was visible'
    assert ego.acted[:-1].all() and not ego.acted[-1]


def test_two_agents_agree_on_what_they_both_see():
    """The premise of the whole reconstruction, stated as a test.

    An observed agent's view is copied from the record rather than
    re-rendered, so it is only usable if both agents' pictures of a shared
    cell are the same picture. They are -- once both are turned north-up,
    since the record keeps each in its own head frame.
    """
    import gymnasium as gym
    import marlenv  # noqa: F401
    from marlenv.flex_wm.egocentric import head_in_view
    from marlenv.grading.compare import unrotate_view
    from marlenv.wm.model import HEADINGS

    env = gym.make('Snake-v1', height=15, width=15, num_snakes=3,
                   num_fruits=4, view_radius=4, observation_noise=2.0,
                   snake_noise_sigma=8.0, background_gradient=0.0,
                   disable_env_checker=True)
    env.reset(seed=3)
    base = env.unwrapped
    radius = 4

    checked = 0
    for _ in range(20):
        views = base.egocentric_rgb()
        upright = [unrotate_view(view, snake.direction)
                   for view, snake in zip(views, base.snakes)]
        for a, first in enumerate(base.snakes):
            for b, second in enumerate(base.snakes):
                if b <= a or not (first.alive and second.alive):
                    continue
                here = np.array(first.head_coord)
                there = np.array(second.head_coord)
                if not head_in_view(here, there, radius):
                    continue
                shift = there - here
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        # the same world cell, seen from the other head
                        row, col = dr - shift[0], dc - shift[1]
                        if abs(row) > radius or abs(col) > radius:
                            continue
                        checked += 1
                        assert np.array_equal(
                            upright[a][dr + radius, dc + radius],
                            upright[b][row + radius, col + radius]), (
                            'two agents disagree about one world cell')
        env.step([0] * base.num_snakes)

    assert checked > 100, f'only {checked} overlapping cells were compared'


# ------------------------------------------------------- one batcher, both
def rectangular_set(steps=6, agents=2, view=9, tokens=9, seed=0):
    from marlenv.flex_wm.batch import flatten_episode

    rng = np.random.default_rng(seed)
    alive = np.ones((steps, agents), bool)
    alive[steps - 2:, 1] = False              # one agent leaves early
    return flatten_episode(
        observations=rng.integers(0, 256, (steps, agents, view, view, 3),
                                  dtype=np.uint8),
        actions=rng.integers(0, 4, (steps, agents)),
        alive=alive, trained=alive,
        positions=rng.integers(0, 9, (steps, agents, 2)),
        tokens=tokens)


def test_one_batcher_serves_both_kinds_of_episode():
    """A rectangle and a set of visits should take the same path.

    The point of pairs is that an episode with a moving agent count is not
    a special case. A batcher that crops rectangles and converts at the end
    cannot express one; a batcher that crops pairs does not notice.
    """
    from marlenv.flex_wm.batch import PairSetBatcher
    from marlenv.flex_wm.egocentric import egocentric_episode

    offsets = np.array([[r, c] for r in (-3, 0, 3) for c in (-3, 0, 3)])
    frames, agents, view = 14, 2, 9
    poses = np.zeros((frames, agents, 3), np.int64)
    poses[:, 0] = (5, 5, 1)
    for t in range(frames):
        poses[t, 1] = (5, 6, 1) if t < 5 or t >= 9 else (5, 40, 1)
    cardinal = np.zeros((frames, agents, 4), np.int64)
    cardinal[..., 1] = 1
    ego = egocentric_episode(
        {'alive_mask': np.ones((frames, agents), bool), 'poses': poses,
         'observations': np.zeros((frames, agents, view, view, 3), np.uint8),
         'cardinal_actions': cardinal}, offsets, ego=0, radius=4)

    ragged = {'observations': ego.observations, 'actions': ego.actions,
              'agent': ego.agent, 'time': ego.time,
              'position': ego.position, 'visible': ego.visible,
              'acted': ego.acted, 'trained': np.ones(len(ego), bool)}

    for name, episodes in (('rectangular', [rectangular_set()]),
                           ('egocentric', [ragged]),
                           ('mixed', [rectangular_set(), ragged])):
        batcher = PairSetBatcher(episodes, context=6, seed=0)
        pairs, weight, dropout = batcher.batch(4)
        assert pairs.batch == 4, name
        assert pairs.visible.shape[:2] == pairs.observations.shape[:2], name
        assert pairs.valid.any(), name
        # padding must not be mistaken for a real identity
        assert (pairs.agent[~pairs.valid] == -1).all(), name


def test_a_crop_keeps_only_its_window():
    from marlenv.flex_wm.batch import PairSetBatcher

    batcher = PairSetBatcher([rectangular_set(steps=20)], context=5, seed=3)
    for _ in range(8):
        crop = batcher.crop(0)
        if len(crop['time']):
            assert crop['time'].min() >= 0
            assert crop['time'].max() < 5, 'a crop reached past its window'


def test_the_flat_set_drops_what_was_never_alive():
    from marlenv.flex_wm.batch import flatten_episode

    steps, agents = 5, 2
    alive = np.ones((steps, agents), bool)
    alive[3:, 1] = False
    flat = flatten_episode(
        observations=np.zeros((steps, agents, 9, 9, 3), np.uint8),
        actions=np.zeros((steps, agents), np.int64), alive=alive,
        trained=alive, positions=np.zeros((steps, agents, 2), np.int64),
        tokens=9)
    assert len(flat['time']) == int(alive.sum())
    assert flat['visible'].shape == (int(alive.sum()), 9)


def test_a_patch_marked_visible_is_one_the_observer_could_reconstruct():
    """The claim `visible` makes, checked against the world.

    Not just that the arithmetic is self consistent: for every patch the
    reconstruction keeps, all nine of its cells must be inside the
    observer's own view and hold the pixels the observer sees there. Both
    views are north-up, so a patch offset is a world offset -- if that
    rotation were wrong this is where it would show.
    """
    import gymnasium as gym
    import marlenv  # noqa: F401
    from marlenv.flex_wm.egocentric import head_in_view, patch_visibility
    from marlenv.grading.compare import unrotate_view

    env = gym.make('Snake-v1', height=15, width=15, num_snakes=3,
                   num_fruits=4, view_radius=4, observation_noise=2.0,
                   snake_noise_sigma=8.0, background_gradient=0.0,
                   disable_env_checker=True)
    env.reset(seed=11)
    base = env.unwrapped
    radius, patch = 4, 3
    offsets = np.array([[r, c] for r in (-3, 0, 3) for c in (-3, 0, 3)])

    checked = 0
    for _ in range(25):
        upright = [unrotate_view(view, snake.direction) for view, snake
                   in zip(base.egocentric_rgb(), base.snakes)]
        for ego, watcher in enumerate(base.snakes):
            for other, target in enumerate(base.snakes):
                if other == ego or not (watcher.alive and target.alive):
                    continue
                here = np.array(watcher.head_coord)
                there = np.array(target.head_coord)
                if not head_in_view(here, there, radius):
                    continue
                seen = patch_visibility(here, there, offsets, radius, patch)
                for index, offset in enumerate(offsets):
                    if not seen[index]:
                        continue
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            # the cell, in the observed agent's own view
                            row = offset[0] + dr
                            col = offset[1] + dc
                            # and the same world cell in the observer's
                            world = there + (row, col)
                            mine = world - here
                            assert abs(mine[0]) <= radius, 'outside the view'
                            assert abs(mine[1]) <= radius, 'outside the view'
                            checked += 1
                            assert np.array_equal(
                                upright[other][row + radius, col + radius],
                                upright[ego][mine[0] + radius,
                                             mine[1] + radius]), (
                                'a patch called visible holds pixels the '
                                'observer does not see')
        env.step([0] * base.num_snakes)

    assert checked > 200, f'only {checked} cells were compared'
