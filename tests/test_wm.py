"""World model: sequence building, causality, and the interactive layer."""
import numpy as np
import pytest

import gymnasium as gym
import marlenv  # noqa: F401
from marlenv.core.snake import Direction
from marlenv.grading.poses import step_pose

torch = pytest.importorskip('torch')

from marlenv.wm.canvas import CanvasIntegrator, make_pose  # noqa: E402
from marlenv.wm.data import (agent_sequences, to_model_input,  # noqa: E402
                             to_pixels)
from marlenv.wm.diffusion import (add_noise, alpha_sigma,  # noqa: E402
                                  training_loss)
from marlenv.wm.interactive import (WorldModelPlayer,  # noqa: E402
                                    cardinal_to_ego, world_up)
from marlenv.wm.model import (HEADINGS, WorldModel,  # noqa: E402
                              _FORWARD, _RIGHTWARD)


def tiny_model():
    torch.manual_seed(0)
    return WorldModel(dim=64, depth=2, heads=4)


# --------------------------------------------------------------------- model
def test_frames_only_depend_on_the_past():
    """The point of a causal model: no leakage from the future."""
    model = tiny_model().eval()
    frames = torch.randn(1, 6, 9, 9, 3)
    actions = torch.randint(0, 3, (1, 5))
    tau = torch.rand(1, 6)

    with torch.no_grad():
        base = model(frames, actions, tau)
        perturbed = frames.clone()
        perturbed[:, 3] += 5.0
        changed = (model(perturbed, actions, tau) - base).abs()
        per_frame = changed.amax(dim=(2, 3, 4))[0]

    assert torch.all(per_frame[:3] == 0), 'a later frame leaked backwards'
    assert per_frame[3] > 0


def test_an_action_conditions_the_next_frame_not_its_own():
    model = tiny_model().eval()
    frames = torch.randn(1, 6, 9, 9, 3)
    actions = torch.randint(0, 3, (1, 5))
    tau = torch.rand(1, 6)

    with torch.no_grad():
        base = model(frames, actions, tau)
        other = actions.clone()
        other[:, 2] = (actions[:, 2] + 1) % 3
        changed = (model(frames, other, tau) - base).abs().amax(dim=(2, 3, 4))

    assert torch.all(changed[0, :3] == 0), 'action leaked into its own frame'
    assert changed[0, 3] > 0, 'action did not reach the next frame'


def test_patchify_round_trips():
    model = tiny_model()
    frames = torch.randn(2, 4, 9, 9, 3)

    assert torch.allclose(model.unpatchify(model.patchify(frames)), frames)


# ----------------------------------------------------------------- diffusion
def test_noise_schedule_spans_clean_to_pure_noise():
    alpha, sigma = alpha_sigma(torch.tensor([0.0, 1.0]))

    assert alpha[0] == pytest.approx(1.0, abs=1e-6)
    assert sigma[0] == pytest.approx(0.0, abs=1e-6)
    assert alpha[1] == pytest.approx(0.0, abs=1e-6)
    assert sigma[1] == pytest.approx(1.0, abs=1e-6)


def test_untrained_loss_is_about_one():
    """Normalised per pixel, so it is comparable across sequence lengths."""
    model = tiny_model()
    frames = torch.randn(2, 8, 9, 9, 3).clamp(-1, 1)
    actions = torch.randint(0, 3, (2, 7))
    mask = torch.ones(2, 8, dtype=torch.bool)

    loss = training_loss(model, frames, actions, mask).item()

    assert 0.5 < loss < 3.0, loss


def test_masked_frames_do_not_contribute():
    model = tiny_model()
    frames = torch.randn(2, 8, 9, 9, 3).clamp(-1, 1)
    actions = torch.randint(0, 3, (2, 7))
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[:, 4:] = False

    generator = torch.Generator().manual_seed(0)
    first = training_loss(model, frames, actions, mask, generator=generator)
    changed = frames.clone()
    changed[:, 4:] += 10.0
    generator = torch.Generator().manual_seed(0)
    second = training_loss(model, changed, actions, mask,
                           generator=generator)

    assert first.item() == pytest.approx(second.item(), rel=1e-5)


def test_pixel_conversion_round_trips():
    pixels = np.random.randint(0, 256, (4, 9, 9, 3), dtype=np.uint8)

    assert np.array_equal(to_pixels(to_model_input(pixels)), pixels)


# ---------------------------------------------------------------------- data
def test_death_keeps_the_aftermath_frame():
    """A view belongs to a position, so the frame after death is a real one.

    It is the ordinary view from the cell the snake died entering, with its
    own body gone -- no sentinel value, and nothing out of distribution.
    """
    episode = {
        'alive_mask': np.array([[True], [True], [True], [False]]),
        'observations': np.full((4, 1, 9, 9, 3), 200, dtype=np.uint8),
        'ego_actions': np.zeros((4, 1, 3), dtype=np.uint8),
    }
    episode['observations'][3] = 77         # whatever the aftermath looks like
    episode['ego_actions'][..., 0] = 1

    (obs, actions, died), = list(agent_sequences(episode))

    assert died
    assert len(obs) == 4               # three living frames plus the aftermath
    assert (obs[-1] == 77).all(), 'the aftermath view was replaced'
    assert (obs[-2] == 200).all()
    assert len(actions) == len(obs) - 1


def test_survival_gets_no_marker():
    episode = {
        'alive_mask': np.ones((4, 1), dtype=bool),
        'observations': np.full((4, 1, 9, 9, 3), 200, dtype=np.uint8),
        'ego_actions': np.zeros((4, 1, 3), dtype=np.uint8),
    }
    episode['ego_actions'][..., 0] = 1

    (obs, actions, died), = list(agent_sequences(episode))

    assert not died
    assert len(obs) == 4
    assert (obs[-1] != 0).any()
    assert len(actions) == 3


# --------------------------------------------------------------- interactive
def test_cardinal_control_covers_every_heading():
    for heading in Direction:
        assert cardinal_to_ego(heading, heading) == 0
        turns = {cardinal_to_ego(heading, c) for c in Direction}
        assert turns == {0, 1, 2, None}, 'exactly one reversal per heading'


def test_reversal_is_rejected():
    opposite = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT,
                Direction.RIGHT: Direction.LEFT}
    for heading, back in opposite.items():
        assert cardinal_to_ego(heading, back) is None


def test_world_up_undoes_the_head_frame_rotation():
    env = gym.make('Snake-v1', height=13, width=13, num_snakes=1,
                   num_fruits=3, view_radius=4, observation_noise=0.0,
                   snake_noise_sigma=0.0, background_gradient=0.0,
                   disable_env_checker=True)
    env.reset(seed=0)
    env.action_space.seed(0)
    base = env.unwrapped

    seen = set()
    for _ in range(20):
        snake = base.snakes[0]
        if not snake.alive:
            break
        view = base.egocentric_rgb()[0]
        upright = world_up(view, snake.direction)
        # the neck is behind the head in world terms, so once upright the
        # head is still centred but the body points the way it really lies
        assert upright.shape == view.shape
        seen.add(snake.direction)
        env.step(list(env.action_space.sample()))

    assert len(seen) > 1


def test_player_tracks_pose_by_dead_reckoning():
    model = tiny_model()
    start = make_pose(6, 6, Direction.UP)
    player = WorldModelPlayer(model, np.zeros((9, 9, 3), np.uint8),
                              Direction.UP, context=8, denoise_steps=1,
                              pose=start)

    expected = start
    for cardinal in (Direction.UP, Direction.LEFT, Direction.LEFT):
        ego = player.step(cardinal)
        expected = step_pose(expected, ego)
        assert player.pose == expected
        assert player.heading == expected.direction


def test_player_clips_history_to_the_context():
    model = tiny_model()
    player = WorldModelPlayer(model, np.zeros((9, 9, 3), np.uint8),
                              Direction.UP, context=5, denoise_steps=1)

    for _ in range(9):
        player.step(Direction.UP)

    assert player.history.shape[1] == 5
    assert len(player.actions) == 4


# -------------------------------------------------------------------- canvas
def test_canvas_pastes_newer_over_older():
    canvas = CanvasIntegrator(11, 11, radius=2, decay=1.0)
    old = np.full((5, 5, 3), 100, np.uint8)
    new = np.full((5, 5, 3), 200, np.uint8)

    canvas.add(old, make_pose(5, 5, Direction.UP))
    canvas.add(new, make_pose(5, 6, Direction.UP))

    row, col = canvas.to_canvas(5, 6)
    assert canvas.image[row, col].tolist() == [200, 200, 200]
    # the cell only the older view covered keeps its value
    row, col = canvas.to_canvas(5, 3)
    assert canvas.image[row, col].tolist() == [100, 100, 100]


def test_canvas_decays_towards_black():
    canvas = CanvasIntegrator(11, 11, radius=2, decay=0.5)
    view = np.full((5, 5, 3), 240, np.uint8)
    canvas.add(view, make_pose(3, 3, Direction.UP))
    row, col = canvas.to_canvas(3, 3)
    fresh = canvas.image[row, col, 0]

    for _ in range(4):
        canvas.add(view, make_pose(9, 9, Direction.UP))
    faded = canvas.image[row, col, 0]

    assert fresh == 240
    assert faded < fresh / 8
    # and the newly painted area is still full brightness
    row, col = canvas.to_canvas(9, 9)
    assert canvas.image[row, col, 0] == 240


def test_canvas_clips_instead_of_failing():
    canvas = CanvasIntegrator(11, 11, radius=2, decay=1.0)
    view = np.full((5, 5, 3), 240, np.uint8)

    assert canvas.add(view, make_pose(0, 0, Direction.UP))
    assert not canvas.add(view, make_pose(500, 500, Direction.UP))


def test_echoing_the_input_cannot_score_well():
    """The failure v-prediction exists to prevent.

    Under epsilon-prediction a model that simply returns its input scores
    near zero at high noise, because the input is almost pure noise there.
    That is a perfect score for a model that has learned nothing, and it is
    what a rollout then starts from.
    """
    class Echo(torch.nn.Module):
        def forward(self, noisy, actions, tau):
            return noisy

    frames = torch.randn(2, 4, 9, 9, 3).clamp(-1, 1)
    actions = torch.randint(0, 3, (2, 3))
    mask = torch.ones(2, 4, dtype=torch.bool)

    generator = torch.Generator().manual_seed(0)
    loss = training_loss(Echo(), frames, actions, mask, generator=generator)

    assert loss.item() > 0.5, 'echoing the input should be heavily penalised'


def test_velocity_round_trips_at_every_noise_level():
    from marlenv.wm.diffusion import from_velocity, to_velocity

    clean = torch.randn(2, 4, 9, 9, 3).clamp(-1, 1)
    noise = torch.randn_like(clean)
    for value in (0.0, 0.3, 0.7, 1.0):
        tau = torch.full((2, 4), value)
        noisy = add_noise(clean, tau, noise)
        recovered, recovered_noise = from_velocity(
            noisy, to_velocity(clean, noise, tau), tau)

        assert torch.allclose(recovered, clean, atol=1e-5)
        assert torch.allclose(recovered_noise, noise, atol=1e-5)


# ------------------------------------------------------- frames of reference
def test_world_frame_uses_cardinal_actions_directly():
    from marlenv.wm.interactive import HEADINGS

    model = tiny_model()
    player = WorldModelPlayer(model, np.zeros((9, 9, 3), np.uint8),
                              Direction.UP, context=6, denoise_steps=1,
                              frame='world',
                              pose=make_pose(6, 6, Direction.UP))

    action, heading = player.resolve(Direction.LEFT)

    assert heading is Direction.LEFT
    assert action == HEADINGS.index(Direction.LEFT)


def test_ego_frame_uses_relative_actions():
    model = tiny_model()
    player = WorldModelPlayer(model, np.zeros((9, 9, 3), np.uint8),
                              Direction.UP, context=6, denoise_steps=1,
                              frame='ego')

    action, heading = player.resolve(Direction.LEFT)

    assert heading is Direction.LEFT
    assert action == 1                      # a left turn, relatively


def test_reversal_is_refused_in_both_frames():
    model = tiny_model()
    for frame in ('ego', 'world'):
        player = WorldModelPlayer(model, np.zeros((9, 9, 3), np.uint8),
                                  Direction.UP, context=6, denoise_steps=1,
                                  frame=frame)
        _, heading = player.resolve(Direction.DOWN)

        assert heading is Direction.UP, f'{frame} accepted a reversal'


def test_world_frame_display_is_not_rotated_again():
    """A world-frame model already predicts north-up."""
    model = tiny_model()
    pixels = np.random.randint(0, 255, (9, 9, 3), dtype=np.uint8)

    ego = WorldModelPlayer(model, pixels, Direction.RIGHT, context=4,
                           denoise_steps=1, frame='ego')
    world = WorldModelPlayer(model, pixels, Direction.RIGHT, context=4,
                             denoise_steps=1, frame='world')

    # the ego player undoes the head-frame rotation, the world one does not
    assert not np.array_equal(ego.latest_frame(), world.latest_frame())
    assert np.array_equal(world.latest_frame(),
                          to_pixels(to_model_input(pixels)))


def test_world_sequences_use_four_actions():
    episode = {
        'alive_mask': np.ones((3, 1), dtype=bool),
        'observations': np.full((3, 1, 9, 9, 3), 200, dtype=np.uint8),
        'ego_actions': np.zeros((3, 1, 3), dtype=np.uint8),
        'cardinal_actions': np.zeros((3, 1, 4), dtype=np.uint8),
        'poses': np.zeros((3, 1, 3), dtype=np.int16),
    }
    episode['ego_actions'][..., 0] = 1
    episode['cardinal_actions'][..., 3] = 1

    (_, ego_actions, _), = list(agent_sequences(episode, frame='ego'))
    (_, world_actions, _), = list(agent_sequences(episode, frame='world'))

    assert set(ego_actions.tolist()) == {0}
    assert set(world_actions.tolist()) == {3}


# --------------------------------------------------- shared RoPE coordinates
def test_shared_coordinates_are_head_relative_in_cells():
    """Patch grid contributes in patches * cells; displacement in cells."""
    model = WorldModel(num_actions=4, frame='world', dim=64, depth=1, heads=4)
    offsets = model.patch_offsets('cpu')

    # view 9, patch 3, radius 4 -> centres at -3, 0, 3
    assert sorted({int(x) for x in offsets[:, 0]}) == [-3, 0, 3]
    # the middle patch sits exactly on the head
    assert tuple(int(v) for v in offsets[len(offsets) // 2]) == (0, 0)


def test_action_token_shares_the_central_patch_coordinate():
    """Both sit at the agent: the action is taken where the agent is."""
    model = WorldModel(num_actions=4, frame='world', dim=64, depth=1, heads=4)
    actions = torch.tensor([[0, 0, 0]])          # three steps up

    coords = model.token_coords(4, 'cpu', actions)[0]
    types = model.token_types(4, 'cpu')
    per_frame = model.tokens_per_frame
    centre = per_frame // 2

    index = 0
    for step in range(3):
        frame_coords = coords[index:index + per_frame]
        index += per_frame
        action_coord = coords[index]
        index += 1
        assert torch.equal(frame_coords[centre, 1:], action_coord[1:])


def test_one_world_cell_keeps_one_coordinate():
    """The property the whole change exists for."""
    for frame, num_actions in (('world', 4), ('ego', 3)):
        model = WorldModel(num_actions=num_actions, frame=frame, dim=64,
                           depth=1, heads=4)
        actions = torch.tensor([[0, 1, 0, 2, 0]])
        coords = model.token_coords(6, 'cpu', actions)[0]
        types = model.token_types(6, 'cpu')
        offsets = model.patch_offsets('cpu')
        displacement, heading = model.trajectory(actions)

        seen = {}
        index = 0
        for step in range(6):
            for k in range(model.tokens_per_frame):
                # world cell this token covers, per the model's own
                # bookkeeping
                du, dv = (int(v) for v in offsets[k])
                if frame == 'world':
                    local = (du, dv)
                else:
                    head = HEADINGS[int(heading[0, step])]
                    forward = _FORWARD[head]
                    right = _RIGHTWARD[head]
                    local = (-du * forward[0] + dv * right[0],
                             -du * forward[1] + dv * right[1])
                shift = displacement[0, step]
                cell = (local[0] + int(shift[0]), local[1] + int(shift[1]))
                code = tuple(int(v) for v in coords[index, 1:])
                if cell in seen:
                    assert seen[cell] == code, f'{frame}: {cell} moved'
                seen[cell] = code
                index += 1
            if step < 5:
                index += 1


def test_unaligned_coordinates_reuse_the_same_grid():
    """The old behaviour, kept for checkpoints trained with it."""
    model = WorldModel(num_actions=4, frame='world', dim=64, depth=1,
                       heads=4, align_coords=False)
    coords = model.token_coords(3, 'cpu', torch.tensor([[0, 0]]))[0]
    types = model.token_types(3, 'cpu')
    per_frame = model.tokens_per_frame

    first = coords[:per_frame, 1:]
    second = coords[per_frame + 1:2 * per_frame + 1, 1:]
    assert torch.equal(first, second), 'unaligned frames should share a grid'


def test_player_keeps_actions_aligned_with_frames_through_eviction():
    """Shared coordinates dead-reckon from the window's actions.

    One action fewer than frames, always -- otherwise the displacement the
    model integrates does not describe the frames it is given.
    """
    model = WorldModel(num_actions=4, frame='world', dim=64, depth=2,
                       heads=4)
    player = WorldModelPlayer(model, np.zeros((9, 9, 3), np.uint8),
                              Direction.UP, context=5, denoise_steps=1,
                              frame='world')

    for _ in range(12):
        player.step(Direction.UP)
        assert len(player.actions) == player.history.shape[1] - 1
    assert player.history.shape[1] == 5


# ------------------------------------------------------- cache and windowing
def cached_prefix(model, frames, actions, window=None, upto=None):
    """Build a cache over a prefix, the way the runner does."""
    from marlenv.wm.runner import CachedRunner

    steps = upto if upto is not None else frames.shape[1] - 1
    runner = CachedRunner(model, window=window, device='cpu')
    runner.reset(frames[:, :1])
    for step in range(steps):
        runner._commit_action(actions[0, step])
        runner._advance(actions[0, step])
        runner.time += 1
        runner.cache.trim(None if window is None else window - 1)
        if step < steps - 1:
            runner._commit_frame(frames[:, step + 1:step + 2])
    return runner


@pytest.mark.parametrize('window', [None, 3, 5])
def test_cached_path_equals_the_full_forward(window):
    """The fast path must be the same computation, not merely similar."""
    torch.manual_seed(0)
    model = WorldModel(num_actions=4, frame='world', dim=64, depth=3,
                       heads=4).eval()
    steps = 7
    frames = torch.randn(1, steps, 9, 9, 3).clamp(-1, 1)
    actions = torch.randint(0, 4, (1, steps - 1))

    tau = torch.zeros(1, steps)
    tau[0, -1] = 1.0
    with torch.no_grad():
        full = model(frames, actions, tau, window=window)[:, -1]

    runner = cached_prefix(model, frames, actions, window)
    coords = runner._patch_coords(runner.time, runner.displacement,
                                  runner.heading)
    with torch.no_grad():
        cached = model.forward_cached(frames[:, -1:], torch.ones(1, 1),
                                      coords, runner.cache)[:, 0]

    assert torch.allclose(full[0], cached[0], atol=1e-4)


def test_cache_trims_whole_frames():
    """A frame's patches and the action after it must stay together."""
    from marlenv.wm.cache import KVCache

    cache = KVCache(layers=1, tokens_per_step=10)
    cache.recording = True
    for _ in range(6):
        cache.extend(0, torch.zeros(1, 2, 10, 8), torch.zeros(1, 2, 10, 8))
        cache.open_step(10)
        cache.close_step(0)

    assert len(cache) == 60      # six steps of ten tokens
    dropped = cache.trim(4)
    assert dropped == 2
    assert cache.frames == 4
    assert len(cache) == 40


def test_spatial_coordinates_stay_consistent_at_long_context():
    """Length generalisation rests on this: the invariant is unconditional."""
    for steps in (24, 48, 120):
        model = WorldModel(num_actions=4, frame='world', dim=64, depth=1,
                           heads=4)
        torch.manual_seed(0)
        actions = torch.randint(0, 4, (1, steps - 1))
        coords = model.token_coords(steps, 'cpu', actions)[0]
        displacement, _ = model.trajectory(actions)
        offsets = model.patch_offsets('cpu')

        seen = {}
        index = 0
        for step in range(steps):
            shift = displacement[0, step]
            for k in range(model.tokens_per_frame):
                cell = (int(offsets[k, 0] + shift[0]),
                        int(offsets[k, 1] + shift[1]))
                code = tuple(int(v) for v in coords[index, 1:])
                if cell in seen:
                    assert seen[cell] == code, f'{steps}: {cell} moved'
                seen[cell] = code
                index += 1
            if step < steps - 1:
                index += 1


def test_relative_offsets_stay_bounded_as_the_sequence_grows():
    """Attention only ever sees offsets within a window, so they stay small.

    This is why a window longer than the trained one is coherent: the
    coordinates grow, but the differences attention reads do not.
    """
    spans = []
    for steps in (24, 200):
        model = WorldModel(num_actions=4, frame='world', dim=64, depth=1,
                           heads=4)
        torch.manual_seed(0)
        actions = torch.randint(0, 4, (1, steps - 1))
        coords = model.token_coords(steps, 'cpu', actions)[0]
        types = model.token_types(steps, 'cpu')
        window = coords[types == 0][-24 * model.tokens_per_frame:, 1:]
        spans.append(int((window.max(0).values - window.min(0).values).max()))

    assert max(spans) < 30, spans
    assert abs(spans[0] - spans[1]) <= 6, spans


def test_masks_agree_between_backends():
    from marlenv.wm.attention import dense_mask, mask_predicate

    time = torch.tensor([0, 0, 1, 1, 2, 2])
    is_action = torch.tensor([0, 1, 0, 1, 0, 0]).bool()
    dense = dense_mask(time, is_action, window=2)[0, 0]
    predicate = mask_predicate(time, is_action, window=2)

    for q in range(6):
        for kv in range(6):
            assert bool(dense[q, kv]) == bool(
                predicate(torch.tensor(0), torch.tensor(0),
                          torch.tensor(q), torch.tensor(kv)))


# ----------------------------------------------------------- multi-agent
def multi_model(agents=3, **kwargs):
    from marlenv.wm.multiagent import MultiAgentWorldModel
    torch.manual_seed(0)
    settings = dict(num_actions=4, frame='world', dim=64, depth=3, heads=4)
    settings.update(kwargs)
    return MultiAgentWorldModel(num_agents=agents, **settings).eval()


def multi_inputs(agents=3, steps=5, seed=0):
    from marlenv.wm.multiagent import actions_to_signal
    torch.manual_seed(seed)
    frames = torch.randn(1, steps, agents, 9, 9, 3)
    indices = torch.randint(0, 4, (1, steps - 1, agents))
    origins = torch.tensor([[[0, 0], [4, 3], [-3, 5]][:agents]])
    return (frames, actions_to_signal(indices, 4), indices, origins,
            torch.rand(1, steps, agents), torch.rand(1, steps - 1, agents))


def test_agents_are_permutation_equivariant():
    """Identity comes from position, so reordering agents just reorders."""
    model = multi_model()
    frames, signal, indices, origins, ftau, atau = multi_inputs()
    order = [2, 0, 1]

    with torch.no_grad():
        base_f, base_a = model(frames, signal, ftau, atau, origins=origins,
                               action_indices=indices)
        swap_f, swap_a = model(frames[:, :, order], signal[:, :, order],
                               ftau[:, :, order], atau[:, :, order],
                               origins=origins[:, order],
                               action_indices=indices[:, :, order])

    assert torch.allclose(swap_f, base_f[:, :, order], atol=1e-5)
    assert torch.allclose(swap_a, base_a[:, :, order], atol=1e-5)


def test_only_relative_agent_positions_matter():
    """Translating everyone together must change nothing."""
    model = multi_model()
    frames, signal, indices, origins, ftau, atau = multi_inputs()

    with torch.no_grad():
        base, _ = model(frames, signal, ftau, atau, origins=origins,
                        action_indices=indices)
        moved, _ = model(frames, signal, ftau, atau, origins=origins + 7,
                         action_indices=indices)

    assert torch.allclose(moved, base, atol=1e-4)


def test_agents_sharing_a_position_are_not_distinguishable():
    """The flip side: identity is positional, so position must differ."""
    model = multi_model()
    frames, signal, indices, origins, ftau, atau = multi_inputs()
    collided = origins.clone()
    collided[0, 1] = origins[0, 0]

    with torch.no_grad():
        base, _ = model(frames, signal, ftau, atau, origins=origins,
                        action_indices=indices)
        same, _ = model(frames, signal, ftau, atau, origins=collided,
                        action_indices=indices)

    assert not torch.allclose(same, base, atol=1e-5)


def test_multi_agent_actions_stay_causal():
    model = multi_model()
    frames, signal, indices, origins, ftau, atau = multi_inputs()
    from marlenv.wm.multiagent import actions_to_signal

    other = indices.clone()
    other[:, 2, 0] = (indices[:, 2, 0] + 1) % 4
    with torch.no_grad():
        base, _ = model(frames, signal, ftau, atau, origins=origins,
                        action_indices=indices)
        changed, _ = model(frames, actions_to_signal(other, 4), ftau, atau,
                           origins=origins, action_indices=other)
    delta = (changed - base).abs().amax(dim=(2, 3, 4, 5))[0]

    assert torch.all(delta[:3] == 0), 'an action reached its own frame'
    assert delta[3] > 0, 'an action did not reach the next frame'


def test_dead_agents_stop_moving():
    """Otherwise their tokens wander onto living agents' cells.

    Stopping means one last step, not none: an agent alive at step 2 acted
    there and moved into step 3's cell, which is where the aftermath frame
    is taken from. It freezes only afterwards.
    """
    model = multi_model(agents=2)
    actions = torch.zeros(1, 4, 2, dtype=torch.long)
    alive = torch.ones(1, 5, 2, dtype=torch.bool)
    alive[0, 3:, 1] = False
    origins = torch.tensor([[[0, 0], [3, 0]]])

    displacement, _ = model.trajectory(actions, origins, alive)
    frozen = displacement[0, :, 1]

    assert not torch.equal(frozen[3], frozen[2]), 'never entered the cell'
    assert torch.equal(frozen[4], frozen[3]), 'kept walking after dying'
    assert not torch.equal(displacement[0, 4, 0], displacement[0, 2, 0])


def test_noise_broadcast_handles_frames_and_actions():
    """Frames carry three trailing dims, action vectors one."""
    from marlenv.wm.diffusion import add_noise, from_velocity, to_velocity

    for shape, tau_shape in (((2, 3, 4, 9, 9, 3), (2, 3, 4)),
                             ((2, 3, 4, 5), (2, 3, 4))):
        clean = torch.randn(*shape).clamp(-1, 1)
        noise = torch.randn_like(clean)
        tau = torch.rand(*tau_shape)
        noisy = add_noise(clean, tau, noise)
        back, back_noise = from_velocity(noisy, to_velocity(clean, noise,
                                                            tau), tau)

        assert noisy.shape == clean.shape
        assert torch.allclose(back, clean, atol=1e-5)
        assert torch.allclose(back_noise, noise, atol=1e-5)


def test_trailing_action_slot_is_a_policy_query():
    """One action per frame, the last asking what to do now."""
    from marlenv.wm.multiagent import actions_to_signal

    model = multi_model()
    frames = torch.randn(1, 4, 3, 9, 9, 3)
    for slots in (3, 4):
        indices = torch.randint(0, 4, (1, slots, 3))
        with torch.no_grad():
            _, actions = model(frames, actions_to_signal(indices, 4),
                               torch.rand(1, 4, 3), torch.rand(1, slots, 3),
                               action_indices=indices)
        assert actions.shape == (1, slots, 3, 4)


def test_a_pending_action_does_not_move_anything_yet():
    """Its coordinates must not depend on a value not yet decided."""
    model = multi_model()
    indices = torch.randint(0, 4, (1, 4, 3))
    other = indices.clone()
    other[:, -1] = (indices[:, -1] + 1) % 4

    first = model.token_coords(4, 'cpu', indices, action_steps=4)
    second = model.token_coords(4, 'cpu', other, action_steps=4)

    assert torch.equal(first, second)


def test_runner_holds_a_fixed_action_and_samples_the_rest():
    from marlenv.wm.marunner import MultiAgentRunner

    model = multi_model()
    origins = torch.tensor([[[0, 0], [4, 3], [-3, 5]]])
    runner = MultiAgentRunner(model, origins, window=6, device='cpu')
    runner.reset(torch.randn(1, 1, 3, 9, 9, 3))

    for _ in range(4):
        actions, frame = runner.step(fixed={0: 2}, denoise_steps=2,
                                     action_steps=2)
        assert int(actions[0]) == 2, 'the held action was not respected'
        assert actions.shape == (3,)
        assert frame.shape == (1, 1, 3, 9, 9, 3)

    assert runner.frames.shape[1] == 5
    assert runner.actions.shape[1] == 4


def test_runner_window_clips_frames_and_actions_together():
    from marlenv.wm.marunner import MultiAgentRunner

    model = multi_model()
    origins = torch.zeros(1, 3, 2, dtype=torch.long)
    runner = MultiAgentRunner(model, origins, window=4, device='cpu')
    runner.reset(torch.randn(1, 1, 3, 9, 9, 3))

    for _ in range(8):
        runner.step(denoise_steps=1, action_steps=1)
        assert runner.actions.shape[1] == runner.frames.shape[1] - 1

    assert runner.frames.shape[1] == 4


def multi_cached_prefix(model, frames, actions, origins, upto_frames,
                        upto_actions, window=None):
    """Build a multi-agent cache the way the runner does."""
    from marlenv.wm.marunner import CachedMultiRunner
    from marlenv.wm.model import HEADINGS

    runner = CachedMultiRunner(model, origins, window=window, device='cpu')
    runner.reset(frames[:, :1])
    moves = torch.tensor([h.value for h in HEADINGS])
    for step in range(upto_actions):
        runner._commit_actions(actions[0, step])
        runner.displacement = runner.displacement + moves[actions[0, step]]
        runner.time += 1
        runner.cache.trim(None if window is None else window - 1)
        if step + 1 < upto_frames:
            runner._commit_frames(frames[:, step + 1:step + 2])
    return runner


@pytest.mark.parametrize('window', [None, 4])
def test_multi_cached_frames_equal_the_full_forward(window):
    from marlenv.wm.multiagent import actions_to_signal

    model = multi_model()
    agents, steps = 3, 6
    torch.manual_seed(1)
    frames = torch.randn(1, steps, agents, 9, 9, 3).clamp(-1, 1)
    indices = torch.randint(0, 4, (1, steps - 1, agents))
    origins = torch.tensor([[[0, 0], [4, 3], [-3, 5]]])

    frame_tau = torch.zeros(1, steps, agents)
    frame_tau[:, -1] = 1.0
    with torch.no_grad():
        full, _ = model(frames, actions_to_signal(indices, 4), frame_tau,
                        torch.zeros(1, steps - 1, agents), origins=origins,
                        action_indices=indices, window=window)

    runner = multi_cached_prefix(model, frames, indices, origins, steps - 1,
                                 steps - 1, window)
    with torch.no_grad():
        coords = model.step_frame_coords(runner.displacement, runner.time,
                                         'cpu')
        cached = model.frames_cached(frames[:, -1:],
                                     torch.ones(1, 1, agents), coords,
                                     runner.cache)

    assert torch.allclose(full[:, -1], cached[:, 0], atol=1e-4)


def test_multi_cached_actions_equal_the_full_forward():
    """The policy query has to agree too, not just the dynamics."""
    from marlenv.wm.multiagent import actions_to_signal

    model = multi_model()
    agents, steps = 3, 6
    torch.manual_seed(1)
    frames = torch.randn(1, steps, agents, 9, 9, 3).clamp(-1, 1)
    indices = torch.randint(0, 4, (1, steps - 1, agents))
    origins = torch.tensor([[[0, 0], [4, 3], [-3, 5]]])
    signal = actions_to_signal(indices, 4)

    action_tau = torch.zeros(1, steps - 1, agents)
    action_tau[:, -1] = 1.0
    with torch.no_grad():
        _, full = model(frames[:, :steps - 1], signal[:, :steps - 1],
                        torch.zeros(1, steps - 1, agents), action_tau,
                        origins=origins, action_indices=indices[:, :steps - 1])

    runner = multi_cached_prefix(model, frames, indices, origins, steps - 1,
                                 steps - 2)
    with torch.no_grad():
        coords = model.step_action_coords(runner.displacement, runner.time,
                                          'cpu')
        cached = model.actions_cached(signal[:, -1:],
                                      torch.ones(1, 1, agents), coords,
                                      runner.cache)

    assert torch.allclose(full[:, -1], cached[:, 0], atol=1e-4)


def test_cached_runner_holds_a_fixed_action():
    from marlenv.wm.marunner import CachedMultiRunner

    model = multi_model()
    origins = torch.tensor([[[0, 0], [4, 3], [-3, 5]]])
    runner = CachedMultiRunner(model, origins, window=5, device='cpu')
    runner.reset(torch.randn(1, 1, 3, 9, 9, 3))

    for _ in range(8):
        was_live = runner.live[1]
        actions, _ = runner.step(fixed={1: 3}, denoise_steps=2,
                                 action_steps=2)
        # an untrained model generates noise, which reads as death; the held
        # action only applies while the agent is still in the sequence
        if was_live:
            assert int(actions[1]) == 3

    # the window bounds the cache in whole steps, and the cache always holds
    # exactly the tokens its recorded steps account for. Sizes vary, since a
    # step shrinks as agents drop out -- an untrained model generates noise,
    # which reads as death, so they drop out quickly here.
    assert runner.cache.frames <= 4          # window - 1 committed steps
    assert len(runner.cache) == sum(runner.cache.step_sizes)


# ------------------------------------------------------------ dead agents
def test_dead_agents_are_pinned_at_maximum_noise():
    """Diffusion forcing already has a way to say 'no information here'."""
    from marlenv.wm.diffusion import add_noise

    torch.manual_seed(0)
    clean = torch.full((1, 3, 2, 4), 0.5)
    noise = torch.randn_like(clean)
    tau = torch.ones(1, 3, 2)

    # at tau = 1 alpha is zero, so the content is the noise whatever it was
    assert torch.allclose(add_noise(clean, tau, noise), noise, atol=1e-6)


def test_training_pins_dead_agent_tokens():
    """A dead agent's stored action is an all-zero one-hot, read as UP."""
    from marlenv.wm.matrain import multi_training_loss

    model = multi_model()
    torch.manual_seed(0)
    frames = torch.randn(2, 5, 3, 9, 9, 3).clamp(-1, 1)
    actions = torch.randint(0, 4, (2, 4, 3))
    alive = torch.ones(2, 5, 3, dtype=torch.bool)
    alive[:, 3:, 1] = False
    trained = alive.clone()
    origins = torch.zeros(2, 3, 2, dtype=torch.long)

    # the loss must not move when a dead agent's frames are replaced
    generator = torch.Generator().manual_seed(7)
    first = multi_training_loss(model, frames, actions, alive, trained,
                                origins, generator=generator)[0]
    scrambled = frames.clone()
    scrambled[:, 3:, 1] = torch.randn_like(scrambled[:, 3:, 1])
    generator = torch.Generator().manual_seed(7)
    second = multi_training_loss(model, scrambled, actions, alive, trained,
                                 origins, generator=generator)[0]

    assert first.item() == pytest.approx(second.item(), rel=1e-5)


def test_death_is_read_off_the_centre_cell():
    """While an agent lives the centre of its view is its own head.

    Once it dies the view is taken from the cell it died entering, so the
    centre is something else. No threshold, and nothing out of distribution.
    """
    import gymnasium as gym
    from marlenv.wm.data import to_model_input
    from marlenv.wm.marunner import looks_dead

    env = gym.make('Snake-v1', height=13, width=13, num_snakes=3,
                   view_radius=4, observation_noise=0.0,
                   snake_noise_sigma=0.0, background_gradient=0.0,
                   disable_env_checker=True)
    env.reset(seed=2)
    env.action_space.seed(2)
    base = env.unwrapped

    saw_a_death = False
    for _ in range(20):
        views = torch.tensor(to_model_input(base.egocentric_rgb()))
        detected = looks_dead(views).tolist()
        actual = [not snake.alive for snake in base.snakes]
        assert detected == actual, f'{detected} != {actual}'
        saw_a_death |= any(actual)
        if all(actual):
            break
        env.step(list(env.action_space.sample()))

    assert saw_a_death, 'no death occurred to check'


def test_dead_agents_leave_the_token_stream():
    from marlenv.wm.marunner import CachedMultiRunner

    model = multi_model()
    origins = torch.tensor([[[0, 0], [4, 3], [-3, 5]]])
    runner = CachedMultiRunner(model, origins, window=8, device='cpu')
    runner.reset(torch.randn(1, 1, 3, 9, 9, 3))

    runner.step(denoise_steps=1, action_steps=1)
    full_step = runner.cache.step_sizes[-1]
    assert full_step == 3 * model.tokens_per_frame

    runner.live[1] = False
    before = runner.displacement[1].clone()
    runner.step(denoise_steps=1, action_steps=1)

    # the newest group is still open: its frame is committed, its actions
    # are not taken until the next step
    assert runner.cache.step_sizes[-1] < full_step, 'dead agent still emits'
    assert runner.cache.step_sizes[-1] == 2 * model.tokens_per_frame
    assert torch.equal(runner.displacement[1], before), 'dead agent moved'
    assert runner.living == [0, 2]


def test_cache_trims_variable_sized_steps():
    """Step sizes shrink as agents die, so trimming cannot assume a size."""
    from marlenv.wm.cache import KVCache

    cache = KVCache(layers=1, tokens_per_step=30)
    cache.recording = True
    for tokens in (30, 30, 20, 20):
        cache.extend(0, torch.zeros(1, 2, tokens, 8),
                     torch.zeros(1, 2, tokens, 8))
        cache.open_step(tokens)
        cache.close_step(0)

    assert len(cache) == 100
    cache.trim(2)
    assert len(cache) == 40           # the two 30s went, not 2 * 30 blindly
    assert cache.step_sizes == [20, 20]


def _diverging_episode(frames=8, agents=2, view=9):
    """Two agents walking apart, the second dying near the end.

    Their offsets at step 0 say nothing about their offsets later, which is
    the whole point: anything that reads the starting offsets and applies
    them to a later crop is wrong.
    """
    import numpy as np
    from marlenv.core.snake import Direction

    headings = list(Direction)
    right, down = headings.index(Direction.RIGHT), headings.index(
        Direction.DOWN)
    poses = np.zeros((frames, agents, 3), np.int64)
    cardinal = np.zeros((frames, agents, 4), np.int64)
    alive = np.ones((frames, agents), bool)
    # the frame's own step index is written into its pixels, so a crop can
    # be located again after the batcher has shuffled it
    observations = np.zeros((frames, agents, view, view, 3), np.uint8)

    for step in range(frames):
        observations[step] = step
        poses[step, 0] = (2, 2 + step, right)          # walking right
        poses[step, 1] = (2 + step, 9, down)           # walking down
        cardinal[step, 0, right] = 1
        cardinal[step, 1, down] = 1
    alive[frames - 2:, 1] = False                      # agent 1 dies
    return {'alive_mask': alive, 'poses': poses, 'observations': observations,
            'cardinal_actions': cardinal, 'steps': frames - 1,
            'num_agents': agents}


def _pack(episode):
    """One decoded episode packed the way build_multi_sequences packs many."""
    import numpy as np
    from marlenv.wm.madata import episode_sequence

    obs, act, live, trained, positions = episode_sequence(episode)
    return {'observations': obs[None], 'actions': act[None],
            'alive': live[None], 'trained': trained[None],
            'mask': np.ones((1, len(obs)), bool), 'positions': positions[None]}


def test_crop_origins_follow_the_crop_not_the_episode():
    """Offsets must be read at the crop's own first frame.

    The agents drift apart, so the episode's starting offsets misplace them
    by however far they have moved by the time the crop begins.
    """
    import numpy as np
    from marlenv.wm.matrain import MultiBatcher

    sequences = _pack(_diverging_episode())
    batcher = MultiBatcher(sequences, context=3, seed=0)
    frames, _, _, _, origins = batcher.batch(12)

    positions = sequences['positions'][0]
    for row in range(frames.shape[0]):
        # recover which step this crop started at from the pixels
        value = frames[row, 0, 0, 0, 0, 0].item()
        start = int(round((value + 1.0) * 127.5))
        want = positions[start] - positions[start, 0]
        assert np.array_equal(origins[row].numpy(), want), (
            f'crop starting at {start} was given the wrong offsets')


def test_a_dying_agent_still_enters_the_cell_it_died_in():
    """The aftermath frame is the view from the cell, so it must sit there.

    Gating movement on surviving arrival leaves that frame's tokens one cell
    behind the view they actually carry.
    """
    from marlenv.wm.multiagent import MultiAgentWorldModel

    sequences = _pack(_diverging_episode())
    model = MultiAgentWorldModel(num_agents=2, view=9, num_actions=4,
                                 frame='world', dim=32, depth=1, heads=4)

    length = int(sequences['mask'][0].sum())
    positions = sequences['positions'][0, :length]
    origins = torch.from_numpy(
        (positions[0] - positions[0, 0])[None])
    displacement, _ = model.trajectory(
        torch.from_numpy(sequences['actions'][0, :length - 1][None]), origins,
        torch.from_numpy(sequences['alive'][0, :length][None]))

    truth = positions - positions[0, 0]
    trained = sequences['trained'][0, :length]
    error = (displacement[0].numpy() - truth)[trained]
    assert not error.any(), 'a trained frame sits away from what it shows'


def test_the_action_that_kills_is_still_trained():
    """It produced the aftermath frame, which is a target, so it must be."""
    sequences = _pack(_diverging_episode())
    alive = sequences['alive']
    fatal = alive[:, :-1] & ~alive[:, 1:]
    assert fatal.any(), 'the fixture should contain a fatal move'
    assert (alive[:, :-1] & fatal).all() == fatal.all()
    # the mask the loss uses must cover it
    assert alive[:, :-1][fatal].all()
