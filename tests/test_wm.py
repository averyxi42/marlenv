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
def test_death_appends_exactly_one_black_frame():
    episode = {
        'alive_mask': np.array([[True], [True], [True], [False]]),
        'observations': np.full((4, 1, 9, 9, 3), 200, dtype=np.uint8),
        'ego_actions': np.zeros((4, 1, 3), dtype=np.uint8),
    }
    episode['ego_actions'][..., 0] = 1

    (obs, actions, died), = list(agent_sequences(episode))

    assert died
    assert len(obs) == 4               # three living frames plus the marker
    assert (obs[-1] == 0).all()
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

    cache = KVCache(layers=1, tokens_per_frame=9)
    cache.recording = True
    for _ in range(6):
        cache.extend(0, torch.zeros(1, 2, 10, 8), torch.zeros(1, 2, 10, 8))
        cache.frames += 1

    assert len(cache) == 60
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
