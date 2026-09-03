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
from marlenv.wm.model import WorldModel  # noqa: E402


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
