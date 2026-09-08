"""Retirement must not change the ownership of a surviving agent's action."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from marlenv.flex_wm.model import FlexWorldModel
from marlenv.flex_wm.runner import CachedFlexRunner, FlexRunner
from marlenv.wm.diffusion import alpha_sigma
from marlenv.wm.model import HEADINGS
from marlenv.wm.multiagent import actions_to_signal


@pytest.fixture(params=[FlexRunner, CachedFlexRunner])
def runner(request):
    model = FlexWorldModel(view=9, num_actions=4, dim=32, depth=2,
                           heads=4, schedule='AG').eval()
    runner = request.param(model, [91, 7, 42], [[0, 0], [2, 0], [0, 2]],
                           window=4, death_patience=1)
    runner.reset(torch.zeros(1, 1, 3, 9, 9, 3))
    return runner


def oracle(runner, monkeypatch):
    """Return identity-specific actions, and inspect what the model receives."""
    calls = []

    def forward(pairs, frames, actions, ft, at, *args, **kwargs):
        calls.append((pairs.agent.clone(), pairs.time.clone(), actions.clone(),
                      at.clone(), kwargs.get('record', False)))
        desired = torch.where(pairs.agent == 91, 1,
                              torch.where(pairs.agent == 7, 2, 3))
        clean = actions_to_signal(desired, 4)
        alpha, sigma = alpha_sigma(at)
        velocity = (alpha[..., None] * actions - clean) / sigma[..., None].clamp(min=1e-6)
        return torch.zeros_like(frames), velocity

    monkeypatch.setattr(runner.model, 'forward', forward)
    monkeypatch.setattr(runner.model, 'forward_cached', forward)
    return calls


def test_middle_retirement_preserves_sampled_action_ownership(runner, monkeypatch):
    oracle(runner, monkeypatch)
    runner.live[1] = False
    full = runner.sample_actions(steps=2)
    assert full.tolist() == [1, 0, 3]
    assert runner.pairs.actions.tolist() == [[1, 0, 3]]
    assert runner.actions_known.tolist() == [[True, False, True]]
    assert runner.pairs.acted.tolist() == [[True, False, True]]
    assert runner.pairs.valid.all() and runner.pairs.trained.all()
    if isinstance(runner, CachedFlexRunner):
        assert runner.frontier.actions.tolist() == [[1, 0, 3]]


def test_fixed_actions_are_clean_conditioning_during_sampling(runner, monkeypatch):
    calls = oracle(runner, monkeypatch)
    full = runner.sample_actions(fixed={1: 3}, steps=3)
    assert full.tolist() == [1, 3, 3]
    for ids, _, signal, tau, _ in calls:
        slot = ids == 7
        assert (tau[slot] == 0).all()
        assert torch.equal(signal[slot], actions_to_signal(torch.tensor([3]), 4))
    count = len(calls)
    assert torch.equal(runner.sample_actions(), full)
    assert len(calls) == count, 'decided actions were sampled again'


def test_real_death_keeps_aftermath_once_and_has_no_outgoing_action(runner):
    frames = torch.full((1, 1, 3, 9, 9, 3), 0.75)
    actions = torch.tensor([1, 2, 3])
    runner.observe(actions, frames, [True, False, True])
    assert runner.living == [0, 2]
    assert torch.equal(runner.frames, frames)
    assert runner.pairs.agent.tolist() == [[91, 7, 42, 91, 7, 42]]
    assert runner.pairs.acted.tolist() == [[True, True, True, True, False, True]]
    moves = torch.tensor([h.value for h in HEADINGS])
    assert torch.equal(runner.position, runner.start + moves[actions])
    runner.observe(torch.tensor([3, 0, 1]), -frames, [True, False, True])
    assert runner.pairs.agent.tolist() == [[91, 7, 42, 91, 7, 42, 91, 42]]
    assert runner.pairs.actions[0, 3:6].tolist() == [3, 0, 1]
    assert not runner.actions_known[0, 4]
    assert torch.equal(runner.frames[0, 0, 1], frames[0, 0, 1])
    # The death observation and unavailable action survive trimming intact.
    for _ in range(2):
        runner.observe(actions, frames, [True, False, True])
    dead = runner.pairs.agent == 7
    assert dead.sum() == 1
    assert not runner.pairs.acted[dead].any()
    assert not runner.actions_known[dead].any()


def test_external_actions_drive_both_history_and_positions(runner):
    runner.live[1] = False
    actions = torch.tensor([1, 0, 3])
    runner.generate_frames(actions, steps=1, generator=torch.Generator().manual_seed(4))
    assert runner.pairs.actions[0, :3].tolist() == [1, 0, 3]
    assert runner.actions_known[0, :3].tolist() == [True, False, True]
    assert runner.pairs.agent[0, -2:].tolist() == [91, 42]
    moves = torch.tensor([h.value for h in HEADINGS])
    assert torch.equal(runner.position[[0, 2]], runner.start[[0, 2]] + moves[actions[[0, 2]]])
    assert torch.equal(runner.position[1], runner.start[1])


def test_dead_action_is_never_presented_as_known(runner, monkeypatch):
    calls = oracle(runner, monkeypatch)
    runner.observe(torch.tensor([1, 2, 3]), runner.frames.clone(), [True, False, True])
    calls.clear()
    runner.sample_actions(steps=2)
    runner.generate_frames(torch.tensor([1, 0, 3]), steps=2)
    seen = 0
    for ids, times, signal, tau, _ in calls:
        dead = (ids == 7) & (times == 1)
        if dead.any():
            seen += 1
            assert (tau[dead] == 1).all()
            assert not torch.equal(signal[dead], actions_to_signal(torch.tensor([0]), 4))
    assert seen


def test_all_retired_is_safe_without_fake_actions_or_new_pairs(runner, monkeypatch):
    runner.observe(torch.tensor([1, 2, 3]), runner.frames.clone(), [False] * 3)
    calls = oracle(runner, monkeypatch)
    count = runner.pairs.pairs
    positions = runner.position.clone()
    for _ in range(2):
        actions, frames = runner.step(denoise_steps=1, action_steps=1)
        assert actions.tolist() == [0, 0, 0]
        assert frames.shape[1] == 0
    assert not calls, "an empty population triggered another forward pass"
    assert runner.pairs.pairs == count
    assert torch.equal(runner.position, positions)
    assert not runner.actions_known[0, -3:].any()


def test_display_and_canvas_keep_newly_retired_view():
    from marlenv.wm.canvas import make_pose
    from marlenv.wm.showreel import Showreel
    from marlenv.core.snake import Direction

    class Runner:
        alive = torch.ones(1, 1, 3, dtype=torch.bool)
        frames = torch.full((1, 1, 3, 9, 9, 3), -1.0)

        def step(self, **kwargs):
            self.frames[0, 0, 1] = 1.0
            self.alive[0, 0, 1] = False
            return torch.tensor([0, 1, 2]), self.frames[:, 0]

    reel = Showreel.__new__(Showreel)
    reel.runner = Runner()
    reel.poses = [make_pose(5, 5, Direction.UP) for _ in range(3)]
    reel.steps, reel.snap = 0, False
    reel.last_seen = [np.zeros((9, 9, 3), np.uint8) for _ in range(3)]
    pasted = []
    reel.canvas = SimpleNamespace(fade=lambda: None,
                                  paste=lambda view, pose: pasted.append(view.copy()))
    reel.step()
    assert (reel.last_seen[1] == 255).all()
    assert (reel.views_for_display()[1] == 255).all()
    assert len(pasted) == 3 and (pasted[1] == 255).all()
    reel.step()
    assert len(pasted) == 5, 'old death observation was repainted'


def test_generated_death_has_only_one_final_pair(runner, monkeypatch):
    import marlenv.flex_wm.runner as module
    monkeypatch.setattr(module, 'looks_dead',
                        lambda frames: torch.tensor([False, True, False]))
    runner.generate_frames(torch.tensor([1, 2, 3]), steps=1)
    runner.retire(runner.frames[:, 0], [0, 1, 2])
    assert runner.living == [0, 2]
    final = runner.frames[0, 0, 1].clone()
    for _ in range(2):
        runner.sample_actions(steps=1)
        runner.generate_frames(torch.tensor([3, 0, 1]), steps=1)
    dead = runner.pairs.agent == 7
    assert runner.pairs.time[dead].tolist() == [0, 1]
    assert runner.pairs.acted[dead].tolist() == [True, False]
    assert torch.equal(runner.frames[0, 0, 1], final)


def test_trimming_preserves_patch_visibility(runner):
    runner.pairs.visible = torch.ones(1, 3, runner.model.tokens_per_frame,
                                      dtype=torch.bool)
    runner.pairs.visible[0, 1, 0] = False
    runner.window = 1
    # Compact directly tests the pair operation shared by rollout trimming.
    from marlenv.flex_wm.pairs import compact
    kept = compact(runner.pairs, torch.tensor([[False, True, True]]))
    assert torch.equal(kept.visible, runner.pairs.visible[:, 1:])
