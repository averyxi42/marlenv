"""Driving the world model by hand, in human coordinates.

The model lives in the agent's own frame: every observation is rotated so
the snake faces up, and actions are relative (straight, left, right). That
is the right representation to learn from and the wrong one to play in --
"left" means something different every time you turn.

This module converts both ends. Views are rotated back so world-up is up,
which keeps the board still while the snake turns, and cardinal key presses
are converted to relative actions against the current heading. Reversing
into your own neck is not a legal move, so that key is ignored the way a
snake game normally does.
"""
import numpy as np
import torch

from marlenv.core.snake import Direction
from marlenv.grading.compare import unrotate_view
from marlenv.grading.poses import LEFT_TURN, RIGHT_TURN, step_pose, turn
from marlenv.wm.data import to_model_input, to_pixels
from marlenv.wm.diffusion import denoise_next

STRAIGHT, LEFT, RIGHT = 0, 1, 2


def cardinal_to_ego(heading, cardinal):
    """The relative action that turns ``heading`` towards ``cardinal``.

    Returns ``None`` for a reversal, which a snake cannot do.
    """
    if cardinal == heading:
        return STRAIGHT
    if cardinal == LEFT_TURN[heading]:
        return LEFT
    if cardinal == RIGHT_TURN[heading]:
        return RIGHT
    return None


def world_up(view, heading):
    """Undo the head-frame rotation, so north on the board is up."""
    return unrotate_view(np.asarray(view), heading)


class WorldModelPlayer:
    """Rolls the world model forward under keyboard control.

    Holds the head-frame history the model needs and the heading needed to
    display it, and keeps the history clipped to the context the model was
    trained on -- a longer one would be out of distribution and slower for
    nothing.
    """

    def __init__(self, model, observation, heading, context=24,
                 denoise_steps=8, device='cpu', seed=0, pose=None):
        self.model = model.to(device).eval()
        self.device = device
        self.context = context
        self.denoise_steps = denoise_steps
        self.generator = torch.Generator(device=device).manual_seed(seed)

        frame = to_model_input(np.asarray(observation)[None, None])
        self.history = torch.from_numpy(frame).to(device)
        self.headings = [heading]
        self.actions = []
        self.steps = 0
        # dead reckoned for compositing the canvas only; the model is never
        # told where it is
        self.pose = pose
        self.poses = [pose] if pose is not None else []

    @property
    def heading(self):
        return self.headings[-1]

    def latest_frame(self):
        """The most recent frame as world-up uint8 pixels."""
        pixels = to_pixels(self.history[0, -1].cpu().numpy())
        return world_up(pixels, self.heading)

    def step(self, cardinal):
        """Advance one frame. Returns the action actually taken."""
        ego = cardinal_to_ego(self.heading, cardinal) if cardinal else None
        if ego is None:
            ego = STRAIGHT           # ignore reversals, keep going

        actions = torch.tensor([self.actions + [ego]], dtype=torch.long,
                               device=self.device)
        frame = denoise_next(self.model, self.history, actions,
                             denoise_steps=self.denoise_steps,
                             generator=self.generator)

        self.history = torch.cat([self.history, frame], dim=1)
        self.actions.append(ego)
        self.headings.append(turn(self.heading, ego))
        if self.pose is not None:
            self.pose = step_pose(self.pose, ego)
            self.poses.append(self.pose)
        self.steps += 1

        if self.history.shape[1] > self.context:
            drop = self.history.shape[1] - self.context
            self.history = self.history[:, drop:]
            self.actions = self.actions[drop:]
        return ego

    def bootstrap(self, observations, headings, actions):
        """Seed the history with real frames before handing over."""
        for observation, heading, action in zip(observations, headings,
                                                actions):
            frame = to_model_input(np.asarray(observation)[None, None])
            self.history = torch.cat(
                [self.history, torch.from_numpy(frame).to(self.device)], 1)
            self.actions.append(int(action))
            self.headings.append(heading)
            if self.pose is not None:
                self.pose = step_pose(self.pose, int(action))
                self.poses.append(self.pose)
        if self.history.shape[1] > self.context:
            drop = self.history.shape[1] - self.context
            self.history = self.history[:, drop:]
            self.actions = self.actions[drop:]


class SimulatorReference:
    """Steps the real env alongside, for side-by-side comparison."""

    def __init__(self, env, agent=0, others=None):
        self.env = env
        self.agent = agent
        self.others = others
        self.alive = True

    @property
    def snake(self):
        return self.env.unwrapped.snakes[self.agent]

    def observation(self):
        views = self.env.unwrapped.egocentric_rgb()
        return world_up(views[self.agent], self.snake.direction)

    def step(self, ego_action):
        if not self.alive:
            return False
        base = self.env.unwrapped
        actions = ([0] * base.num_snakes if self.others is None
                   else list(self.others(self.env)))
        actions[self.agent] = int(ego_action)
        _, _, terminated, truncated, _ = self.env.step(actions)
        self.alive = base.snakes[self.agent].alive
        return not (all(terminated) or all(truncated))
