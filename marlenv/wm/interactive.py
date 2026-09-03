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

A world-frame model needs less of this: it already predicts north-up views
and takes the four cardinal actions, so only the reversal rule applies. Both
kinds are played through the same class, which keeps the two comparable --
what you see on screen is north-up either way.
"""
import numpy as np
import torch

from marlenv.core.snake import Direction
from marlenv.grading.compare import unrotate_view
from marlenv.grading.poses import LEFT_TURN, Pose, RIGHT_TURN, turn
from marlenv.wm.data import to_model_input, to_pixels
from marlenv.wm.diffusion import denoise_next

STRAIGHT, LEFT, RIGHT = 0, 1, 2
HEADINGS = list(Direction)
OPPOSITE = {heading: LEFT_TURN[LEFT_TURN[heading]] for heading in Direction}


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
                 denoise_steps=8, device='cpu', seed=0, pose=None,
                 frame='ego', use_cache=True):
        """``use_cache`` runs the sliding-window KV cache path.

        It is equivalent to recomputing the whole window every step, pinned
        by a test to float precision, and avoids re-encoding the history once
        per denoising step -- which is most of the work.
        """
        if frame not in ('ego', 'world'):
            raise ValueError("frame must be 'ego' or 'world'")
        self.frame = frame
        self.model = model.to(device).eval()
        self.device = device
        self.context = context
        self.denoise_steps = denoise_steps
        self.generator = torch.Generator(device=device).manual_seed(seed)

        first = to_model_input(np.asarray(observation)[None, None])
        self.history = torch.from_numpy(first).to(device)

        self.runner = None
        if use_cache:
            from marlenv.wm.runner import CachedRunner
            self.runner = CachedRunner(model, window=context, device=device)
            # heading 0 matches the convention WorldModel.trajectory uses when
            # it dead-reckons a window; only coordinate differences matter
            self.runner.reset(self.history)
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
        """The most recent frame as north-up uint8 pixels.

        A world-frame model already predicts north-up, so only the ego frame
        needs undoing. Either way the caller gets the same thing.
        """
        pixels = to_pixels(self.history[0, -1].cpu().numpy())
        if self.frame == 'world':
            return pixels
        return world_up(pixels, self.heading)

    def resolve(self, cardinal):
        """The action index and resulting heading for a key press.

        Reversals are refused in both frames -- a snake cannot turn back into
        its own neck -- and an absent or refused press keeps it going.
        """
        heading = self.heading
        if cardinal is None or cardinal is OPPOSITE[heading]:
            cardinal = heading
        if self.frame == 'world':
            return HEADINGS.index(cardinal), cardinal
        action = cardinal_to_ego(heading, cardinal)
        return action, turn(heading, action)

    def step(self, cardinal):
        """Advance one frame. Returns the action index actually taken."""
        action, heading = self.resolve(cardinal)

        if self.runner is not None:
            frame = self.runner.step(action,
                                     denoise_steps=self.denoise_steps,
                                     generator=self.generator)
        else:
            actions = torch.tensor([self.actions + [action]],
                                   dtype=torch.long, device=self.device)
            frame = denoise_next(self.model, self.history, actions,
                                 denoise_steps=self.denoise_steps,
                                 generator=self.generator)

        self.history = torch.cat([self.history, frame], dim=1)
        self.actions.append(action)
        self.headings.append(heading)
        if self.pose is not None:
            self.pose = Pose(self.pose.row + heading.value[0],
                             self.pose.col + heading.value[1], heading)
            self.poses.append(self.pose)
        self.steps += 1

        self._trim()
        return action

    def _trim(self):
        """Slide the window, keeping actions aligned with the frames.

        With shared coordinates the model dead-reckons displacement from the
        actions it is given, so the window must always carry exactly one
        fewer action than frames. Re-basing the origin on the window start is
        harmless: RoPE reads differences, and shifting every coordinate by a
        constant leaves those unchanged.
        """
        extra = self.history.shape[1] - self.context
        if extra > 0:
            self.history = self.history[:, extra:]
            self.actions = self.actions[extra:]
        assert len(self.actions) == self.history.shape[1] - 1, (
            'actions and frames fell out of step')

    def bootstrap(self, observations, headings, actions, poses=None):
        """Seed the history with real frames before handing over.

        Observations must already be in this player's frame, and actions in
        its encoding -- relative for ego, cardinal index for world.
        """
        poses = poses or [None] * len(observations)
        for observation, heading, action, pose in zip(observations, headings,
                                                      actions, poses):
            pixels = to_model_input(np.asarray(observation)[None, None])
            tensor = torch.from_numpy(pixels).to(self.device)
            if self.runner is not None:
                self.runner._commit_action(int(action))
                self.runner._advance(int(action))
                self.runner.time += 1
                self.runner.cache.trim(None if self.context is None
                                       else self.context - 1)
                self.runner._commit_frame(tensor)
            self.history = torch.cat([self.history, tensor], 1)
            self.actions.append(int(action))
            self.headings.append(heading)
            if self.pose is not None:
                self.pose = pose or Pose(self.pose.row + heading.value[0],
                                         self.pose.col + heading.value[1],
                                         heading)
                self.poses.append(self.pose)
        self._trim()


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
