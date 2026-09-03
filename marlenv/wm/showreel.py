"""Recording a world action model playing itself, laid out for watching.

The play script exists to be steered; this exists to be looked at. Nobody
is at the controls, so every action comes from the model, and the layout is
arranged for a viewer rather than a player: the stitched map large across
the top, where the geometry either holds together or visibly does not, and
each agent's own view along the bottom, which is what the model actually
generates and all it ever sees.

Views are shown north-up, the frame the model works in, so a snake turning
does not spin the picture. Retired viewpoints stay in place, dimmed, rather
than vanishing and reshuffling the row under the reader.
"""
import numpy as np

from marlenv.core.palette import snap_to_palette
from marlenv.core.snake import Direction
from marlenv.grading.compare import PALETTE_SNAKES, unrotate_view
from marlenv.wm.canvas import CanvasIntegrator, make_pose
from marlenv.wm.data import to_model_input, to_pixels

HEADINGS = list(Direction)
BACKDROP = (18, 18, 22)
INK = (232, 232, 238)
MUTED = (128, 128, 138)
REWARD_DICT = {'fruit': 1.0, 'kill': 0.0, 'lose': -5.0, 'win': 0.0,
               'time': 0.01}


def world_views(env):
    """Every agent's view from the simulator, north-up."""
    base = env.unwrapped
    return np.stack([unrotate_view(view, snake.direction) for view, snake
                     in zip(base.egocentric_rgb(), base.snakes)])


class Showreel:
    """Drives a passive rollout and keeps the pictures it makes."""

    def __init__(self, model, env, runner, radius, side, decay=0.94,
                 snap=True):
        self.model = model
        self.env = env
        self.runner = runner
        self.radius = radius
        self.snap = snap
        base = env.unwrapped
        self.canvas = CanvasIntegrator(side, side, radius, decay=decay)
        self.poses = [make_pose(s.head_coord[0], s.head_coord[1], s.direction)
                      for s in base.snakes]
        self.headings = [s.direction for s in base.snakes]
        self.steps = 0
        # a retired viewpoint keeps the last thing it saw, which says more
        # than the placeholder the runner leaves in its slot
        self.last_seen = list(self.observe())
        self.paint(self.observe())

    # ------------------------------------------------------------- driving
    def observe(self):
        return world_views(self.env)

    @property
    def living(self):
        alive = self.runner.alive[0, -1]
        return [i for i in range(len(self.poses)) if bool(alive[i])]

    def dream(self):
        """The views the model just generated, as pixels."""
        return [to_pixels(frame.cpu().numpy())
                for frame in self.runner.frames[0, -1]]

    def paint(self, views):
        """Age the canvas once, then lay every living view onto it."""
        self.canvas.fade()
        for index in self.living:
            self.canvas.paste(views[index], self.poses[index])

    def bootstrap(self, steps, solver):
        """Play real steps in before handing the rollout to the model."""
        import torch

        base = self.env.unwrapped
        for _ in range(steps):
            _, _, terminated, truncated, _ = self.env.step(
                solver.solve(self.env))
            self.headings = [s.direction for s in base.snakes]
            self.poses = [make_pose(s.head_coord[0], s.head_coord[1],
                                    s.direction) for s in base.snakes]
            views = self.observe()
            live = torch.tensor([s.alive for s in base.snakes],
                                dtype=torch.bool, device=self.runner.device)
            actions = torch.tensor([HEADINGS.index(h) for h in self.headings],
                                   dtype=torch.long, device=self.runner.device)
            self.runner.observe(actions, torch.from_numpy(
                to_model_input(views[None, None])).to(self.runner.device),
                live)
            self.remember(views)
            self.paint(views)
            self.steps += 1
            if all(terminated) or all(truncated):
                break

    def step(self, denoise_steps=12, action_steps=4, generator=None):
        """One step of the model driving itself."""
        moving = self.living
        actions, _ = self.runner.step(denoise_steps=denoise_steps,
                                      action_steps=action_steps,
                                      generator=generator)
        chosen = [HEADINGS[int(a)] for a in actions]
        self.headings = chosen
        for index in moving:
            pose, heading = self.poses[index], chosen[index]
            self.poses[index] = make_pose(pose.row + heading.value[0],
                                          pose.col + heading.value[1], heading)
        dreamt = self.dream()
        self.paint(dreamt)
        self.remember(dreamt)
        self.steps += 1
        return actions

    # ------------------------------------------------------------ pictures
    def views_for_display(self):
        living = set(self.living)
        views = [view if index in living else self.last_seen[index]
                 for index, view in enumerate(self.dream())]
        if self.snap:
            views = [snap_to_palette(view, PALETTE_SNAKES) for view in views]
        return views

    def remember(self, views):
        for index in self.living:
            self.last_seen[index] = views[index]


def upscale(image, factor):
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def label(sheet, text, x, y, colour=INK):
    """Draw a short caption, using whatever font PIL has to hand."""
    from PIL import Image, ImageDraw

    picture = Image.fromarray(sheet)
    draw = ImageDraw.Draw(picture)
    draw.text((x, y), text, fill=colour)
    return np.asarray(picture)


def compose(reel, canvas_scale=22, tile_scale=14, gap=10, margin=14):
    """The canvas across the top, each agent's own view along the bottom."""
    canvas = upscale(reel.canvas.image, canvas_scale)
    views = reel.views_for_display()
    living = set(reel.living)

    tiles = []
    for index, view in enumerate(views):
        tile = upscale(view, tile_scale)
        if index not in living:
            tile = (tile * 0.3).astype(np.uint8)
        tiles.append(tile)

    strip_width = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    tile_height = tiles[0].shape[0]
    caption = 18
    width = max(canvas.shape[1], strip_width) + margin * 2
    height = (canvas.shape[0] + tile_height + caption * 2 + gap
              + margin * 2)

    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    sheet[:] = BACKDROP

    x = (width - canvas.shape[1]) // 2
    sheet[margin:margin + canvas.shape[0], x:x + canvas.shape[1]] = canvas

    top = margin + canvas.shape[0] + caption
    x = (width - strip_width) // 2
    spots = []
    for tile in tiles:
        sheet[top:top + tile_height, x:x + tile.shape[1]] = tile
        spots.append(x)
        x += tile.shape[1] + gap

    sheet = label(sheet, f'canvas   step {reel.steps}', margin,
                  margin + canvas.shape[0] + 4)
    for index, spot in enumerate(spots):
        text = f'agent {index}' + ('' if index in living else '  retired')
        sheet = label(sheet, text, spot, top + tile_height + 4,
                      INK if index in living else MUTED)
    return sheet


def save(frames, path, duration=160, hold=1200):
    """Write the recorded frames out as a gif.

    The last frame is held by lengthening it rather than by repeating it:
    the encoder merges runs of identical frames, so copies simply vanish.
    """
    import os

    from PIL import Image

    if not frames:
        raise ValueError('nothing recorded')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    timing = [duration] * len(images)
    timing[-1] = hold
    images[0].save(path, save_all=True, append_images=images[1:],
                   format='GIF', loop=0, duration=timing)
    return path
