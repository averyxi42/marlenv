"""A picture of what one agent's account of an episode actually contains.

The egocentric training data is easy to describe and hard to believe
without seeing it, because the interesting part is what is *missing*. This
lays a short stretch of an episode out left to right, one column per step,
so the three levels can be read against each other:

    world       the true board, which no agent ever sees, with the
                observer's view outlined on it
    observer    what the observer recorded, complete, with the action it
                took drawn as a cardinal arrow between frames
    others      each snake the observer managed to recover, appearing when
                its head enters view and vanishing when it leaves, with
                the patches the observer could not see dimmed out

A row is one real snake, which is ground truth the observer does not have.
The bubbles drawn on that row are what it *does* have: a fresh identity for
every visit, because nothing survives an absence. Two bubbles on one row is
the same snake counted twice, and that is correct -- the observer has no way
to know it is the same snake, so the data must not say so.

Read down a column and you see how much of the truth survives into the
data. Read across a row and you see an identity begin, last a few steps and
end -- the transients popping in and out, which is the thing the model has
to learn to deduce.

Segments are not picked by hand. Every window of the scanned episodes is
scored for how much appearing and disappearing it contains, and the best
one is drawn, because a stretch where nothing enters or leaves makes a
truthful but useless picture.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import load_from_disk

from marlenv.core import render
from marlenv.core.palette import (BODY_WHEEL, EMPTY_RGB, FRUIT_RGB,
                                  WALL_RGB)
from marlenv.core.render import draw_frame
from marlenv.core.snake import Cell, Snake
from marlenv.data import decode_episode
from marlenv.data.state import snake_bodies
from marlenv.flex_wm.egocentric import (egocentric_pairs, head_in_view,
                                        patch_offsets, visible_runs)

BACKDROP = (16, 16, 20)
PANEL = (26, 26, 32)
INK = (236, 236, 242)
MUTED = (132, 132, 144)
DIM = (70, 70, 82)
ACCENT = (250, 208, 92)
# one per identity, never per snake: two visits by the same snake must not
# share a colour, or the picture claims a link the observer does not have
IDENTITY_INK = [(120, 210, 255), (255, 150, 170), (170, 255, 160),
                (220, 170, 255), (255, 196, 120), (140, 240, 230),
                (200, 200, 120), (240, 160, 230)]
ARROWS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

# both glyphs are drawn from the same grid at the same scale, so the arrow
# and the stick under it read as one piece of pixel art rather than as a
# drawing next to a sprite
UNIT = 2
ARROW_BITS = ('.........',
              '.....#...',
              '.....##..',
              '########.',
              '#########',
              '########.',
              '.....##..',
              '.....#...',
              '.........')
STICK_TALL = 14
# the arrow above points right; turn it clockwise to face anywhere else
TURNS = {0: 3, 1: 0, 2: 1, 3: 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--data-root', default='marlenv/demodata')
    p.add_argument('--component', default='expert_nogradient')
    p.add_argument('--scan', type=int, default=60,
                   help='episodes to search for a good stretch')
    p.add_argument('--steps', type=int, default=8,
                   help='columns in the timeline')
    p.add_argument('--cell', type=int, default=15,
                   help='pixels per cell in a view')
    p.add_argument('--out', default='diagrams/ego_timeline.png')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------- choosing
def runs_for(episode, ego, radius):
    """``{other: [(start, stop), ...]}`` of the visits the observer sees."""
    alive, poses = episode['alive_mask'], episode['poses']
    frames, agents = alive.shape
    out = {}
    for other in range(agents):
        if other == ego:
            continue
        seen = np.array([
            bool(alive[t, ego] and alive[t, other]
                 and head_in_view(poses[t, ego, :2], poses[t, other, :2],
                                  radius))
            for t in range(frames)])
        runs = visible_runs(seen, shortest=2)
        if runs:
            out[other] = runs
    return out


def score_window(runs, start, stop, alive_ego):
    """How much appearing and disappearing a window contains."""
    if not alive_ego[start:stop].all():
        return -1
    entries = exits = present = 0
    for stretches in runs.values():
        for begin, end in stretches:
            if begin >= stop or end <= start:
                continue
            present += 1
            # an edge strictly inside the window is one the reader can see
            entries += int(start < begin < stop)
            exits += int(start < end < stop)
    if not present:
        return -1
    # popping in and out is the point; more identities is a tiebreak
    return 4 * entries + 4 * exits + present


def best_segment(dataset, args, radius):
    """The most legible stretch in the scanned episodes."""
    best = None
    for index in range(min(args.scan, len(dataset))):
        episode = decode_episode(dataset[index])
        alive = episode['alive_mask']
        for ego in range(alive.shape[1]):
            if alive[:, ego].sum() < args.steps:
                continue
            runs = runs_for(episode, ego, radius)
            for start in range(alive.shape[0] - args.steps):
                score = score_window(runs, start, start + args.steps,
                                     alive[:, ego])
                if best is None or score > best[0]:
                    best = (score, index, ego, start, episode)
    if best is None or best[0] <= 0:
        raise SystemExit('no stretch with a visible transient was found')
    return best


# ------------------------------------------------------------- drawing
def owners(episode, pairs):
    """``{identity: real agent}``, by matching where each head was.

    Ground truth for the diagram only. The reconstruction deliberately does
    not carry it, which is the thing being illustrated.
    """
    poses = episode['poses']
    found = {}
    for identity in np.unique(pairs['agent']):
        rows = np.flatnonzero(pairs['agent'] == identity)
        rows = rows[np.argsort(pairs['time'][rows])]
        times = pairs['time'][rows]
        for candidate in range(poses.shape[1]):
            if np.array_equal(poses[times, candidate, :2],
                              pairs['position'][rows]):
                found[int(identity)] = candidate
                break
    return found


def match_palette():
    """Draw the board in the hues a view uses, keeping the pixel art.

    The game's renderer carries its own retro palette, so the same snake
    comes out green in a view and yellow on the board -- exactly the
    confusion this diagram cannot afford. Swapping the colours underneath
    keeps the bevelled walls, eyes and rounded joints, and makes a snake
    one colour wherever it appears.
    """
    render.SNAKE_COLORS = list(BODY_WHEEL)
    render.WALL = WALL_RGB
    render.WALL_LIGHT = render._shade(WALL_RGB, 1.45)
    render.WALL_DARK = render._shade(WALL_RGB, 0.63)
    render.BACKGROUND = (EMPTY_RGB, render._shade(EMPTY_RGB, 1.3))
    render.FRUIT = FRUIT_RGB


def board_image(episode, step, cell):
    """The true board at ``step``, drawn the way the game draws itself."""
    content = episode['content'][step]
    grid = np.zeros(content.shape, dtype=np.int32)
    grid[content == 1] = Cell.WALL.value
    grid[content == 2] = Cell.FRUIT.value
    snakes = []
    for index, body in snake_bodies(content, episode['body_index'][step]
                                    ).items():
        if len(body) < 2:
            continue
        for coord in body:
            grid[coord] = Cell.BODY.value + 10 * index
        grid[body[0]] = Cell.HEAD.value + 10 * index
        grid[body[-1]] = Cell.TAIL.value + 10 * index
        snake = Snake(index, body)
        snake.alive = True
        snakes.append(snake)
    return Image.fromarray(draw_frame(grid, snakes, cell), 'RGB')


def view_image(view, visible, cell):
    """A stored view, upscaled, with what was not seen dimmed away."""
    pixels = view.astype(np.float32)
    if visible is not None:
        grid = int(round(len(visible) ** 0.5))
        patch = view.shape[0] // grid
        flags = visible.reshape(grid, grid)
        flags = np.repeat(np.repeat(flags, patch, 0), patch, 1)
        pixels = np.where(flags[..., None], pixels, pixels * 0.14 + 12)
    image = Image.fromarray(pixels.clip(0, 255).astype(np.uint8), 'RGB')
    image = image.resize((view.shape[1] * cell, view.shape[0] * cell),
                         Image.NEAREST)
    if visible is not None and not visible.all():
        # a faint lattice, so a dimmed patch reads as withheld and not dark
        draw = ImageDraw.Draw(image)
        grid = int(round(len(visible) ** 0.5))
        span = view.shape[0] // grid * cell
        for i in range(grid):
            for j in range(grid):
                if visible.reshape(grid, grid)[i, j]:
                    continue
                draw.rectangle([j * span, i * span, (j + 1) * span - 1,
                                (i + 1) * span - 1], outline=DIM)
    return image


def turned(bits, quarters):
    """The same little bitmap, rotated clockwise."""
    grid = [list(row) for row in bits]
    for _ in range(quarters % 4):
        grid = [list(row) for row in zip(*grid[::-1])]
    return [''.join(row) for row in grid]


def stamp(draw, bits, left, top, colour):
    """Paint a bitmap at ``UNIT`` pixels to the cell."""
    for r, row in enumerate(bits):
        for c, mark in enumerate(row):
            if mark == '#':
                draw.rectangle([left + c * UNIT, top + r * UNIT,
                                left + (c + 1) * UNIT - 1,
                                top + (r + 1) * UNIT - 1], fill=colour)


def joystick(draw, cx, top, colour):
    """A small pixel-art stick, the customary mark for an action."""
    draw.rectangle([cx - 3, top, cx + 2, top + 4], fill=colour)
    draw.rectangle([cx - 1, top + 4, cx, top + 8], fill=colour)
    draw.rectangle([cx - 5, top + 8, cx + 4, top + 10], fill=colour)
    draw.rectangle([cx - 7, top + 10, cx + 6, top + 13], fill=colour)


def action_mark(draw, cx, top, action, colour, spacing):
    """A cardinal arrow with a stick beneath it, as one glyph.

    The arrow says which way, in white, because direction belongs to nobody;
    the stick says whose decision it was, and carries the identity's colour.
    """
    side = len(ARROW_BITS) * UNIT
    stamp(draw, turned(ARROW_BITS, TURNS[int(action)]), cx - side // 2, top,
          INK)
    joystick(draw, cx, top + side + spacing, colour)


def font_for(size):
    for name in ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(episode, ego, start, pairs, args, radius):
    """Lay the whole thing out."""
    steps = args.steps
    view = episode['observations'].shape[2]
    tile = view * args.cell
    board_cell = max(tile // episode['content'].shape[1], 6)
    board = episode['content'].shape[1] * board_cell
    column = max(tile, board)
    gap, margin, label = 34, 20, 176
    head, row_gap = 108, 44
    window = range(start, start + steps)

    inside = (pairs['time'] >= start) & (pairs['time'] < start + steps)
    held = owners(episode, pairs)
    # one row per real snake; the identities on it are what the observer has
    lanes = {}
    for identity in np.unique(pairs['agent'][inside]):
        if identity == 0:
            continue
        lanes.setdefault(held.get(int(identity), -1), []).append(int(identity))
    order_of = {snake: index for index, snake
                in enumerate(sorted(lanes))}
    ink_of = {0: ACCENT}
    for number, identity in enumerate(
            sorted(i for lane in lanes.values() for i in lane)):
        ink_of[identity] = IDENTITY_INK[number % len(IDENTITY_INK)]

    rows = len(lanes) + 2
    width = margin * 2 + label + steps * column + (steps - 1) * gap
    height = (margin * 2 + head + rows * tile + (rows - 1) * row_gap
              + (board - tile))
    image = Image.new('RGB', (width, height), BACKDROP)
    draw = ImageDraw.Draw(image)
    big, small, tiny = font_for(20), font_for(15), font_for(12)

    draw.text((margin, margin - 4),
              "One agent's account of " + str(steps) + " steps",
              fill=INK, font=big)
    draw.text((margin, margin + 24),
              'a row is one real snake; a bubble is one identity the '
              'observer assigned. two bubbles on a row are the same snake, '
              'which the observer cannot know.', fill=MUTED, font=tiny)
    draw.text((margin, margin + 40),
              'every identity has its own colour; the snake names are our '
              'bookkeeping and reach the model nowhere.', fill=MUTED,
              font=tiny)

    key = width - margin - 268
    for offset, (shade, note) in enumerate((
            ((104, 210, 96), 'seen by the observer'),
            ((34, 34, 40), 'withheld: masked by noise, not dropped'))):
        top = margin + 2 + offset * 22
        draw.rectangle([key, top, key + 15, top + 15], fill=shade,
                       outline=DIM)
        draw.text((key + 23, top + 1), note, fill=MUTED, font=tiny)

    def left(index):
        return margin + label + index * (column + gap)

    def band(order):
        return margin + head + order * (tile + row_gap)

    for index in range(steps):
        draw.text((left(index) + column / 2 - 14, margin + head - 54),
                  f't={start + index}', fill=MUTED, font=small)

    def draw_row(order, identities, name, note):
        draw.text((margin, band(order) + tile / 2 - 16), name, fill=INK,
                  font=small)
        draw.text((margin, band(order) + tile / 2 + 4), note, fill=MUTED,
                  font=tiny)
        for index in range(steps):
            box = (left(index), band(order), left(index) + column,
                   band(order) + tile)
            draw.rectangle(box, outline=PANEL)

        for identity in identities:
            colour = ink_of[identity]
            rows_of = pairs['agent'] == identity
            columns = [index for index in range(steps)
                       if (rows_of & (pairs['time'] == start + index)).any()]
            if not columns:
                continue
            for index in columns:
                row = np.flatnonzero(
                    rows_of & (pairs['time'] == start + index))[0]
                seen = None if identity == 0 else pairs['visible'][row]
                picture = view_image(pairs['observations'][row], seen,
                                     args.cell)
                x = left(index) + (column - picture.width) // 2
                image.paste(picture, (x, band(order)))
                spot = (x, band(order), x + picture.width,
                        band(order) + picture.height)
                if pairs['acted'][row] and index < steps - 1:
                    # the pair reads as one mark, so it is the point between
                    # their centres that sits level with the tile
                    side, spacing = len(ARROW_BITS) * UNIT, 9
                    # the two glyphs are different heights, so the point to
                    # centre on is worked out from where each centre lands
                    middle = (side / 2 + side + spacing
                              + STICK_TALL / 2) / 2
                    action_mark(draw, spot[2] + gap / 2,
                                spot[1] + tile / 2 - middle,
                                int(pairs['actions'][row]), colour, spacing)

            # the bubble: where this identity begins and ends
            first, last = min(columns), max(columns)
            pad = 9
            bubble = [left(first) + (column - tile) // 2 - pad,
                      band(order) - pad,
                      left(last) + (column + tile) // 2 + pad,
                      band(order) + tile + pad]
            draw.rounded_rectangle(bubble, radius=14, outline=colour,
                                   width=2)
            tag = f'identity {identity}'
            if identity == 0:
                tag += '  ·  its own record, nothing withheld'
            if identity != 0:
                fraction = pairs['visible'][rows_of & inside].mean()
                tag += f'  ·  {fraction:.0%} of patches seen'
            tag_x = min(bubble[0] + 8,
                        width - margin - draw.textlength(tag, font=tiny))
            draw.text((tag_x, bubble[1] - 19), tag, fill=colour, font=tiny)

    for snake, identities in sorted(lanes.items()):
        draw_row(order_of[snake], identities, f'snake {chr(65 + snake)}',
                 'recovered, in view')
    draw_row(len(lanes), [0], 'the observer',
             f'snake {chr(65 + ego)}, complete')

    order = rows - 1
    draw.text((margin, band(order) + board / 2 - 10), 'world', fill=INK,
              font=small)
    draw.text((margin, band(order) + board / 2 + 10), 'never observed',
              fill=MUTED, font=tiny)
    for index in range(steps):
        step = start + index
        picture = board_image(episode, step, board_cell)
        x = left(index) + (column - picture.width) // 2
        image.paste(picture, (x, band(order)))
        head_pose = episode['poses'][step, ego, :2]
        draw.rectangle(
            [x + (head_pose[1] - radius) * board_cell,
             band(order) + (head_pose[0] - radius) * board_cell,
             x + (head_pose[1] + radius + 1) * board_cell - 1,
             band(order) + (head_pose[0] + radius + 1) * board_cell - 1],
            outline=ACCENT)
    return image


def main():
    args = parse_args()
    match_palette()
    dataset = load_from_disk(os.path.join(args.data_root, args.component))
    radius = int(dataset[0]['view_radius'])
    score, index, ego, start, episode = best_segment(dataset, args, radius)
    pairs = egocentric_pairs(episode, patch_offsets(
        episode['observations'].shape[2], 3), ego=ego)
    print(f'episode {index}, observer agent {ego}, steps '
          f'{start}-{start + args.steps - 1}  (score {score})')

    image = compose(episode, ego, start, pairs, args, radius)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    image.save(args.out)
    print(f'wrote {args.out}  {image.width}x{image.height}')


if __name__ == '__main__':
    main()
