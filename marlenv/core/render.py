"""Retro pixel-art renderer for the snake grid.

Everything is drawn procedurally -- there are no sprite assets. Each cell
becomes a ``cell_size`` block, and a snake is drawn as a *path* rather than as
independent cells: every segment is a rounded block bridged towards the
neighbours it actually connects to, so corners read as corners and the body
shows which way it runs. The head carries eyes pointing along its heading and
the tail tapers, which makes direction readable at a glance.

This also sidesteps the frame-merging problem the flat renderer has, where a
snake rotating within its own cells produced a byte-identical image: here the
head, eyes and taper move even when the occupied cells do not.
"""
import numpy as np
from PIL import Image

from marlenv.core.snake import Cell

# a small, deliberately limited retro palette
BACKGROUND = ((18, 20, 28), (23, 26, 35))   # checkerboard shades
WALL = (58, 62, 78)
WALL_LIGHT = (86, 92, 112)
WALL_DARK = (36, 39, 50)
FRUIT = (223, 46, 58)
FRUIT_SHINE = (255, 176, 176)
FRUIT_STEM = (96, 176, 72)
EYE = (16, 16, 20)

# one hue per snake, cycled; body is the base, head a lift, tail a shade
SNAKE_COLORS = [
    (124, 232, 88),
    (255, 198, 62),
    (255, 96, 132),
    (86, 172, 255),
    (196, 132, 255),
    (86, 226, 214),
]


def _shade(color, factor):
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)


def _snake_palette(idx):
    base = SNAKE_COLORS[idx % len(SNAKE_COLORS)]
    cycle = idx // len(SNAKE_COLORS)
    base = _shade(base, 0.75 ** cycle)
    return {
        'body': base,
        'head': _shade(base, 1.18),
        'tail': _shade(base, 0.72),
        'joint': _shade(base, 0.62),
        'edge': _shade(base, 0.38),
    }


def _disc(size, radius):
    """Boolean disc mask of side ``size``, used for fruit."""
    axis = np.arange(size) - (size - 1) / 2.0
    rows, cols = np.meshgrid(axis, axis, indexing='ij')
    return (rows ** 2 + cols ** 2) <= radius ** 2


def _draw_background(image, shape, cell):
    for r in range(shape[0]):
        for c in range(shape[1]):
            image[r * cell:(r + 1) * cell,
                  c * cell:(c + 1) * cell] = BACKGROUND[(r + c) % 2]


def _draw_wall(image, r, c, cell):
    top, left = r * cell, c * cell
    image[top:top + cell, left:left + cell] = WALL
    bevel = max(1, cell // 8)
    image[top:top + bevel, left:left + cell] = WALL_LIGHT
    image[top:top + cell, left:left + bevel] = WALL_LIGHT
    image[top + cell - bevel:top + cell, left:left + cell] = WALL_DARK
    image[top:top + cell, left + cell - bevel:left + cell] = WALL_DARK


def _draw_fruit(image, r, c, cell):
    top, left = r * cell, c * cell
    radius = cell * 0.3            # deliberately smaller than the cell
    mask = _disc(cell, radius)
    block = image[top:top + cell, left:left + cell]
    block[mask] = FRUIT

    shine = max(1, cell // 8)
    off = int(cell * 0.32)
    block[off:off + shine, off:off + shine] = FRUIT_SHINE
    stem = max(1, cell // 10)
    mid = cell // 2
    block[max(0, int(cell * 0.2) - stem):int(cell * 0.2),
          mid:mid + stem] = FRUIT_STEM


def _draw_segment(image, r, c, cell, color, link_color, links, pad,
                  link_pad):
    """A bead, joined to its neighbours by a narrower, darker link.

    Two things separate one segment from the next: the link is thinner than
    the bead, so the background shows through at every joint, and it is
    darker, so the seam stays visible even where two beads nearly touch.
    Together they make the body a countable chain rather than a solid bar,
    without mottling the beads themselves.
    """
    top, left = r * cell, c * cell
    lo, hi = pad, cell - pad
    llo, lhi = link_pad, cell - link_pad
    for dr, dc in links:
        if dr < 0:
            image[top:top + lo, left + llo:left + lhi] = link_color
        elif dr > 0:
            image[top + hi:top + cell, left + llo:left + lhi] = link_color
        elif dc < 0:
            image[top + llo:top + lhi, left:left + lo] = link_color
        elif dc > 0:
            image[top + llo:top + lhi, left + hi:left + cell] = link_color
    image[top + lo:top + hi, left + lo:left + hi] = color


def _draw_eyes(image, r, c, cell, direction):
    """Two pixels set along the heading, so the head points somewhere."""
    top, left = r * cell, c * cell
    size = max(1, cell // 7)
    dr, dc = direction
    centre = (cell - size) / 2.0
    forward = cell * 0.16
    sideways = cell * 0.18
    for sign in (-1, 1):
        row = centre + dr * forward + sign * (-dc) * sideways
        col = centre + dc * forward + sign * dr * sideways
        row = int(np.clip(row, 0, cell - size))
        col = int(np.clip(col, 0, cell - size))
        image[top + row:top + row + size,
              left + col:left + col + size] = EYE


def draw_frame(grid, snakes, cell_size=16):
    """Render one frame as ``(H * cell_size, W * cell_size, 3)`` uint8."""
    height, width = grid.shape
    image = np.zeros((height * cell_size, width * cell_size, 3),
                     dtype=np.uint8)
    _draw_background(image, grid.shape, cell_size)

    cell_kind = grid % 10
    for r in range(height):
        for c in range(width):
            kind = cell_kind[r, c]
            if kind == Cell.WALL.value:
                _draw_wall(image, r, c, cell_size)
            elif kind == Cell.FRUIT.value:
                _draw_fruit(image, r, c, cell_size)

    # beads sit well inside their cell so the background frames every
    # segment; the head is the widest, the tail the narrowest
    head_pad = max(1, round(cell_size * 0.18))
    body_pad = max(1, round(cell_size * 0.22))
    tail_pad = max(body_pad + 1, round(cell_size * 0.34))
    # the link is roughly half the bead's width, so every joint visibly
    # pinches and the segments stay countable
    link_pad = max(body_pad + 1, round(cell_size * 0.38))
    outline = max(1, cell_size // 14)

    for snake in snakes:
        if not snake.alive:
            continue
        palette = _snake_palette(snake.idx)
        coords = [(r, c) for r, c in snake.coords
                  if 0 <= r < height and 0 <= c < width]
        if not coords:
            continue
        last = len(coords) - 1

        def pad_for(position):
            if position == 0:
                return head_pad
            return tail_pad if position == last else body_pad

        def links_for(position, r, c):
            # link only to the neighbours this segment is actually joined to
            out = []
            for other in (position - 1, position + 1):
                if 0 <= other <= last:
                    orow, ocol = coords[other]
                    out.append((orow - r, ocol - c))
            return out

        # pass one lays a dark outline under the whole snake, so segments
        # stay separated from each other and from the background
        for position, (r, c) in enumerate(coords):
            _draw_segment(image, r, c, cell_size, palette['edge'],
                          palette['edge'], links_for(position, r, c),
                          max(0, pad_for(position) - outline),
                          max(0, link_pad - outline))

        for position, (r, c) in enumerate(coords):
            links = links_for(position, r, c)
            if position == 0:
                _draw_segment(image, r, c, cell_size, palette['head'],
                              palette['joint'], links, head_pad, link_pad)
                _draw_eyes(image, r, c, cell_size, snake.direction.value)
            elif position == last:
                _draw_segment(image, r, c, cell_size, palette['tail'],
                              palette['joint'], links, tail_pad, link_pad)
            else:
                _draw_segment(image, r, c, cell_size, palette['body'],
                              palette['joint'], links, body_pad, link_pad)
    return image


def image_from_env(grid, snakes, cell_size=16):
    """The same frame as a PIL image, for the gif buffer."""
    return Image.fromarray(draw_frame(grid, snakes, cell_size), 'RGB')
