"""Render the palette reference sheet to showcase/palette.png.

    python examples/analysis/palette_sheet.py

Shows every cell class as a swatch, and the background gradient both as a
strip and as the four headings a snake can read off it. Regenerate this
whenever the palette or the gradient defaults change.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from marlenv.core.observation import egocentric_crop, heading_gradient
from marlenv.core.palette import (EMPTY_RGB, cell_color, palette_entries,
                                  safety_report)
from marlenv.core.snake import Cell, Direction

BACKDROP = (22, 22, 26)
INK = (232, 232, 238)
MUTED = (150, 150, 160)


def label(draw, xy, text, fill=INK):
    draw.text(xy, text, fill=fill)


def swatch_block(num_snakes=6, cell=48, pad=8, label_w=96):
    """Every class colour, statics on the first row, then one row per snake."""
    rows = 1 + num_snakes
    width = pad + 3 * (cell + pad) + label_w
    height = pad + 18 + rows * (cell + pad) + 16
    img = Image.new('RGB', (width, height), BACKDROP)
    draw = ImageDraw.Draw(img)
    label(draw, (pad, 2), 'cell classes')

    y = pad + 18
    for i, kind in enumerate((Cell.EMPTY, Cell.WALL, Cell.FRUIT)):
        x = pad + i * (cell + pad)
        draw.rectangle([x, y, x + cell, y + cell],
                       fill=tuple(cell_color(kind.value).astype(int)))
        label(draw, (x + 2, y + 2), kind.name.lower(), MUTED)
    label(draw, (pad + 3 * (cell + pad), y + cell // 2 - 4), 'static', MUTED)

    for snake_id in range(num_snakes):
        y += cell + pad
        for i, kind in enumerate((Cell.HEAD, Cell.BODY, Cell.TAIL)):
            x = pad + i * (cell + pad)
            colour = tuple(cell_color(kind.value, snake_id).astype(int))
            draw.rectangle([x, y, x + cell, y + cell], fill=colour)
            if snake_id == 0:
                label(draw, (x + 2, y + 2), kind.name.lower(), (20, 20, 20))
        label(draw, (pad + 3 * (cell + pad), y + cell // 2 - 4),
              f'snake {snake_id}', MUTED)
    return img


def gradient_block(amplitude, period, cells=13, scale=16, pad=8, radius=4):
    """The gradient over a few periods, and what each heading sees of it."""
    field = heading_gradient((cells, cells), pad=0, period=period,
                             amplitude=amplitude)
    strip = np.clip(np.array(EMPTY_RGB, dtype=np.float32) + field,
                    0, 255).astype(np.uint8)
    big = np.repeat(np.repeat(strip, scale, 0), scale, 1)

    views = []
    for direction in Direction:
        view = egocentric_crop(strip, (cells // 2, cells // 2), direction,
                               radius=radius, pad=0)
        views.append(np.repeat(np.repeat(view, scale, 0), scale, 1))

    view_w, view_h = views[0].shape[1], views[0].shape[0]
    width = pad + big.shape[1] + 2 * pad + 4 * (view_w + pad)
    height = pad + 18 + max(big.shape[0], view_h) + 18
    img = Image.new('RGB', (width, height), BACKDROP)
    draw = ImageDraw.Draw(img)
    label(draw, (pad, 2),
          f'heading gradient, empty cells only  '
          f'(amplitude {amplitude:g}, period {period})')

    img.paste(Image.fromarray(big), (pad, pad + 18))
    label(draw, (pad, pad + 18 + big.shape[0] + 3), 'world', MUTED)

    x = pad + big.shape[1] + 2 * pad
    top = pad + 18 + (big.shape[0] - view_h) // 2
    for direction, view in zip(Direction, views):
        img.paste(Image.fromarray(view), (x, top))
        label(draw, (x, top + view_h + 3),
              f'facing {direction.name.lower()}', MUTED)
        x += view_w + pad
    return img


def stack(images, pad=12, footer_lines=0):
    width = max(i.width for i in images) + 2 * pad
    height = (sum(i.height for i in images) + pad * (len(images) + 1)
              + 16 * footer_lines)
    sheet = Image.new('RGB', (width, height), BACKDROP)
    y = pad
    for image in images:
        sheet.paste(image, (pad, y))
        y += image.height + pad
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='showcase/palette.png')
    parser.add_argument('--num-snakes', type=int, default=6)
    parser.add_argument('--gradient', type=float, default=16.0)
    parser.add_argument('--gradient-period', type=int, default=6)
    parser.add_argument('--sigma-bg', type=float, default=2.0)
    parser.add_argument('--sigma-snake', type=float, default=8.0)
    args = parser.parse_args()

    blocks = [swatch_block(args.num_snakes),
              gradient_block(args.gradient, args.gradient_period)]
    sheet = stack(blocks, footer_lines=3)
    draw = ImageDraw.Draw(sheet)
    y = sheet.height - 46
    draw.text((12, y), f'decoding margin at sigma_bg={args.sigma_bg:g}, '
                       f'sigma_snake={args.sigma_snake:g}:', fill=INK)
    for mode in (True, False):
        slack, why = safety_report(args.num_snakes, args.sigma_bg,
                                   args.sigma_snake,
                                   gradient_amplitude=args.gradient,
                                   strict=mode)
        y += 14
        name = 'strict ' if mode else 'relaxed'
        draw.text((24, y), f'{name} {slack:+6.1f}   {why}', fill=MUTED)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    sheet.save(args.out)
    print(f'wrote {args.out} ({sheet.width}x{sheet.height})')


if __name__ == '__main__':
    main()
