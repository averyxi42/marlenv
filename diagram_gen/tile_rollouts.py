"""Rollouts side by side, so the comparison is one picture.

Reading gifs in separate windows means holding one in memory while watching
another, which is exactly the comparison a reader is worst at. Tiling them
puts the same step of each next to the others, under captions saying what
differs, and the difference either survives that or it was never there.

The sources are left untouched. This writes a third file.

Rollouts can end at different lengths -- a model whose snakes all die stops
early -- so the shorter side holds its final frame rather than looping back
to the start, which would read as the rollout continuing. A held frame is
greyed and marked, because a bright still picture beside a moving one still
reads as a rollout that happens to be sitting quiet.
"""
import argparse
import os

from PIL import (Image, ImageDraw, ImageEnhance, ImageFont,
                  ImageSequence)

BACKDROP = (16, 16, 20)
INK = (236, 236, 242)
MUTED = (132, 132, 144)
RULE = (52, 52, 62)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--gifs', nargs='+', required=True,
                   help='rollouts to tile, left to right')
    p.add_argument('--labels', nargs='+', default=None,
                   help='one heading per gif')
    p.add_argument('--notes', nargs='+', default=None,
                   help='one subheading per gif; pass an empty string to '
                        'leave one blank')
    p.add_argument('--title', default='')
    p.add_argument('--out', default='diagrams/rollout_tiled.gif')
    p.add_argument('--gap', type=int, default=26)
    p.add_argument('--margin', type=int, default=18)
    return p.parse_args()


def read_gif(path):
    """Every frame as RGB, with the duration each was shown for."""
    frames, durations = [], []
    with Image.open(path) as source:
        for frame in ImageSequence.Iterator(source):
            frames.append(frame.convert('RGB'))
            durations.append(frame.info.get('duration', 100))
    return frames, durations


def spent(image, font):
    """A finished panel: colour drained, dimmed, and said so in words."""
    faded = ImageEnhance.Color(image).enhance(0.18)
    faded = ImageEnhance.Brightness(faded).enhance(0.5)
    draw = ImageDraw.Draw(faded)
    text = 'rollout ended'
    pad, box = 6, draw.textlength(text, font=font)
    left, top = 8, faded.height - 26
    draw.rectangle([left, top, left + box + 2 * pad, top + 19],
                   fill=BACKDROP, outline=RULE)
    draw.rectangle([left + pad, top + 7, left + pad + 5, top + 12],
                   fill=MUTED)
    draw.text((left + pad + 11, top + 3), text, fill=MUTED, font=font)
    return faded


def font_for(size):
    for name in ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    args = parse_args()
    panels = [read_gif(path) for path in args.gifs]
    labels = args.labels or [os.path.basename(p) for p in args.gifs]
    notes = args.notes or [''] * len(panels)
    for name, values in (('--labels', labels), ('--notes', notes)):
        if len(values) != len(panels):
            raise SystemExit(f'{name} needs one entry per gif')

    big, small, tiny = font_for(21), font_for(16), font_for(13)
    head = (34 if args.title else 0) + (48 if any(notes) else 26)
    widths = [frames[0].width for frames, _ in panels]
    tallest = max(frames[0].height for frames, _ in panels)
    width = (args.margin * 2 + args.gap * (len(panels) - 1) + sum(widths))
    height = args.margin * 2 + head + tallest

    canvas = Image.new('RGB', (width, height), BACKDROP)
    sketch = ImageDraw.Draw(canvas)
    if args.title:
        sketch.text((args.margin, args.margin - 2), args.title, fill=INK,
                    font=big)

    top = args.margin + (34 if args.title else 0)
    lefts, cursor = [], args.margin
    for index, panel in enumerate(panels):
        lefts.append(cursor)
        cursor += widths[index] + args.gap
    for x, label, note in zip(lefts, labels, notes):
        sketch.text((x, top), label, fill=INK, font=small)
        if note:
            sketch.text((x, top + 20), note, fill=MUTED, font=tiny)
    body = args.margin + head
    sketch.line([args.margin, body - 8, width - args.margin, body - 8],
                fill=RULE)

    # worked out once: a marked-up still is the same picture every frame
    finished = [spent(frames[-1], tiny) for frames, _ in panels]

    length = max(len(frames) for frames, _ in panels)
    merged, timings = [], []
    for index in range(length):
        frame = canvas.copy()
        for x, (frames, _), done in zip(lefts, panels, finished):
            # a finished rollout holds, rather than looping under a running one
            frame.paste(done if index >= len(frames) else frames[index],
                        (x, body))
        merged.append(frame)
        timings.append(max(ms[min(index, len(ms) - 1)] for _, ms in panels))

    # one palette for the whole gif: left to itself PIL picks a fresh
    # adaptive palette per frame, and the captions -- identical pixels
    # every frame -- come out speckled as the text colour lands on a
    # different entry each time
    shared = merged[len(merged) // 2].quantize(colors=256,
                                               method=Image.MEDIANCUT)
    merged = [frame.quantize(palette=shared, dither=Image.Dither.NONE)
              for frame in merged]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    merged[0].save(args.out, save_all=True, append_images=merged[1:],
                   duration=timings, loop=0, optimize=True)
    counts = ', '.join(str(len(frames)) for frames, _ in panels)
    print(f'wrote {args.out}  {width}x{height}  {len(merged)} frames '
          f'(from {counts})')


if __name__ == '__main__':
    main()
