"""GIF palette conversion must preserve rare class colours and captions."""
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def test_tiler_preserves_every_snake_colour_even_if_absent_in_later_frames():
    from marlenv.core.palette import palette_entries
    from marlenv.wm.showreel import BACKDROP, INK, MUTED, quantize_gif
    path = Path(__file__).resolve().parents[1] / 'diagram_gen/tile_rollouts.py'
    spec = importlib.util.spec_from_file_location('tiler', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, colours = palette_entries(6)
    colours = np.vstack([colours, colours * 0.3,
                         [BACKDROP, INK, MUTED, module.INK, module.MUTED]])
    original = colours.astype(np.uint8)[None]
    encoded = quantize_gif(Image.fromarray(original), module.rollout_palette())
    assert np.array_equal(np.asarray(encoded.convert('RGB')), original)


def test_source_gif_preserves_class_colours_and_caption_ink(tmp_path):
    from marlenv.core.palette import palette_entries
    from marlenv.wm.showreel import save, BACKDROP, INK, MUTED
    _, colours = palette_entries(6)
    colours = np.vstack([colours, colours * 0.3, [BACKDROP, INK, MUTED]])
    first = colours.astype(np.uint8)[None]
    # The second frame has no snakes; the first must keep all their colours.
    second = np.full_like(first, BACKDROP)
    path = tmp_path / 'colours.gif'
    save([first, second], str(path))
    with Image.open(path) as gif:
        assert np.array_equal(np.asarray(gif.convert('RGB')), first)
        gif.seek(1)
        assert np.array_equal(np.asarray(gif.convert('RGB')), second)
