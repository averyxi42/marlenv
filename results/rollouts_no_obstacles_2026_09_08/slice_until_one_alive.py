"""Losslessly truncate the paired GIF at the first single-survivor frame."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
diagrams = root / 'diagrams'
source = diagrams / 'rollout_no_obstacles_trio40_b12_dp1_seed0.gif'
out = diagrams / 'rollout_no_obstacles_trio40_b12_dp1_seed0_until_one_alive.gif'
records = {}
for arm in ('solo', 'ego', 'ceiling'):
    path = diagrams / f'rollout_no_obstacles_{arm}40_b12_dp1_seed0.json'
    records[arm] = json.loads(path.read_text())
bootstrap = records['solo']['bootstrap_steps']
assert all(record['bootstrap_steps'] == bootstrap for record in records.values())

# Retirement stamps are simulation times. Frame zero is the last real frame.
candidates = []
for arm, record in records.items():
    deaths = sorted(t for t in record['retirement_step'].values() if t is not None)
    required = record['settings']['num_agents'] - 1
    if len(deaths) >= required:
        candidates.append((deaths[required - 1], arm))
cutoff, first_arm = min(candidates)
last_frame = cutoff - bootstrap
assert last_frame > 0

def survivors(time):
    return {arm: sum(t is None or t > time for t in record['retirement_step'].values())
            for arm, record in records.items()}

assert min(survivors(cutoff - 1).values()) >= 2
assert survivors(cutoff)[first_arm] == 1

# Copy the original GIF blocks through the desired image, then append a
# trailer. No re-encoding: palette, disposal, loop setting and timing stay exact.
data = source.read_bytes()
assert data[:6] in (b'GIF87a', b'GIF89a')
pos = 13
if data[10] & 0x80:
    pos += 3 * (2 ** ((data[10] & 7) + 1))

def skip_subblocks(pos):
    while True:
        size = data[pos]
        pos += 1
        if size == 0:
            return pos
        pos += size

images = 0
while images <= last_frame:
    tag = data[pos]
    pos += 1
    if tag == 0x21:  # extension label, followed by data subblocks
        pos = skip_subblocks(pos + 1)
    elif tag == 0x2C:  # image descriptor, optional local palette, LZW data
        packed = data[pos + 8]
        pos += 9
        if packed & 0x80:
            pos += 3 * (2 ** ((packed & 7) + 1))
        pos = skip_subblocks(pos + 1)
        images += 1
    else:
        raise ValueError(f'GIF ended or has an unexpected block before cutoff: {tag:#x}')
out.write_bytes(data[:pos] + b'\x3b')

with Image.open(source) as original, Image.open(out) as sliced:
    assert sliced.n_frames == last_frame + 1
    assert original.info.get('loop') == sliced.info.get('loop')
    durations = []
    for index in range(sliced.n_frames):
        original.seek(index)
        sliced.seek(index)
        assert np.array_equal(np.asarray(original.convert('RGB')),
                              np.asarray(sliced.convert('RGB'))), index
        assert original.info.get('duration') == sliced.info.get('duration')
        durations.append(sliced.info['duration'])

metadata = dict(
    source=source.name, output=out.name,
    source_sha256=hashlib.sha256(data).hexdigest(),
    output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    first_single_survivor_arm=first_arm,
    cutoff_simulation_step=cutoff, cutoff_generated_step=last_frame,
    frames=last_frame + 1, duration_ms=sum(durations),
    alive_before_cutoff=survivors(cutoff - 1), alive_at_cutoff=survivors(cutoff),
    retirement_frame_included=True, all_pixels_and_durations_match_source=True,
    method='Original GIF block prefix plus trailer; no re-encoding or added hold')
metadata_path = root / 'results/rollouts_no_obstacles_2026_09_08/slice_until_one_alive.json'
metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
