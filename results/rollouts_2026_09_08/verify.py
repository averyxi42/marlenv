from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
labels=['solo24','ego24','ceiling24','solo40','ego40','ceiling40']
headings=np.array([[-1,0],[0,1],[1,0],[0,-1]])
reference=None
checks=[]
for label in labels:
    stem=ROOT/'diagrams'/f'rollout_corrected_{label}_b12_dp1_seed0'
    meta=json.loads(stem.with_suffix('.json').read_text())
    with np.load(stem.with_suffix('.prefix.npz')) as data:
        prefix={k:data[k] for k in data.files}
    if reference is None:
        reference=prefix
    else:
        assert prefix.keys()==reference.keys()
        for k in prefix:
            assert np.array_equal(prefix[k],reference[k]), (label,k)
    h=hashlib.sha256()
    for k,v in sorted(prefix.items()):
        h.update(k.encode());h.update(str(v.shape).encode());h.update(v.tobytes())
    assert h.hexdigest()==meta['prefix_sha256']
    assert not meta['code_dirty']
    with np.load(stem.with_suffix('.rollout.npz')) as trace:
        agent=trace['agent'];time=trace['time'];position=trace['position'];action=trace['action']
        for i in range(3):
            slots=np.flatnonzero(agent==i)
            times=time[slots];acts=action[slots]
            assert np.array_equal(times,np.arange(times[0],times[-1]+1)),(label,i,'gaps/duplicates')
            assert np.all((acts[:-1]>=0)&(acts[:-1]<4))
            assert acts[-1]==-1
            assert np.array_equal(np.diff(position[slots],axis=0),headings[acts[:-1]]),(label,i,'wrong action owner or pose')
            death=meta['retirement_step'][str(i)]
            if death is not None:
                assert times[-1]==death,(label,i,'post-death entries')
            else:
                assert times[-1]==meta['bootstrap_steps']+meta['generated_steps']
        rows=len(agent)
    for key,suffix in [('gif_sha256','.gif'),('trajectory_sha256','.rollout.npz')]:
        assert hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()==meta[key]
    with Image.open(stem.with_suffix('.gif')) as gif:
        assert gif.n_frames==meta['generated_steps']+1
        first=np.asarray(gif.convert('RGB'))
        if label==labels[0]:
            reference_first=first
        else:
            assert np.array_equal(first,reference_first),(label,'first GIF frame differs')
    checks.append(dict(label=label,generated_steps=meta['generated_steps'],
                       emitted_rows=rows,retirement_step=meta['retirement_step'],
                       prefix_sha256=meta['prefix_sha256'],code_commit=meta['code_commit']))
print(json.dumps(dict(checks=checks, verified=['All six actual prefixes and initial GIF frames identical',
    'Each agent has one contiguous sequence of emitted observations',
    'Every recorded move equals its own outgoing action, including fatal moves',
    'No rows or actions follow a death; no repeated death observations',
    'GIF/trajectory hashes and frame counts match metadata',
    'All runs use committed, unmodified source']),indent=2))
