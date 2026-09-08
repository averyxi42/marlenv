"""Check exported live/retired tiles against sparse raw emissions, then pairing."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from marlenv.core.palette import snap_to_palette

ROOT = Path(__file__).resolve().parents[2]
checks=[]
for meta_path in sorted((ROOT/'diagrams').glob('rollout_corrected_*.json')):
    stem=meta_path.with_suffix('');d=json.loads(meta_path.read_text());settings=d['settings']
    trace=np.load(stem.with_suffix('.rollout.npz'))
    view=2*settings['view_radius']+1;scale=settings['tile_scale']
    count=settings['num_agents'];total=0
    with Image.open(stem.with_suffix('.gif')) as gif:
        canvas_height=(settings['side']+2*settings['view_radius'])*settings['canvas_scale']
        top=14+canvas_height+18
        left=(gif.width-(count*view*scale+(count-1)*10))//2
        for step in range(gif.n_frames):
            gif.seek(step);rgb=np.asarray(gif.convert('RGB'));time=step+d['bootstrap_steps']
            for agent in range(count):
                death=d['retirement_step'][str(agent)]
                retired=death is not None and time>=death
                observation_time=death if retired else time
                slots=np.flatnonzero((trace['agent']==agent)&(trace['time']==observation_time))
                assert len(slots)==1
                expected=snap_to_palette(trace['observation'][slots[0]],6)
                if retired:
                    expected=(expected*0.3).astype(np.uint8)
                got=rgb[top+scale//2+np.arange(view)[:,None]*scale,
                        left+agent*(view*scale+10)+scale//2+np.arange(view)[None,:]*scale]
                assert np.array_equal(got,expected),(stem.name,step,agent,'tile colour changed')
                total+=view*view
    checks.append(dict(gif=stem.with_suffix('.gif').name,exact_tile_cells=total))
for budget in (24,40):
    with Image.open(ROOT/'diagrams'/f'rollout_corrected_trio{budget}_b12_dp1_seed0.gif') as combined:
        for column,arm in enumerate(['solo','ego','ceiling']):
            with Image.open(ROOT/'diagrams'/f'rollout_corrected_{arm}{budget}_b12_dp1_seed0.gif') as source:
                for index in range(source.n_frames):
                    combined.seek(index);source.seek(index)
                    rgb=np.asarray(combined.convert('RGB'))
                    left=18+column*(source.width+26)
                    panel=rgb[100:100+source.height,left:left+source.width]
                    assert np.array_equal(panel,np.asarray(source.convert('RGB'))),(budget,arm,index,'paired panel changed')
        checks.append(dict(gif=f'rollout_corrected_trio{budget}_b12_dp1_seed0.gif',
                           exact_active_panels=True,frames=combined.n_frames))
print(json.dumps(dict(checks=checks),indent=2))
