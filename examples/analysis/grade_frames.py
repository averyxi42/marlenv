"""Score next-frame prediction by cell type, for one model or several.

    python examples/analysis/grade_frames.py --models a.pt b.pt --names old tuned

Teacher forced: the history is real and clean, and only the final frame is
generated. That separates what the model can do from how a rollout drifts.

Single agent and multi agent checkpoints are both accepted and are scored
the same way, so snake quality can be compared directly between them --
which is the point, since the aggregate loss cannot be compared at all.
"""
import argparse

import torch
from datasets import load_from_disk

from marlenv.grading.frames import grade_multi, grade_single, show


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--names', nargs='+', default=None,
                   help='labels for the report; defaults to the paths')
    p.add_argument('--data-root', default='marlenv/demodata')
    p.add_argument('--component', default='expert')
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--context', type=int, default=None,
                   help='defaults to what each checkpoint was trained with')
    p.add_argument('--denoise-steps', type=int, default=16)
    p.add_argument('--limit', type=int, default=144,
                   help='viewpoints to score')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    return p.parse_args()


def load(path, device):
    """Rebuild a checkpoint, and say whether it carries several agents."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    multi = 'num_agents' in state
    if multi:
        from marlenv.wm.multiagent import MultiAgentWorldModel
        model = MultiAgentWorldModel(
            num_agents=state['num_agents'], view=state.get('view', 9),
            num_actions=state.get('num_actions', 4), frame='world',
            dim=state['dim'], depth=state['depth'], heads=state['heads'])
    else:
        from marlenv.wm.model import WorldModel
        model = WorldModel(
            view=state.get('view', 9), dim=state['dim'],
            depth=state['depth'], heads=state['heads'],
            num_actions=state.get('num_actions', 4),
            frame=state.get('frame', 'world'),
            align_coords=state.get('align_coords', False))
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state.get('context', 24), multi


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    names = args.names or args.models
    if len(names) != len(args.models):
        raise SystemExit('--names needs one label per model')

    dataset = load_from_disk(f'{args.data_root}/{args.component}')
    dataset = dataset.select(range(min(args.episodes, len(dataset))))
    built = {}

    for name, path in zip(names, args.models):
        model, trained_context, multi = load(path, device)
        context = args.context or trained_context
        kind = 'multi' if multi else 'single'
        if kind not in built:
            if multi:
                from marlenv.wm.madata import build_multi_sequences
                built[kind] = build_multi_sequences([dataset])
            else:
                from marlenv.wm.data import build_sequences
                built[kind] = build_sequences([dataset], frame='world')

        grade = grade_multi if multi else grade_single
        scores = grade(model, built[kind], context, device,
                       denoise_steps=args.denoise_steps, limit=args.limit,
                       seed=args.seed)
        show(f'{name}  [{kind}, context {context}]', scores)


if __name__ == '__main__':
    main()
