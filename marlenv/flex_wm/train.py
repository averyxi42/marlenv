"""Training over sets of pairs, with the attention window made explicit.

One thing here is not just a re-expression of the older loop. Training used
to crop an episode and let every token in the crop see every earlier token
in it, while a rollout slides a fixed window and so gives each token
exactly that many frames of history once it is warm. Those are not the same
computation: a token early in a crop is trained with less history than it
will ever have at play time, and a token late in a long crop with more.

``window`` closes that. Applied during training it caps how far back any
token may look, in frames, exactly as the cache does at play time. With a
crop of ``window`` frames it changes nothing, which is why it can be on by
default; with a crop of ``window`` plus enough frames for the receptive
field to fill, every token deep enough in the crop sees precisely what it
would see mid-rollout, and the gap is gone rather than merely small.
"""
import numpy as np
import torch

from marlenv.flex_wm.data import pairs_from_arrays
from marlenv.flex_wm.pairs import compact
from marlenv.wm.diffusion import add_noise, to_velocity
from marlenv.wm.matrain import MultiBatcher
from marlenv.wm.multiagent import actions_to_signal


class PairBatcher(MultiBatcher):
    """Crops, handed over as sets of pairs rather than as a rectangle.

    The rectangle is an accident of how the data was collected, so it is
    unpacked here and never mentioned again.
    """

    def pairs(self, size, model, drop_retired=True):
        """A crop as a set of pairs, plus the per-episode action weights.

        size         crops in the batch
        model        supplies the dead reckoning and the patch geometry
        drop_retired remove pairs that are not targets, rather than pinning
                     them at the top of the noise schedule

        Returns ``(PairBatch, weight (size,), dropout (size,))``.

        Removal is the default because a token that is present is a token
        the model can read: even pinned at maximum noise, its position and
        the count of them carry information a dead agent should not be
        supplying. A rollout stops simulating a dead agent outright, so
        removal is also what makes the two agree.
        """
        (frames, actions, alive, trained, origins, weight,
         dropout) = self.batch(size)
        # every observation is paired with the action taken from it,
        # including the last, which is the one a policy is asked for
        trailing = torch.cat([actions, actions[:, -1:]], dim=1)
        batch = pairs_from_arrays(frames, trailing, origins, alive, trained,
                                  model=model)
        if drop_retired:
            batch = compact(batch, batch.trained)
        return batch, weight, dropout


def flex_training_loss(model, pairs, weight=None, dropout=None, window=None,
                       generator=None):
    """Masked v-loss over the observations and actions of a set of pairs."""
    device = pairs.observations.device
    shape = pairs.observations.shape[:2]

    frame_tau = torch.rand(shape, device=device, generator=generator)
    action_tau = torch.rand(shape, device=device, generator=generator)
    # a pair that is not a target contributes nothing, said the way
    # diffusion forcing says it: pinned at the top of the noise schedule,
    # where the content is noise whatever was written there
    frame_tau = torch.where(pairs.trained, frame_tau,
                            torch.ones_like(frame_tau))
    action_tau = torch.where(pairs.acted, action_tau,
                             torch.ones_like(action_tau))

    # the same statement one level finer. A patch nobody saw is unknown,
    # not wrong: pinned at the top like an untrained observation, and kept
    # out of the loss, because there is no truth to regress it against.
    # With everything visible this is exactly ``trained`` again
    view, tokens = pairs.observations.shape[2], model.tokens_per_frame
    seen = pairs.cell_mask(tokens, view) & pairs.trained[:, :, None, None]
    ones = torch.ones_like(frame_tau)
    cell_tau = torch.where(seen, frame_tau[:, :, None, None],
                           ones[:, :, None, None])
    patch_tau = torch.where(
        pairs.patch_mask(tokens) & pairs.trained[:, :, None],
        frame_tau[:, :, None], ones[:, :, None])

    clean = actions_to_signal(pairs.actions, model.action_out.out_features)
    frame_noise = torch.randn(pairs.observations.shape, device=device,
                              generator=generator)
    action_noise = torch.randn(clean.shape, device=device,
                               generator=generator)

    predicted_frames, predicted_actions = model(
        pairs,
        add_noise(pairs.observations, cell_tau, frame_noise),
        add_noise(clean, action_tau, action_noise),
        patch_tau, action_tau, window=window)

    frame_target = to_velocity(pairs.observations, frame_noise, cell_tau)
    action_target = to_velocity(clean, action_noise, action_tau)

    # averaged over the cells that were seen, rather than per observation.
    # Patches partition the view evenly, so with nothing masked the two
    # give the same number
    squared = ((predicted_frames - frame_target) ** 2).mean(dim=-1)
    frame_mask = seen.float()
    frame_loss = ((squared * frame_mask).sum()
                  / frame_mask.sum().clamp(min=1))

    action_error = ((predicted_actions - action_target) ** 2).mean(dim=-1)
    contributing = pairs.acted.float()
    if dropout is not None:
        keep = torch.rand(contributing.shape, device=device,
                          generator=generator) >= dropout.view(-1, 1)
        contributing = contributing * keep.float()
    total = contributing.sum().clamp(min=1)

    action_loss = (action_error * contributing).sum() / total
    pull = (action_loss if weight is None else
            (action_error * contributing * weight.view(-1, 1)).sum() / total)
    return frame_loss + pull, frame_loss, action_loss
