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
from marlenv.wm.diffusion import add_noise, to_velocity
from marlenv.wm.matrain import MultiBatcher
from marlenv.wm.multiagent import actions_to_signal


class PairBatcher(MultiBatcher):
    """Crops, handed over as sets of pairs rather than as a rectangle.

    The rectangle is an accident of how the data was collected, so it is
    unpacked here and never mentioned again.
    """

    def pairs(self, size, model):
        (frames, actions, alive, trained, origins, weight,
         dropout) = self.batch(size)
        # every observation is paired with the action taken from it,
        # including the last, which is the one a policy is asked for
        trailing = torch.cat([actions, actions[:, -1:]], dim=1)
        batch = pairs_from_arrays(frames, trailing, origins, alive, trained,
                                  model=model)
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

    clean = actions_to_signal(pairs.actions, model.action_out.out_features)
    frame_noise = torch.randn(pairs.observations.shape, device=device,
                              generator=generator)
    action_noise = torch.randn(clean.shape, device=device,
                               generator=generator)

    predicted_frames, predicted_actions = model(
        pairs,
        add_noise(pairs.observations, frame_tau, frame_noise),
        add_noise(clean, action_tau, action_noise),
        frame_tau, action_tau, window=window)

    frame_target = to_velocity(pairs.observations, frame_noise, frame_tau)
    action_target = to_velocity(clean, action_noise, action_tau)

    frame_error = ((predicted_frames - frame_target) ** 2).mean(
        dim=(-3, -2, -1))
    frame_mask = pairs.trained.float()
    frame_loss = ((frame_error * frame_mask).sum()
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
