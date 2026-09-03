"""Training the multi-agent world action model.

Frames and actions are diffused under one objective, each with its own
independent noise level, so the model learns dynamics and policy at once.
That is what makes a rollout possible: the other agents' actions can be
sampled rather than supplied.
"""
import numpy as np
import torch

from marlenv.wm.data import to_model_input
from marlenv.wm.diffusion import add_noise, to_velocity
from marlenv.wm.multiagent import actions_to_signal


def multi_training_loss(model, frames, actions, alive, trained, origins,
                        action_weight=1.0, generator=None):
    """Masked v-loss over frames and actions together."""
    batch, steps, agents = frames.shape[:3]
    device = frames.device

    action_mask = alive[:, :-1] & alive[:, 1:]

    frame_tau = torch.rand(batch, steps, agents, device=device,
                           generator=generator)
    action_tau = torch.rand(batch, steps - 1, agents, device=device,
                            generator=generator)
    # A dead agent contributes nothing, expressed the way diffusion forcing
    # already expresses "no information": pin its tokens at the maximum
    # noise level, where alpha is zero and the input is pure noise whatever
    # was written there. Without this the model is fed a dead agent's stored
    # action, which is an all-zero one-hot that argmax reads as UP -- a
    # confident claim about an agent that did not act.
    frame_tau = torch.where(trained, frame_tau, torch.ones_like(frame_tau))
    action_tau = torch.where(action_mask, action_tau,
                             torch.ones_like(action_tau))

    frame_noise = torch.randn(frames.shape, device=device,
                              generator=generator)
    signal = actions_to_signal(actions, model.action_out.out_features)
    action_noise = torch.randn(signal.shape, device=device,
                               generator=generator)

    noisy_frames = add_noise(frames, frame_tau, frame_noise)
    noisy_actions = add_noise(signal, action_tau, action_noise)

    predicted_frames, predicted_actions = model(
        noisy_frames, noisy_actions, frame_tau, action_tau, origins=origins,
        action_indices=actions, alive=alive)

    frame_target = to_velocity(frames, frame_noise, frame_tau)
    action_target = to_velocity(signal, action_noise, action_tau)

    frame_error = ((predicted_frames - frame_target) ** 2).mean(
        dim=(-3, -2, -1))
    frame_mask = trained.float()
    frame_loss = ((frame_error * frame_mask).sum()
                  / frame_mask.sum().clamp(min=1))

    action_error = ((predicted_actions - action_target) ** 2).mean(dim=-1)
    # only train an action the agent was alive to take
    weights = action_mask.float()
    action_loss = ((action_error * weights).sum()
                   / weights.sum().clamp(min=1))

    return frame_loss + action_weight * action_loss, frame_loss, action_loss


class MultiBatcher:
    """Random fixed-length crops over multi-agent episodes."""

    def __init__(self, sequences, context, seed=0, device='cpu'):
        self.data = sequences
        self.context = context
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.lengths = sequences['mask'].sum(axis=1)
        self.usable = np.flatnonzero(self.lengths >= 2)

    def batch(self, size):
        picks = self.rng.choice(self.usable, size=size, replace=True)
        agents = self.data['observations'].shape[2]
        view = self.data['observations'].shape[3]
        context = self.context

        frames = np.zeros((size, context, agents, view, view, 3), np.uint8)
        actions = np.zeros((size, context - 1, agents), np.int64)
        alive = np.zeros((size, context, agents), bool)
        trained = np.zeros((size, context, agents), bool)
        origins = np.zeros((size, agents, 2), np.int64)

        for row, index in enumerate(picks):
            length = int(self.lengths[index])
            span = min(length, context)
            start = int(self.rng.integers(0, length - span + 1))
            stop = start + span
            frames[row, :span] = self.data['observations'][index, start:stop]
            alive[row, :span] = self.data['alive'][index, start:stop]
            trained[row, :span] = self.data['trained'][index, start:stop]
            take = max(span - 1, 0)
            actions[row, :take] = self.data['actions'][index,
                                                       start:start + take]
            # origins are re-based on the crop, since only differences matter
            origins[row] = self.data['origins'][index]

        to = lambda x: torch.from_numpy(x).to(self.device)
        return (to(to_model_input(frames)), to(actions), to(alive),
                to(trained), to(origins))
