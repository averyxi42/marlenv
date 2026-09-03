"""Training loop for the single-agent world model."""
import time

import numpy as np
import torch

from marlenv.wm.data import to_model_input
from marlenv.wm.diffusion import training_loss


class SequenceBatcher:
    """Serves random crops of fixed length from padded sequences.

    Cropping rather than always starting at frame 0 keeps the model from
    only ever seeing episode openings, and keeps sequence length -- and so
    attention cost -- fixed.
    """

    def __init__(self, sequences, context, seed=0, device='cpu'):
        self.observations = sequences['observations']
        self.actions = sequences['actions']
        self.mask = sequences['mask']
        self.context = context
        self.device = device
        self.rng = np.random.default_rng(seed)

        lengths = self.mask.sum(axis=1)
        # a crop needs at least two frames to contain a transition
        self.usable = np.flatnonzero(lengths >= 2)
        self.lengths = lengths

    def __len__(self):
        return len(self.usable)

    def batch(self, size):
        picks = self.rng.choice(self.usable, size=size, replace=True)
        frames = np.zeros((size, self.context, *self.observations.shape[2:]),
                          dtype=np.uint8)
        actions = np.zeros((size, self.context - 1), dtype=np.int64)
        mask = np.zeros((size, self.context), dtype=bool)

        for row, index in enumerate(picks):
            length = int(self.lengths[index])
            span = min(length, self.context)
            start = int(self.rng.integers(0, length - span + 1))
            frames[row, :span] = self.observations[index, start:start + span]
            mask[row, :span] = True
            take = max(span - 1, 0)
            actions[row, :take] = self.actions[index, start:start + take]

        return (
            torch.from_numpy(to_model_input(frames)).to(self.device),
            torch.from_numpy(actions).to(self.device),
            torch.from_numpy(mask).to(self.device),
        )


def train(model, batcher, steps=2000, batch_size=32, lr=3e-4,
          warmup=100, log_every=100, device='cpu', validation=None,
          on_log=None):
    """Plain AdamW with cosine decay; returns the loss history."""
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=0.01, betas=(0.9, 0.95))

    def learning_rate(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    history, window, start = [], [], time.time()

    for step in range(steps):
        frames, actions, mask = batcher.batch(batch_size)
        loss = training_loss(model, frames, actions, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        window.append(loss.item())

        if (step + 1) % log_every == 0:
            record = {'step': step + 1, 'loss': float(np.mean(window)),
                      'lr': schedule.get_last_lr()[0],
                      'elapsed': time.time() - start}
            if validation is not None:
                record['val_loss'] = evaluate(model, validation, device)
            history.append(record)
            window = []
            if on_log:
                on_log(record)
    return history


@torch.no_grad()
def evaluate(model, batcher, device='cpu', batches=8, batch_size=32,
             seed=1234):
    """Held-out loss at a fixed set of noise levels, so it is comparable."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        frames, actions, mask = batcher.batch(batch_size)
        total += training_loss(model, frames, actions, mask,
                               generator=generator).item()
    model.train(was_training)
    return total / batches
