"""Diffusion forcing: independent noise levels per frame.

Training draws a noise level for every frame independently rather than one
for the whole sequence. That is the whole trick: the model sees histories at
every mixture of noise levels, so at rollout it can denoise the next frame
against a history that is already clean, which is exactly the case plain
video diffusion never trains on.
"""
import math

import torch

MAX_TAU = 1.0


def alpha_sigma(tau):
    """Cosine schedule. ``tau=0`` is clean, ``tau=1`` is pure noise."""
    alpha_bar = torch.cos(tau * math.pi / 2) ** 2
    return alpha_bar.sqrt(), (1 - alpha_bar).sqrt()


def add_noise(frames, tau, noise):
    """``(b, t, v, v, c)`` frames noised at a per-frame level."""
    alpha, sigma = alpha_sigma(tau)
    shape = (*tau.shape, 1, 1, 1)
    return alpha.view(shape) * frames + sigma.view(shape) * noise


def training_loss(model, frames, actions, mask, generator=None):
    """Masked noise-prediction loss with per-frame noise levels."""
    batch, steps = frames.shape[:2]
    device = frames.device
    tau = torch.rand(batch, steps, device=device, generator=generator)
    noise = torch.randn(frames.shape, device=device, generator=generator)

    noisy = add_noise(frames, tau, noise)
    predicted = model(noisy, actions, tau)

    # mean over pixels first, then a masked mean over frames, so the value
    # is comparable across sequence lengths and is ~1 for an untrained model
    per_frame = ((predicted - noise) ** 2).mean(dim=(-3, -2, -1))
    return (per_frame * mask.float()).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def denoise_next(model, history, actions, denoise_steps=16, generator=None,
                 temperature=1.0):
    """Sample the frame that follows ``history`` under ``actions``.

    The history is passed at noise level zero, which is the situation
    diffusion forcing trains for.
    """
    device = history.device
    batch, past = history.shape[:2]
    frame = torch.randn((batch, 1, *history.shape[2:]), device=device,
                        generator=generator) * temperature

    taus = torch.linspace(MAX_TAU, 0.0, denoise_steps + 1, device=device)
    for step in range(denoise_steps):
        tau_now, tau_next = taus[step], taus[step + 1]
        sequence = torch.cat([history, frame], dim=1)
        levels = torch.zeros((batch, past + 1), device=device)
        levels[:, -1] = tau_now

        predicted = model(sequence, actions, levels)[:, -1:]
        alpha, sigma = alpha_sigma(tau_now)
        clean = (frame - sigma * predicted) / alpha.clamp(min=1e-4)
        clean = clean.clamp(-1.0, 1.0)
        alpha_next, sigma_next = alpha_sigma(tau_next)
        frame = alpha_next * clean + sigma_next * predicted
    return frame


@torch.no_grad()
def rollout(model, context, actions, horizon, denoise_steps=16,
            generator=None):
    """Generate ``horizon`` frames after ``context``, one at a time.

    ``actions`` must cover every transition being generated, i.e. at least
    ``context.shape[1] + horizon - 1`` of them.
    """
    history = context
    needed = history.shape[1] + horizon - 1
    if actions.shape[1] < needed:
        raise ValueError(f'need {needed} actions, got {actions.shape[1]}')

    for _ in range(horizon):
        step = history.shape[1]
        frame = denoise_next(model, history, actions[:, :step],
                             denoise_steps=denoise_steps,
                             generator=generator)
        history = torch.cat([history, frame], dim=1)
    return history
