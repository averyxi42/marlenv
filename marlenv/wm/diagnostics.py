"""Checks that separate a good objective score from a useful model.

The headline training loss averages over noise levels, which hides the one
that matters: a rollout starts from pure noise, so the model's ability to
reconstruct a frame at high tau is the whole game. An epsilon-parameterised
model scored 0.0003 at tau = 1 -- effectively perfect -- while its
reconstruction error there was 1.60 on unit-variance data, because echoing
its own input satisfies the objective exactly. :func:`noise_level_report`
splits the two apart.
"""
import numpy as np
import torch

from marlenv.wm.diffusion import add_noise, from_velocity


@torch.no_grad()
def noise_level_report(model, batcher, levels=(0.05, 0.2, 0.4, 0.6, 0.8,
                                               0.95, 1.0),
                       batches=4, batch_size=16, seed=0):
    """Reconstruction error per noise level.

    Returns a list of ``{tau, target_mse, reconstruction_mse, echo}``.
    ``reconstruction_mse`` is what a rollout depends on. ``echo`` is the
    cosine similarity between the prediction and the model's own input: near
    1 at high tau means the model is repeating what it was given.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)

    report = []
    for tau_value in levels:
        reconstruction, echo = [], []
        for _ in range(batches):
            frames, actions, mask = batcher.batch(batch_size)
            tau = torch.full(frames.shape[:2], float(tau_value),
                             device=device)
            noise = torch.randn(frames.shape, device=device,
                                generator=generator)
            noisy = add_noise(frames, tau, noise)
            predicted = model(noisy, actions, tau)

            clean, _ = from_velocity(noisy, predicted, tau)
            # mean over pixels first, then a masked mean over frames -- the
            # same normalisation the loss uses, so the numbers are per pixel
            per_frame = ((clean.clamp(-1, 1) - frames) ** 2).mean(
                dim=(-3, -2, -1))
            reconstruction.append(float(
                (per_frame * mask.float()).sum() / mask.sum().clamp(min=1)))
            echo.append(float(torch.nn.functional.cosine_similarity(
                predicted.flatten(2), noisy.flatten(2), dim=-1)[mask].mean()))

        report.append({'tau': tau_value,
                       'reconstruction_mse': float(np.mean(reconstruction)),
                       'echo': float(np.mean(echo))})
    model.train(was_training)
    return report


def format_report(report):
    lines = [f'{"tau":>6} {"reconstruction":>15} {"echo(pred,input)":>18}']
    for row in report:
        lines.append(f'{row["tau"]:6.2f} {row["reconstruction_mse"]:15.4f} '
                     f'{row["echo"]:18.3f}')
    return '\n'.join(lines)
