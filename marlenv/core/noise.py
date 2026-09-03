"""Persistent observation noise, bound to what is on the board.

Per-frame i.i.d. noise makes the observation continuous but *unlearnable*:
nothing predicts the next frame's noise, so a generative world model can only
fit its marginal. Binding the noise to physical structure instead keeps the
observation continuous while leaving it fully determined by the state, which
is what a diffusion or flow-matching model needs in order to have something
to learn.

Two fields, resampled once per episode:

``cell_noise``
    One sample per grid cell, added to every non-snake cell. Background and
    obstacles therefore hold exactly the same value for the whole episode.

``snake_noise``
    ``(num_snakes, period, 3)``, indexed by snake and by distance from that
    snake's head. A body cell's noise is a function of how far back along the
    body it sits, so the pattern travels with the snake as it moves rather
    than staying pinned to the board.
"""
import numpy as np


class ObservationNoise:
    """Fixed noise fields for one episode's RGB observations."""

    def __init__(self, grid_shape, num_snakes, sigma=2.0, period=8,
                 np_random=None):
        rng = np_random if np_random is not None else np.random.default_rng()
        self.sigma = float(sigma)
        self.period = int(period)
        self.cell_noise = rng.normal(
            0.0, self.sigma, size=(*grid_shape, 3)).astype(np.float32)
        self.snake_noise = rng.normal(
            0.0, self.sigma,
            size=(num_snakes, self.period, 3)).astype(np.float32)

    def apply(self, rgb, snakes):
        """Return ``rgb`` with the bound noise added, as uint8.

        Snake cells take their snake's noise instead of the cell's, so a
        segment carries the same offset wherever it happens to be standing.
        """
        out = rgb.astype(np.float32) + self.cell_noise
        height, width = rgb.shape[:2]
        for snake in snakes:
            if not snake.alive:
                continue
            row = self.snake_noise[snake.idx % len(self.snake_noise)]
            for distance, (r, c) in enumerate(snake.coords):
                if 0 <= r < height and 0 <= c < width:
                    out[r, c] = rgb[r, c] + row[distance % self.period]
        return np.clip(out, 0, 255).astype(np.uint8)
