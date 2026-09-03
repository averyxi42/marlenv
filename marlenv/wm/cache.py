"""A key/value cache with a sliding window over frames.

Playing generates one frame at a time, and each frame takes many denoising
steps that all see the same history. Recomputing that history every step is
the bulk of the work, so it is computed once and kept.

The window is expressed in frames rather than tokens, since that is what the
model reasons about; a frame contributes its patch tokens plus the action
that follows it.
"""
import torch


class KVCache:
    """Per-layer keys and values for tokens already committed."""

    def __init__(self, layers, tokens_per_frame):
        self.layers = layers
        self.tokens_per_frame = tokens_per_frame
        self.keys = [None] * layers
        self.values = [None] * layers
        self.recording = False
        self.frames = 0

    def __len__(self):
        return 0 if self.keys[0] is None else self.keys[0].shape[2]

    def reset(self):
        self.keys = [None] * self.layers
        self.values = [None] * self.layers
        self.frames = 0

    def extend(self, layer, key, value):
        """Return the full key/value for this layer, recording if asked.

        Always concatenates the cache with the new tokens, so the caller
        attends over both; appending happens only while recording, which
        keeps the many denoising passes over a provisional frame from
        polluting the cache.
        """
        past_key, past_value = self.keys[layer], self.values[layer]
        if past_key is None:
            full_key, full_value = key, value
        else:
            full_key = torch.cat([past_key, key], dim=2)
            full_value = torch.cat([past_value, value], dim=2)
        if self.recording:
            self.keys[layer] = full_key
            self.values[layer] = full_value
        return full_key, full_value

    def trim(self, max_frames):
        """Drop whole frames off the front, oldest first.

        Trimming in frame units keeps a frame's patches and the action that
        follows them together; splitting them would leave the model an
        action whose frame it cannot see.
        """
        if max_frames is None or self.frames <= max_frames:
            return 0
        drop_frames = self.frames - max_frames
        drop_tokens = drop_frames * (self.tokens_per_frame + 1)
        for layer in range(self.layers):
            if self.keys[layer] is not None:
                self.keys[layer] = self.keys[layer][:, :, drop_tokens:]
                self.values[layer] = self.values[layer][:, :, drop_tokens:]
        self.frames -= drop_frames
        return drop_frames


class recording:
    """Context manager marking a pass whose tokens should be kept."""

    def __init__(self, cache):
        self.cache = cache

    def __enter__(self):
        self.cache.recording = True
        return self.cache

    def __exit__(self, *exc):
        self.cache.recording = False
        return False
