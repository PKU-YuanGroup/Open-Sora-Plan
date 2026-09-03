"""Helpers for temporal shapes shared by the dataset and sequence parallel path."""

import math


def _positive_int(value, name):
    """Normalize a configuration value and reject invalid stride values."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if normalized < 1 or normalized != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return normalized


def temporal_parallel_multiple(sp_size=1, patch_size_t=1):
    """Return the latent-frame multiple required by SP and temporal patching.

    Sequence parallelism partitions the latent time axis, while temporal
    patching consumes that same axis in fixed-size chunks.  Rounding to the
    least common multiple satisfies both constraints without adding avoidable
    padding when the two factors share a divisor.
    """
    sp_size = _positive_int(sp_size, "sp_size")
    patch_size_t = _positive_int(patch_size_t, "patch_size_t")
    return abs(sp_size * patch_size_t) // math.gcd(sp_size, patch_size_t)


def latent_frame_count(num_frames, ae_stride_t):
    """Return the number of causal-VAE latents needed for raw frames.

    Causal video VAEs retain the first frame and downsample subsequent frames
    by ``ae_stride_t``.  The ceiling is intentional: this helper is used for
    padding arbitrary input lengths and therefore must never return a raw
    length smaller than the input.
    """
    num_frames = _positive_int(num_frames, "num_frames")
    ae_stride_t = _positive_int(ae_stride_t, "ae_stride_t")
    return (num_frames - 1 + ae_stride_t - 1) // ae_stride_t + 1


def pad_temporal_frames(
    num_frames,
    ae_stride_t,
    patch_size_t=1,
    sp_size=1,
    preserve_single_frame=False,
):
    """Return a causal-VAE-aligned raw frame length safe for SP.

    The returned length is at least ``num_frames`` and encodes to a latent
    temporal length divisible by the least common multiple of ``sp_size`` and
    ``patch_size_t``.  A single-frame batch can opt out of padding: the
    training loop treats such batches as images and deliberately disables SP.
    """
    num_frames = _positive_int(num_frames, "num_frames")
    ae_stride_t = _positive_int(ae_stride_t, "ae_stride_t")
    multiple = temporal_parallel_multiple(sp_size, patch_size_t)

    if preserve_single_frame and num_frames == 1:
        return 1

    latent_frames = latent_frame_count(num_frames, ae_stride_t)
    padded_latent_frames = ((latent_frames + multiple - 1) // multiple) * multiple
    return (padded_latent_frames - 1) * ae_stride_t + 1


def find_closest_y(x, vae_stride_t=4, model_ds_t=1, min_num_frames=29):
    """Find the largest VAE-compatible frame count not exceeding ``x``.

    ``model_ds_t`` is the required latent-time multiple.  The arithmetic form
    avoids scanning every candidate frame and keeps dataset indexing cheap for
    long clips.  ``-1`` preserves the historical signal for clips shorter
    than the minimum encodable duration.
    """
    original_x = x
    if isinstance(original_x, bool):
        raise TypeError(f"x must be a non-negative integer, got {original_x!r}")
    try:
        x = int(original_x)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"x must be a non-negative integer, got {original_x!r}"
        ) from exc
    if x < 0 or x != original_x:
        raise ValueError(f"x must be a non-negative integer, got {original_x!r}")
    vae_stride_t = _positive_int(vae_stride_t, "vae_stride_t")
    model_ds_t = _positive_int(model_ds_t, "model_ds_t")
    min_num_frames = _positive_int(min_num_frames, "min_num_frames")

    if x < min_num_frames:
        return -1

    # Largest latent index whose corresponding raw frame count is <= x.
    max_latent_frames = (x - 1) // vae_stride_t + 1
    # Smallest latent index whose corresponding raw frame count is >= the
    # minimum.  This differs from latent_frame_count for non-aligned minima.
    min_latent_frames = (min_num_frames - 1 + vae_stride_t - 1) // vae_stride_t + 1
    candidate = max_latent_frames - max_latent_frames % model_ds_t
    if candidate < min_latent_frames:
        return -1
    return (candidate - 1) * vae_stride_t + 1
