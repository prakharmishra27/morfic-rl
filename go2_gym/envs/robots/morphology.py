"""Shared morphology ordering, training bounds, and sampling."""

import torch


MORPHOLOGY_KEYS = (
    "thigh_mass",
    "calf_mass",
    "hip_mass",
    "foot_mass",
    "base_mass",
    "hip_limit_min",
    "hip_limit_max",
    "thigh_limit_min",
    "thigh_limit_max",
    "calf_limit_min",
    "calf_limit_max",
)

MORPHOLOGY_RANGES = {
    "thigh_mass": (0.634, 1.152),
    "calf_mass": (0.064, 0.166),
    "hip_mass": (0.510299, 0.696),
    "foot_mass": (0.0, 0.06),
    "base_mass": (3.3, 6.921),
    "hip_limit_min": (-1.10, -0.80),
    "hip_limit_max": (0.80, 1.10),
    "thigh_limit_min": (-1.65, -1.05),
    "thigh_limit_max": (3.49, 4.05),
    "calf_limit_min": (-2.72, -2.60),
    "calf_limit_max": (-0.95, -0.78),
}


def config_morphology(cfg):
    """Attach shared training bounds without changing the robot descriptor."""
    cfg.Morphology.morphology_range = dict(MORPHOLOGY_RANGES)
    cfg.Morphology.morphology_min = torch.tensor(
        [MORPHOLOGY_RANGES[key][0] for key in MORPHOLOGY_KEYS],
        dtype=torch.float32,
    )
    cfg.Morphology.morphology_max = torch.tensor(
        [MORPHOLOGY_RANGES[key][1] for key in MORPHOLOGY_KEYS],
        dtype=torch.float32,
    )


def _validate_bounds(morph_min, morph_max):
    if morph_min.shape != (11,) or morph_max.shape != (11,):
        raise ValueError("Morphology bounds must each have shape (11,).")
    if not torch.all(torch.isfinite(morph_min)):
        raise ValueError("Morphology minimum contains nonfinite values.")
    if not torch.all(torch.isfinite(morph_max)):
        raise ValueError("Morphology maximum contains nonfinite values.")
    if not torch.all(morph_max > morph_min):
        raise ValueError("Each morphology maximum must exceed its minimum.")


def normalize_morphology(raw, morph_min, morph_max):
    """Normalize using training bounds; retain out-of-range target values."""
    _validate_bounds(morph_min, morph_max)
    raw = torch.as_tensor(
        raw, dtype=morph_min.dtype, device=morph_min.device
    )
    if raw.ndim == 0 or raw.shape[-1] != 11:
        raise ValueError("Morphology descriptor must have 11 features.")
    return (raw - morph_min) / (morph_max - morph_min)


def denormalize_morphology(normalized, morph_min, morph_max):
    """Convert normalized descriptors back to physical parameters."""
    _validate_bounds(morph_min, morph_max)
    normalized = torch.as_tensor(
        normalized, dtype=morph_min.dtype, device=morph_min.device
    )
    if normalized.ndim == 0 or normalized.shape[-1] != 11:
        raise ValueError("Morphology descriptor must have 11 features.")
    return normalized * (morph_max - morph_min) + morph_min


def sample_morphologies(num_envs, morph_min, morph_max):
    """Sample independent normalized descriptors, one per environment."""
    _validate_bounds(morph_min, morph_max)
    return torch.rand(
        (num_envs, len(MORPHOLOGY_KEYS)),
        dtype=morph_min.dtype,
        device=morph_min.device,
    )