"""
Shared noise generators for the stochastic-resonance pipeline.

Supports:
  - Gaussian / AWGN
  - Ornstein-Uhlenbeck (colored / correlated)
  - Levy-stable (white / heavy-tailed)
"""

from __future__ import annotations

import numpy as np

DEFAULT_OU_TAU = 5.0
DEFAULT_OU_STEP = 1.0
DEFAULT_LEVY_ALPHA = 1.5
DEFAULT_LEVY_BETA = 0.0

NOISE_TYPE_ALIASES = {
    "gaussian": "gaussian",
    "awgn": "gaussian",
    "ou": "ou",
    "levy": "levy",
}


def canonical_noise_type(noise_type: str) -> str:
    """Map user-facing aliases onto the internal generator name."""
    try:
        return NOISE_TYPE_ALIASES[noise_type]
    except KeyError as exc:
        raise ValueError(f"Unknown noise type: {noise_type!r}") from exc


def generate_gaussian(
    size: int | tuple[int, ...],
    sigma: float,
    seed: int = 0,
) -> np.ndarray:
    """Additive white Gaussian noise with standard deviation sigma."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=sigma, size=size)


def generate_ou(
    size: int | tuple[int, ...],
    sigma: float,
    tau: float = DEFAULT_OU_TAU,
    h: float = DEFAULT_OU_STEP,
    seed: int = 0,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck noise with stationary variance sigma^2.

    Accepted shapes:
      - int: a single trajectory of length n
      - tuple[int, int]: (n_trials, n_steps) matrix of trajectories
    """
    size_tuple = (size,) if isinstance(size, int) else tuple(size)
    if len(size_tuple) not in (1, 2):
        raise ValueError("OU noise supports only 1D or 2D shapes.")

    rng = np.random.default_rng(seed)
    a = np.exp(-h / tau)
    diffusion = sigma * np.sqrt(1.0 - a**2)

    if len(size_tuple) == 1:
        n_steps = size_tuple[0]
        eta = np.empty(n_steps, dtype=float)
        eta[0] = 0.0
        for step in range(1, n_steps):
            eta[step] = a * eta[step - 1] + diffusion * rng.standard_normal()
        return eta

    n_trials, n_steps = size_tuple
    eta = np.empty((n_trials, n_steps), dtype=float)
    eta[:, 0] = 0.0
    for step in range(1, n_steps):
        eta[:, step] = (
            a * eta[:, step - 1] + diffusion * rng.standard_normal(n_trials)
        )
    return eta


def generate_levy_stable(
    size: int | tuple[int, ...],
    sigma: float,
    alpha: float = DEFAULT_LEVY_ALPHA,
    beta: float = DEFAULT_LEVY_BETA,
    seed: int = 0,
) -> np.ndarray:
    """
    Symmetric or skewed alpha-stable noise via the CMS sampler.

    The pipeline uses these as white heavy-tailed perturbations on biomarker
    signals, so sigma is treated as a direct amplitude parameter rather than a
    time-discretised SDE increment scale.
    """
    if not (0.0 < alpha <= 2.0):
        raise ValueError("levy alpha must be in (0, 2].")
    if not (-1.0 <= beta <= 1.0):
        raise ValueError("levy beta must be in [-1, 1].")
    if np.isclose(alpha, 2.0):
        return generate_gaussian(size, sigma, seed=seed)

    rng = np.random.default_rng(seed)
    phi = (rng.random(size=size) - 0.5) * np.pi
    w = rng.exponential(scale=1.0, size=size)

    if np.isclose(alpha, 1.0):
        denom = np.clip(np.pi / 2.0 + beta * phi, 1e-12, None)
        cos_phi = np.clip(np.cos(phi), 1e-12, None)
        xi = (2.0 / np.pi) * (
            denom * np.tan(phi)
            - beta * np.log(((np.pi / 2.0) * w * cos_phi) / denom)
        )
    else:
        theta0 = np.arctan(beta * np.tan(np.pi * alpha / 2.0)) / alpha
        cos_phi = np.clip(np.cos(phi), 1e-12, None)
        inner = np.cos(phi - alpha * (phi + theta0)) / w
        inner = np.clip(inner, 1e-12, None)
        xi = (
            np.sin(alpha * (phi + theta0))
            / (cos_phi ** (1.0 / alpha))
            * (inner ** ((1.0 - alpha) / alpha))
        )

    return sigma * xi


def generate_noise(
    noise_type: str,
    size: int | tuple[int, ...],
    sigma: float,
    *,
    tau: float = DEFAULT_OU_TAU,
    h: float = DEFAULT_OU_STEP,
    levy_alpha: float = DEFAULT_LEVY_ALPHA,
    levy_beta: float = DEFAULT_LEVY_BETA,
    seed: int = 0,
) -> np.ndarray:
    """Unified dispatcher used across the SR analysis scripts."""
    kind = canonical_noise_type(noise_type)
    if kind == "gaussian":
        return generate_gaussian(size, sigma, seed=seed)
    if kind == "ou":
        return generate_ou(size, sigma, tau=tau, h=h, seed=seed)
    if kind == "levy":
        return generate_levy_stable(
            size, sigma, alpha=levy_alpha, beta=levy_beta, seed=seed
        )
    raise ValueError(f"Unknown noise type: {noise_type!r}")
