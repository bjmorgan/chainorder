"""Order parameters for ReO3-type anion ordering (chain-level statistics)."""
import numpy as np


def chain_fft(anion_direction: np.ndarray) -> np.ndarray:
    """Discrete Fourier transform along each chain.

    Args:
        anion_direction: Binary species array along one chain direction (from
            `decompose()`), shape (N, N, N).

    Returns:
        Complex array of shape (N, N, N). Last axis is the Fourier index
        k = 0, 1, ..., N-1. Normalised so that a chain with a single F per
        period-p gives |tilde_s_{N/p}| = 1/p for stoichiometry 1/p.
    """
    N = anion_direction.shape[-1]
    return np.fft.fft(anion_direction, axis=-1) / N
