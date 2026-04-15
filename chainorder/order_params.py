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


def _canonical_rotation(window: tuple[int, ...]) -> tuple[int, ...]:
    """Return the lexicographically smallest cyclic rotation of a tuple.

    Args:
        window: Tuple of values to canonicalise.

    Returns:
        The rotation of `window` with the smallest lexicographic order.
    """
    return min(window[i:] + window[:i] for i in range(len(window)))


def motif_counts(
    anion_direction: np.ndarray,
    window_length: int,
) -> dict[tuple[int, ...], np.ndarray]:
    """Count cyclic-equivalence classes of length-`window_length` motifs per chain.

    Windows wrap periodically: every chain position is the start of exactly one
    window, so counts per chain sum to N regardless of `window_length`.

    Args:
        anion_direction: Binary species array along one chain direction,
            shape (N, N, N).
        window_length: Length of the sliding window.

    Returns:
        Dictionary mapping each canonical motif tuple to an integer array of
        shape (N, N) giving per-chain counts.
    """
    N = anion_direction.shape[-1]
    w = window_length

    # Build (N, w) index array: windows[start, offset] -> position in chain.
    window_idx = (np.arange(N)[:, None] + np.arange(w)[None, :]) % N   # (N, w)

    # windows[j, k, start, offset] = anion_direction[j, k, (start + offset) % N]
    windows = anion_direction[:, :, window_idx]                        # (N, N, N, w)

    # Encode each window as an integer: bit i = value at offset i.
    powers = (1 << np.arange(w)).astype(np.int64)                      # (w,)
    encoded = (windows.astype(np.int64) * powers).sum(axis=-1)         # (N, N, N)

    # Precompute canonical code for every possible window value.
    n_codes = 1 << w
    canon_code = np.empty(n_codes, dtype=np.int64)
    canon_tuple: dict[int, tuple[int, ...]] = {}
    for code in range(n_codes):
        bits = tuple((code >> i) & 1 for i in range(w))
        best = min(bits[i:] + bits[:i] for i in range(w))
        best_code = sum(b * (1 << i) for i, b in enumerate(best))
        canon_code[code] = best_code
        canon_tuple[best_code] = best

    canonical_encoded = canon_code[encoded]                            # (N, N, N)

    # Tally occurrences per chain (sum over the last axis, which is `start`).
    counts: dict[tuple[int, ...], np.ndarray] = {}
    for code in np.unique(canonical_encoded):
        mask = (canonical_encoded == code)
        counts[canon_tuple[int(code)]] = mask.sum(axis=-1).astype(np.int64)
    return counts


def along_chain_correlation(anion_direction: np.ndarray) -> np.ndarray:
    """Pair correlation g(r) along chains, averaged over all chains of one direction.

    g(r) = <s_i * s_{i+r}> - <s>^2, where the inner average is over position i
    along the chain and over all chains. Subtracting <s>^2 removes the mean-
    density contribution so g(r) oscillates around zero. Wrap-around is
    periodic.

    Args:
        anion_direction: Binary species array along one chain direction,
            shape (N, N, N).

    Returns:
        g(r) for r = 0, 1, ..., N-1, shape (N,).
    """
    N = anion_direction.shape[-1]
    s_mean = float(anion_direction.mean())

    # Vectorised: build a (N_r, N, N, N) stack of r-shifted arrays then average.
    # Memory cost is O(N^4); trivial for typical N (<=24).
    r = np.arange(N)
    shift_idx = (np.arange(N)[None, :] + r[:, None]) % N               # (N_r, N)
    shifted = anion_direction[..., shift_idx]                          # (N, N, N_r, N)
    product = anion_direction[..., None, :] * shifted                  # (N, N, N_r, N)
    g = product.mean(axis=(0, 1, 3)) - s_mean ** 2                     # (N_r,)
    return g
