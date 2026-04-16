"""Order parameters for ReO3-type anion ordering (chain-level statistics)."""
import numpy as np


def _check_binary_chain_array(arr: np.ndarray, *, name: str = "anion_direction") -> None:
    """Validate that `arr` is a 3D cubic integer array (contract for order-param inputs).

    Value check (``0 <= arr <= 1``) is deliberately omitted: `decompose`
    produces 0/1 output by construction, so the guard here is about catching
    non-integer arrays (float buckets where the `|F| = 1/p` normalisation
    identity would silently decay) and wrong-shape inputs.
    """
    if arr.ndim != 3 or not (arr.shape[0] == arr.shape[1] == arr.shape[2]):
        raise ValueError(
            f"{name} must be a cubic 3D array (shape (N, N, N)), got "
            f"shape {arr.shape}."
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            f"{name} must have integer dtype, got {arr.dtype}. Order-parameter "
            f"functions assume the binary species encoding produced by "
            f"`decompose` and their normalisation identities do not hold for "
            f"float inputs."
        )


def chain_fft(anion_direction: np.ndarray) -> np.ndarray:
    """Discrete Fourier transform along each chain.

    Args:
        anion_direction: Binary species array along one chain direction (from
            `decompose()`), shape (N, N, N), integer dtype.

    Returns:
        Complex array of shape (N, N, N). Last axis is the Fourier index
        k = 0, 1, ..., N-1. Computed as `np.fft.fft(arr, axis=-1) / N`.
        For a chain that is strictly periodic with period `p` (N divisible
        by `p`, exactly one flagged atom per period), the DC component and
        the peak at `k = N // p` each have magnitude `1 / p`.

    Raises:
        TypeError: If `anion_direction` has non-integer dtype.
        ValueError: If `anion_direction` is not cubic 3D.
    """
    _check_binary_chain_array(anion_direction)
    N = anion_direction.shape[-1]
    return np.fft.fft(anion_direction, axis=-1) / N


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
        window_length: Length of the sliding window. Must satisfy
            ``1 <= window_length <= min(N, 62)`` (see `Raises`).

    Returns:
        Dictionary mapping each canonical motif tuple to an integer array of
        shape (N, N) giving per-chain counts.

    Raises:
        TypeError: If `window_length` is not an integer, or if
            `anion_direction` has a non-integer dtype (floats silently
            truncate through the bit-packing cast).
        ValueError: If `window_length` is outside the valid range. The upper
            bound of 62 is a hard limit of the int64 bit-packing used for
            the canonicalisation table; the upper bound of N is geometric
            (larger windows would wrap around the chain and alias).

    Notes:
        The canonicalisation table is materialised as an array of size
        `2 ** window_length` (one entry per possible window value). This
        is negligible for typical `window_length <= 8` but grows
        exponentially -- `window_length = 30` already occupies 8 GB. The
        int64 limit of 62 is the theoretical ceiling; practical use
        should keep `window_length` small.
    """
    if not isinstance(window_length, (int, np.integer)):
        raise TypeError(
            f"window_length must be an integer, got "
            f"{type(window_length).__name__}."
        )
    _check_binary_chain_array(anion_direction)
    N = anion_direction.shape[-1]
    w = int(window_length)
    if w < 1:
        raise ValueError(f"window_length must be >= 1, got {w}.")
    if w > N:
        raise ValueError(
            f"window_length ({w}) exceeds chain length N={N}; windows would "
            f"wrap around the chain and alias. Use window_length <= N."
        )
    if w > 62:
        raise ValueError(
            f"window_length ({w}) exceeds the int64 bit-packing limit of 62. "
            f"This limit is well above any realistic chain length."
        )

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

    `g(r) = <s_i * s_{i + r}> - <s>^2`. Both averages are grand averages
    over all chain positions `i` and all chains `(j, k)` of the given
    direction; they are NOT per-chain means. Subtracting `<s>^2` removes
    the mean-density contribution so `g(r)` oscillates around zero.
    Wrap-around is periodic.

    Args:
        anion_direction: Binary species array along one chain direction,
            shape (N, N, N), integer dtype.

    Returns:
        g(r) for r = 0, 1, ..., N-1, shape (N,).

    Raises:
        TypeError: If `anion_direction` has non-integer dtype.
        ValueError: If `anion_direction` is not cubic 3D.
    """
    _check_binary_chain_array(anion_direction)
    N = anion_direction.shape[-1]
    s_mean = float(anion_direction.mean())

    # Wiener-Khinchin: g(r) = IFFT(|FFT(s)|^2) / N, averaged over chains,
    # minus the mean-density contribution. O(N^3 log N) instead of the O(N^4)
    # broadcast-indexed stack of r-shifted copies.
    fft_per_chain = np.fft.fft(anion_direction, axis=-1)               # (N, N, N)
    power = np.abs(fft_per_chain) ** 2                                 # (N, N, N)
    autocorr = np.fft.ifft(power, axis=-1).real / N                    # (N, N, N)
    g = autocorr.mean(axis=(0, 1)) - s_mean ** 2                       # (N,)
    return g


def inter_chain_correlation(anion_direction: np.ndarray) -> np.ndarray:
    """Inter-chain correlation of the period-3 Fourier component.

    For each chain at lateral position `(a, b)`, let `phi(a, b)` be the
    Fourier coefficient at `k = N / 3` (i.e. the amplitude and phase of
    period-3 ordering on that chain). This function returns the spatial
    autocorrelation of `phi` across the chain plane, normalised so that
    `G[0, 0] = 1`:

        G[da, db] = < phi(a, b) * conj(phi(a + da, b + db)) > / < |phi|^2 >

    where the averages run over all `(a, b)` pairs (with periodic wrap in
    `a, b`). Amplitude-weighted: chains with small `|phi|` (disordered)
    contribute proportionally less, so disordered input gives `|G| ~ 0`
    off the origin by construction -- no threshold or guard required.

    `|G|` ranges from `~0` (uncorrelated) to `1` (fully phase-locked);
    `arg(G)` encodes the phase pattern (`0` for uniform alignment, a
    linear gradient for a striped arrangement, etc.).

    A phase-only inter-chain correlator (in which every chain is treated
    as equally ordered regardless of its Fourier amplitude) is a *different*
    quantity from the amplitude-weighted form returned here and cannot be
    recovered from `G` alone -- it requires per-chain phases from
    `chain_fft(arr)[..., N // 3]`. If you need that variant, compute
    `v = np.exp(1j * np.angle(phi))` yourself and take the spatial
    autocorrelation of `v`.

    Args:
        anion_direction: Binary species array along one chain direction,
            shape (N, N, N), integer dtype.

    Returns:
        Complex array of shape (N, N). `G[da, db]` for da, db = 0..N-1.

    Raises:
        TypeError: If `anion_direction` has non-integer dtype.
        ValueError: If `anion_direction` is not cubic 3D, if N is not
            divisible by 3 (no well-defined period-3 phase), or if the
            mean period-3 power is zero or small enough that the
            normalised result is not finite (e.g. all-O or all-F input,
            or subnormal power). To generalise to other wavevectors,
            compute `chain_fft` and extract the component manually.
    """
    _check_binary_chain_array(anion_direction)
    N = anion_direction.shape[-1]
    if N % 3 != 0:
        raise ValueError(
            f"inter_chain_correlation requires N divisible by 3 (for period-3 "
            f"phase), got N={N}."
        )
    phi = chain_fft(anion_direction)[..., N // 3]                      # (N, N)
    power = float(np.mean(np.abs(phi) ** 2))
    if power == 0.0:
        raise ValueError(
            "All chains have zero period-3 amplitude; correlation is "
            "undefined. Inspect chain_fft(arr)[..., N // 3] to confirm."
        )

    # Wiener-Khinchin: IFFT2(|FFT2(phi)|^2) / N^2 is the spatial
    # autocorrelation < phi(a, b) * conj(phi(a - da, b - db)) >. The direct
    # form uses the opposite lag sign, so take the complex conjugate to
    # match < phi(a, b) * conj(phi(a + da, b + db)) >. O(N^2 log N) instead
    # of the (N, N, N, N) shifted-index tensor.
    phi_k = np.fft.fft2(phi)                                           # (N, N)
    numer = np.conj(np.fft.ifft2(np.abs(phi_k) ** 2)) / N ** 2         # (N, N)
    result: np.ndarray = numer / power
    # Subnormal `power` passes the exact-zero check but can push the division
    # into inf / NaN. Catch that here rather than silently returning NaN.
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "inter_chain_correlation produced non-finite output; the mean "
            "period-3 power is too small for reliable normalisation. "
            f"power = {power!r}."
        )
    return result


def structure_factor(
    anion_x: np.ndarray,
    anion_y: np.ndarray,
    anion_z: np.ndarray,
) -> np.ndarray:
    """Anion-sublattice 3D structure factor in a canonical `(kx, ky, kz)` frame.

    Each of the three edge-midpoint sublattices is Fourier-transformed,
    transposed into canonical `(kx, ky, kz)` axes, and summed with the
    half-cell phase offset of its own sublattice relative to the cation
    corner (x-anions sit at `+a/2` along x, y-anions at `+a/2` along y,
    z-anions at `+a/2` along z). The result is the full coherent structure
    factor of the flagged-species occupation, with unit form factor.

    Rotation-equivariant: a lattice-symmetry rotation of the input
    structure produces the correspondingly rotated output. Anisotropy is
    preserved -- chains ordered along one direction give peaks on the
    matching reciprocal axis.

    Args:
        anion_x: x-chain occupation, shape `(N, N, N)`, axes `(j, k, i)`.
        anion_y: y-chain occupation, shape `(N, N, N)`, axes `(i, k, j)`.
        anion_z: z-chain occupation, shape `(N, N, N)`, axes `(i, j, k)`.

    Returns:
        Complex array of shape `(N, N, N)` with axes `(kx, ky, kz)`.
        `|F|**2` is proportional to the kinematic diffuse scattering
        intensity at wavevector `(kx, ky, kz) / N` reciprocal lattice
        units. Normalised so that a fully F-occupied anion sublattice
        gives `F[0, 0, 0] = 3` (three F per unit cell) and a single-
        sublattice period-3 ordering gives `|F| = 1/3` at its peak.

    Notes:
        Per-sublattice contributions can be recovered by zeroing the other
        two arrays: e.g. `structure_factor(ax, np.zeros_like(ax),
        np.zeros_like(ax))` returns the x-sublattice contribution alone.
        For per-chain (not cross-chain) analysis use `chain_fft` instead.
    """
    _check_binary_chain_array(anion_x, name="anion_x")
    _check_binary_chain_array(anion_y, name="anion_y")
    _check_binary_chain_array(anion_z, name="anion_z")
    if anion_x.shape != anion_y.shape or anion_x.shape != anion_z.shape:
        raise ValueError(
            f"All three sublattice arrays must have the same shape; got "
            f"{anion_x.shape}, {anion_y.shape}, {anion_z.shape}."
        )
    N = anion_x.shape[-1]
    k = np.arange(N)

    # FFT each sublattice and transpose to canonical (kx, ky, kz).
    # anion_x axes are (j, k, i) = (y, z, x); FFT output axes (ky, kz, kx).
    # anion_y axes are (i, k, j) = (x, z, y); FFT output axes (kx, kz, ky).
    # anion_z axes are (i, j, k) = (x, y, z); already canonical.
    f_x = np.fft.fftn(anion_x).transpose(2, 0, 1)
    f_y = np.fft.fftn(anion_y).transpose(0, 2, 1)
    f_z = np.fft.fftn(anion_z)

    # Half-unit-cell spatial offset of each sublattice along its own axis.
    phase_x = np.exp(-1j * np.pi * k[:, None, None] / N)
    phase_y = np.exp(-1j * np.pi * k[None, :, None] / N)
    phase_z = np.exp(-1j * np.pi * k[None, None, :] / N)

    return (f_x * phase_x + f_y * phase_y + f_z * phase_z) / N ** 3
