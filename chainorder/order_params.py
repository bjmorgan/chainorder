"""Order parameters for ReO3-type anion ordering (chain-level statistics).

All functions here expect the per-direction binary integer arrays
exposed by `chainorder.decompose.SublatticeOccupation` via its `.x`,
`.y`, `.z` chain-layout views, with the last axis along-chain. The
exact shape is direction-specific: ``(Ny, Nz, Nx)`` for x, ``(Nx, Nz,
Ny)`` for y, ``(Nx, Ny, Nz)`` for z. In the cubic case all three reduce
to ``(N, N, N)``.
"""
import numpy as np

from chainorder.decompose import SublatticeOccupation


def chain_fft(anion_direction: np.ndarray) -> np.ndarray:
    """Discrete Fourier transform along each chain.

    Args:
        anion_direction: Binary species array along one chain direction (a
            chain-layout view such as `SublatticeOccupation.x`), shape
            `(N_lat0, N_lat1, N_chain)`. Last axis is along-chain.

    Returns:
        Complex array of the same shape as `anion_direction`. Last axis
        is the Fourier index `k = 0, 1, ..., N_chain - 1`. Computed as
        `np.fft.fft(arr, axis=-1) / N_chain`. For a chain that is strictly
        periodic with period `p` (`N_chain` divisible by `p`, exactly one
        flagged atom per period), the DC component and the peak at
        `k = N_chain // p` each have magnitude `1 / p`.
    """
    N = anion_direction.shape[-1]
    return np.fft.fft(anion_direction, axis=-1) / N


def motif_frequencies(
    anion_direction: np.ndarray,
    *,
    window_length: int,
) -> dict[tuple[int, ...], np.ndarray]:
    """Per-chain frequency of each length-`window_length` motif.

    Slides a window of length `window_length` along each chain (with
    periodic wrap) and returns the fraction of windows matching each
    distinct bit pattern, per chain. Every chain position is the start
    of exactly one window, so the returned frequencies per chain sum to
    `1` regardless of `window_length`.

    Each motif is keyed by its bit tuple, e.g. `(0, 1, 0)` for `OFO`.

    Args:
        anion_direction: Binary species array for a single chain direction,
            shape `(N_lat0, N_lat1, N_chain)`. Last axis is along-chain.
        window_length: Length of the sliding window. Must be between 1
            and `N_chain` inclusive.

    Returns:
        Dictionary mapping each distinct motif tuple that appears in the
        input to a float array of shape `(N_lat0, N_lat1)` giving the
        per-chain frequency (value in `[0, 1]`). Motif tuples not
        present in the input are absent from the dictionary.

    Raises:
        TypeError: If `window_length` is not an integer, or if
            `anion_direction` has a non-integer dtype.
        ValueError: If `window_length < 1` or `window_length > N_chain`.
    """
    if not isinstance(window_length, (int, np.integer)):
        raise TypeError(
            f"window_length must be an integer, got "
            f"{type(window_length).__name__}."
        )
    if not np.issubdtype(anion_direction.dtype, np.integer):
        raise TypeError(
            f"anion_direction must have integer dtype for motif counting; "
            f"got {anion_direction.dtype}. The bit-packed encoding truncates "
            f"floats silently and would give wrong answers."
        )
    N = anion_direction.shape[-1]
    w = int(window_length)
    if w < 1:
        raise ValueError(f"window_length must be >= 1, got {w}.")
    if w > N:
        raise ValueError(
            f"window_length ({w}) exceeds chain length N={N}; windows would "
            f"wrap around the chain and alias. Use window_length <= N."
        )
    # Build (N_chain, w) index array: windows[start, offset] -> position in chain.
    window_idx = (np.arange(N)[:, None] + np.arange(w)[None, :]) % N   # (N_chain, w)

    # windows[j, k, start, offset] = anion_direction[j, k, (start + offset) % N_chain]
    windows = anion_direction[:, :, window_idx]                        # (N_lat0, N_lat1, N_chain, w)

    # Find unique length-w patterns and their index per window position.
    N_lat0, N_lat1 = windows.shape[:2]
    flat = windows.reshape(-1, w)                                      # (N_lat0 * N_lat1 * N, w)
    unique_rows, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = inverse.reshape(N_lat0, N_lat1, N)

    # Per-chain frequency of each pattern.
    frequencies: dict[tuple[int, ...], np.ndarray] = {}
    for idx, row in enumerate(unique_rows):
        bits = tuple(int(x) for x in row)
        frequencies[bits] = (inverse == idx).mean(axis=-1)
    return frequencies


def along_chain_correlation(anion_direction: np.ndarray) -> np.ndarray:
    """Pair correlation g(r) along chains, averaged over all chains of one direction.

    `g(r) = <s_i * s_{i + r}> - <s>^2`. Both averages are grand averages
    over all chain positions `i` and all chains `(j, k)` of the given
    direction; they are NOT per-chain means. Subtracting `<s>^2` removes
    the mean-density contribution so `g(r)` oscillates around zero.
    Wrap-around is periodic.

    Args:
        anion_direction: Binary species array for a single chain direction,
            shape `(N_lat0, N_lat1, N_chain)`. Last axis is along-chain.

    Returns:
        g(r) for `r = 0, 1, ..., N_chain - 1`, shape `(N_chain,)`.
    """
    N = anion_direction.shape[-1]
    s_mean = float(anion_direction.mean())

    # Wiener-Khinchin: g(r) = IFFT(|FFT(s)|^2) / N_chain, averaged over chains,
    # minus the mean-density contribution. O(N^3 log N) instead of the O(N^4)
    # broadcast-indexed stack of r-shifted copies.
    fft_per_chain = np.fft.fft(anion_direction, axis=-1)               # (N_lat0, N_lat1, N_chain)
    power = np.abs(fft_per_chain) ** 2                                 # (N_lat0, N_lat1, N_chain)
    autocorr = np.fft.ifft(power, axis=-1).real / N                    # (N_lat0, N_lat1, N_chain)
    g = autocorr.mean(axis=(0, 1)) - s_mean ** 2                       # (N_chain,)
    return g


def inter_chain_correlation(
    anion_direction: np.ndarray,
    *,
    period: int,
) -> np.ndarray:
    """Inter-chain correlation of a single Fourier component.

    For each chain at lateral position `(a, b)`, let `phi(a, b)` be the
    Fourier coefficient at `k = N_chain / period` (the amplitude and
    phase of period-`period` ordering on that chain). This function
    returns the spatial autocorrelation of `phi` across the chain
    plane, normalised so that `G[0, 0] = 1`:

        G[da, db] = < phi(a, b) * conj(phi(a + da, b + db)) > / < |phi|^2 >

    where the averages run over all `(a, b)` pairs (with periodic wrap in
    `a, b`). Amplitude-weighted: chains with small `|phi|` (disordered)
    contribute proportionally less, so disordered input gives `|G| ~ 0`
    off the origin by construction -- no threshold or guard required.

    `|G|` ranges from `~0` (uncorrelated) to `1` (fully phase-locked);
    `arg(G)` encodes the phase pattern (`0` for uniform alignment, a
    linear gradient for a shifted arrangement, etc.).

    A phase-only inter-chain correlator (in which every chain is treated
    as equally ordered regardless of its Fourier amplitude) is a *different*
    quantity from the amplitude-weighted form returned here and cannot be
    recovered from `G` alone -- it requires per-chain phases from
    `chain_fft(arr)[..., N_chain // period]`. If you need that variant,
    compute `v = np.exp(1j * np.angle(phi))` yourself and take the
    spatial autocorrelation of `v`.

    Args:
        anion_direction: Binary species array along one chain direction,
            shape `(N_lat0, N_lat1, N_chain)`. Last axis is along-chain.
        period: Target repeat length along the chain. Keyword-only.
            `N_chain` must be divisible by `period`.

    Returns:
        Complex array of shape `(N_lat0, N_lat1)`. `G[da, db]` for
        `da` in `0..N_lat0-1`, `db` in `0..N_lat1-1`.

    Raises:
        ValueError: If `period` is not a positive integer, if
            `N_chain` is not divisible by `period`, or if every chain
            has zero amplitude at the target harmonic so that the
            correlation is undefined.
    """
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(
            f"period must be a positive integer, got {period!r}."
        )
    N = anion_direction.shape[-1]
    if N % period != 0:
        raise ValueError(
            f"inter_chain_correlation requires N_chain divisible by period, "
            f"got N_chain={N}, period={period}."
        )
    phi = chain_fft(anion_direction)[..., N // period]                 # (N_lat0, N_lat1)
    power = float(np.mean(np.abs(phi) ** 2))
    if power == 0.0:
        raise ValueError(
            f"All chains have zero amplitude at period={period}; correlation "
            f"is undefined. Inspect chain_fft(arr)[..., N_chain // period] to "
            f"confirm."
        )

    # Wiener-Khinchin: IFFT2(|FFT2(phi)|^2) / (N_lat0 * N_lat1) is the
    # spatial autocorrelation < phi(a, b) * conj(phi(a - da, b - db)) >.
    # The direct form uses the opposite lag sign, so take the complex
    # conjugate to match < phi(a, b) * conj(phi(a + da, b + db)) >.
    phi_k = np.fft.fft2(phi)                                           # (N_lat0, N_lat1)
    numer = np.conj(np.fft.ifft2(np.abs(phi_k) ** 2)) / phi.size       # (N_lat0, N_lat1)
    return numer / power


def structure_factor(occupation: SublatticeOccupation) -> np.ndarray:
    """Anion-sublattice 3D structure factor in a canonical `(kx, ky, kz)` frame.

    Takes a `SublatticeOccupation` and returns its coherent structure
    factor with unit form factor. The three edge-midpoint sublattices
    are Fourier-transformed together (one 3D FFT of the primary
    `occupation` array along axes 1-3, which are already canonical
    `(kx, ky, kz)` by construction) and summed with the half-cell phase
    offset of each sublattice relative to the cation corner (x-anions at
    `+a/2` along x, y-anions at `+a/2` along y, z-anions at `+a/2` along
    z).

    Rotation-equivariant: a lattice-symmetry rotation of the input
    structure produces the correspondingly rotated output. Anisotropy
    is preserved -- chains ordered along one direction give peaks on
    the matching reciprocal axis.

    Args:
        occupation: `SublatticeOccupation` whose `.occupation` has shape
            `(3, Nx, Ny, Nz)`. Axis 0 indexes the three sublattices
            (x, y, z); axes 1-3 are xyz grid coordinates.

    Returns:
        Complex array of shape `(Nx, Ny, Nz)` with axes `(kx, ky, kz)`.
        `|F|**2` is proportional to the kinematic diffuse scattering
        intensity at wavevector `(kx / Nx, ky / Ny, kz / Nz)` in
        reciprocal lattice units. Normalised so that a fully F-occupied
        anion sublattice gives `F[0, 0, 0] = 3` (three F per unit cell)
        and a single-sublattice period-3 ordering gives `|F| = 1/3` at
        its peak.

    Raises:
        ValueError: If `occupation.occupation` is not a 4-D array with
            leading axis of length 3.

    Notes:
        For per-chain (not cross-chain) analysis use `chain_fft` on the
        appropriate chain-layout view (e.g. `occupation.x`).
    """
    sub = occupation.occupation
    if sub.ndim != 4 or sub.shape[0] != 3:
        raise ValueError(
            f"SublatticeOccupation.occupation must have shape "
            f"(3, Nx, Ny, Nz); got shape {sub.shape}."
        )
    _, Nx, Ny, Nz = sub.shape

    # 3D FFT per sublattice along axes (1, 2, 3). The xyz-coord data is
    # already in canonical (kx, ky, kz) order, so no transpose is needed.
    f = np.fft.fftn(sub, axes=(1, 2, 3))                    # (3, Nx, Ny, Nz)

    # Half-unit-cell spatial offset of each sublattice along its own axis.
    kx = np.arange(Nx)
    ky = np.arange(Ny)
    kz = np.arange(Nz)
    phase_x = np.exp(-1j * np.pi * kx[:, None, None] / Nx)
    phase_y = np.exp(-1j * np.pi * ky[None, :, None] / Ny)
    phase_z = np.exp(-1j * np.pi * kz[None, None, :] / Nz)

    return (f[0] * phase_x + f[1] * phase_y + f[2] * phase_z) / (Nx * Ny * Nz)
