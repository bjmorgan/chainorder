"""Order parameters for ReO3-type anion ordering (chain-level statistics).

All functions here expect the per-direction binary integer arrays
exposed by `chainorder.decompose.SublatticeOccupation` via its `.x`,
`.y`, `.z` chain-layout views, with the last axis along-chain. The
exact shape is direction-specific: ``(Ny, Nz, Nx)`` for x, ``(Nx, Nz,
Ny)`` for y, ``(Nx, Ny, Nz)`` for z. In the cubic case all three reduce
to ``(N, N, N)``.
"""
import itertools
from typing import NamedTuple

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
    # Bit-pack each length-w window into a single integer code: bit k
    # of the code is the species flag at offset k within the window, so
    # each window maps to a code in [0, 2^w).
    window_idx = (np.arange(N)[:, None] + np.arange(w)[None, :]) % N   # (N_chain, w)
    windows = anion_direction[:, :, window_idx]                        # (N_lat0, N_lat1, N_chain, w)
    powers = 1 << np.arange(w, dtype=np.int64)                         # (w,)
    codes = windows.astype(np.int64) @ powers                          # (N_lat0, N_lat1, N_chain)

    frequencies: dict[tuple[int, ...], np.ndarray] = {}
    for c in np.unique(codes):
        mask = codes == c
        bits = tuple((int(c) >> k) & 1 for k in range(w))
        frequencies[bits] = mask.mean(axis=-1)
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
        `da` in `0..N_lat0-1`, `db` in `0..N_lat1-1`. Returns an
        array of NaN if every chain has zero amplitude at the target
        harmonic (the correlation is undefined).

    Raises:
        ValueError: If `period` is not a positive integer, or if
            `N_chain` is not divisible by `period`.
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
        return np.full(phi.shape, np.nan + 0j)

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


_W = np.exp(2j * np.pi / 3)  # omega: primitive cube root of unity


class CubicOp(NamedTuple):
    """One cubic point operation.

    Attributes:
        perm: ``perm[d]`` is the old axis that new axis ``d`` reads from.
        signs: ``signs[d]`` is the sign applied on new axis ``d``.
        det: ``+1`` for a proper operation, ``-1`` for an improper one.
    """

    perm: tuple[int, ...]
    signs: tuple[int, ...]
    det: int


def _make_cubic_ops() -> list[CubicOp]:
    """The 48 cubic point operations: six axis permutations times eight sign
    patterns."""
    ops: list[CubicOp] = []
    for perm in itertools.permutations(range(3)):
        matrix = np.zeros((3, 3), dtype=int)
        for new_axis, old_axis in enumerate(perm):
            matrix[new_axis, old_axis] = 1
        for signs in itertools.product((1, -1), repeat=3):
            det = int(round(float(np.linalg.det(matrix * np.array(signs)))))
            ops.append(CubicOp(perm, signs, det))
    return ops


CUBIC_OPS = _make_cubic_ops()


def _apply_cubic_op(
    occupation: np.ndarray,
    perm: tuple[int, ...],
    signs: tuple[int, ...],
) -> np.ndarray:
    """Offset-aware action of a cubic point operation on the anion sublattices.

    Acts on a ``(3, N, N, N)`` occupancy whose three sublattices carry the x-,
    y- and z-bond anions, each sited at an edge midpoint (half-integer along its
    own bond axis). ``perm[d]`` is the old axis feeding new axis ``d``;
    ``signs[d]`` is the sign on new axis ``d``. A sign flip reflects the grid as
    ``c -> N - 1 - c`` on the moved sublattice's own (half-integer) axis and
    ``c -> -c mod N`` on the other (integer) axes -- the half-cell offset is what
    distinguishes the two flavours.

    Args:
        occupation: ``(3, N, N, N)`` integer occupancy grid.
        perm: Axis permutation; ``perm[d]`` is the old axis feeding new axis ``d``.
        signs: Per-new-axis signs, each ``+1`` or ``-1``.

    Returns:
        The transformed ``(3, N, N, N)`` grid.
    """
    inv = [perm.index(src) for src in range(3)]  # inv[src] = new axis d, perm[d] == src
    out = np.empty_like(occupation)
    for s_new in range(3):
        s_old = perm[s_new]
        slab = occupation[s_old]
        for src in range(3):
            if signs[inv[src]] == -1:
                slab = np.flip(slab, axis=src)
                if src != s_old:
                    slab = np.roll(slab, 1, axis=src)
        out[s_new] = slab.transpose(perm)
    return out


def _raw(occupation: np.ndarray, N: int, period: int) -> tuple[float, float]:
    """Single-arm <111> chirality and coherence from a plain FFT.

    Reads the occupancy's FFT at the symmetric body-diagonal bin
    ``(N/period, N/period, N/period)``, where the three sublattices' half-cell
    phases coincide and cancel. With ``E+`` and ``E-`` the two circular
    combinations of the three sublattice amplitudes, returns
    ``(|E+|^2 - |E-|^2, |E+|^2 + |E-|^2)``.
    """
    f = np.fft.fftn(occupation.astype(float), axes=(1, 2, 3)) / N**3
    idx = (N // period,) * 3
    a, b, c = f[0][idx], f[1][idx], f[2][idx]
    e_plus = a + _W * b + _W * _W * c
    e_minus = a + _W * _W * b + _W * c
    return (
        abs(e_plus) ** 2 - abs(e_minus) ** 2,
        abs(e_plus) ** 2 + abs(e_minus) ** 2,
    )


class CirculationInvariants(NamedTuple):
    """The two <111> circulation invariants (A2 pseudoscalar, A1 scalar).

    Attributes:
        chirality: ``|E+|^2 - |E-|^2`` -- the circulation imbalance
            (a pseudoscalar; its sign is the screw sense). Flips under any
            improper lattice operation, is invariant under proper rotations,
            and is exactly 0 for achiral (centrosymmetric) input and ~0 for
            disordered input.
        coherence: ``|E+|^2 + |E-|^2`` -- the <111> ordering strength
            (``>= 0``). Invariant under the full cubic point group.
    """

    chirality: float
    coherence: float


def circulation_invariants(
    occupation: SublatticeOccupation, *, period: int
) -> CirculationInvariants:
    """Chirality and coherence of the <111> anion ordering.

    On the ReO3 anion sublattice the three edge sublattices (x-, y-, z-bond)
    carry density waves. At the body-diagonal wavevector ``k = (N/period) *
    (1, 1, 1)`` the 3-fold about <111> cyclically permutes them, splitting their
    amplitudes into a symmetric A1 part and a circular E doublet (``E+``, ``E-``).
    The two invariants are ``chirality = |E+|^2 - |E-|^2`` (a pseudoscalar; its
    sign is the screw sense) and ``coherence = |E+|^2 + |E-|^2`` (the <111>
    ordering strength).

    Both are projected onto their representations by averaging over the 48 cubic
    point operations (the Reynolds projector), so chirality is a pseudoscalar and
    coherence a scalar by construction:

        chirality = (1/48) sum_g det(g) q_chi(g . occ)
        coherence = (1/48) sum_g        q_coh(g . occ)

    where ``g . occ`` is the offset-aware action of operation ``g`` on the
    occupancy -- the anions sit at edge midpoints, so a reflection maps a
    sublattice's own axis as ``i -> N - 1 - i`` -- and ``q_chi``, ``q_coh`` are
    the plain-FFT single-arm invariants at the symmetric ``(1, 1, 1)`` bin, where
    the half-cell phases cancel. The perfect single-q <111> helix gives
    ``chirality = coherence = 1/4`` at every N.

    Args:
        occupation: ``SublatticeOccupation`` whose ``.occupation`` has shape
            ``(3, N, N, N)``. Must be cubic.
        period: Target <111> repeat length. Keyword-only, an integer ``>= 2``.
            N must be divisible by ``period``. ``period = 2`` is a degenerate
            zone-boundary case: the wavevector is its own negative and the
            amplitude there is real, so ``chirality`` is identically 0 under the
            projection. The intended use is ``period = 3``.

    Returns:
        ``CirculationInvariants(chirality, coherence)``. Both intensive
        (L-independent).

    Raises:
        ValueError: If ``.occupation`` is not 4-D with leading axis 3; if the
            cell is not cubic (``Nx == Ny == Nz``); or if ``period`` is not an
            integer ``>= 2`` dividing N.
    """
    sub = occupation.occupation
    if sub.ndim != 4 or sub.shape[0] != 3:
        raise ValueError(
            f"SublatticeOccupation.occupation must have shape "
            f"(3, Nx, Ny, Nz); got shape {sub.shape}."
        )
    _, Nx, Ny, Nz = sub.shape
    if not (Nx == Ny == Nz):
        raise ValueError(
            f"circulation_invariants requires a cubic cell (Nx == Ny == Nz); "
            f"got ({Nx}, {Ny}, {Nz}). The <111> 3-fold and the body-diagonal "
            f"wavevector exist only for a cubic supercell."
        )
    N = Nx
    if not isinstance(period, (int, np.integer)) or period < 2:
        raise ValueError(
            f"period must be an integer >= 2, got {period!r}."
        )
    if N % period != 0:
        raise ValueError(
            f"circulation_invariants requires N divisible by period, got "
            f"N={N}, period={period}."
        )
    chirality = 0.0
    coherence = 0.0
    for perm, signs, det in CUBIC_OPS:
        raw_chi, raw_coh = _raw(_apply_cubic_op(sub, perm, signs), N, period)
        chirality += det * raw_chi
        coherence += raw_coh
    return CirculationInvariants(
        float(chirality / len(CUBIC_OPS)),
        float(coherence / len(CUBIC_OPS)),
    )
