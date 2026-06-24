import numpy as np
import pytest
from chainorder import order_params
from chainorder.decompose import SublatticeOccupation
from tests._fixtures import (
    perfect_oof_chain,
    perfect_ofof_chain,
    occupation_from_chain_arrays,
    single_q_111,
    SHAPES,
)


CUBIC_SHAPES: list[tuple[int, int, int]] = [(3, 3, 3), (6, 6, 6)]
# Cubic; used by tests that physically require equal axes.

ICC_SHAPES: list[tuple[int, int, int]] = [(3, 3, 3), (6, 6, 6), (2, 4, 6)]
# Chain-direction (last axis) divisible by 3; lateral shape arbitrary.

ICC_ROT_SHAPES: list[tuple[int, int, int]] = [(3, 3, 3), (6, 6, 6), (3, 4, 6)]
# Nx divisible by 3 (required by rotating_phase analytical identity);
# lateral non-cubic in the (3, 4, 6) case.

OFOF_SHAPES: list[tuple[int, int, int]] = [(3, 3, 4), (6, 6, 6), (2, 4, 6)]
# Nz even; used by period-2 (OFOF) tests along the chain direction.


@pytest.mark.parametrize("shape", ICC_SHAPES)
def test_chain_fft_perfect_oof_peaks_at_period_3(shape):
    """Perfect OOF chain (Nz divisible by 3): |phi| = 1/3 at k = Nz/3."""
    Nx, Ny, Nz = shape
    arr = perfect_oof_chain(shape, phase=2, direction="z")
    fft = order_params.chain_fft(arr)
    assert fft.shape == (Nx, Ny, Nz)
    k = Nz // 3
    # |phi| at k = Nz/3 should be 1/3 for every chain
    np.testing.assert_allclose(np.abs(fft[..., k]), 1.0 / 3.0, atol=1e-12)


@pytest.mark.parametrize("shape", [s for s in ICC_SHAPES if s[2] > 3])
def test_chain_fft_perfect_oof_other_components_zero(shape):
    """Non-period-3 Fourier components should be zero (except DC).

    Nz=3 is excluded: the OFOF index Nz//2 coincides with the OOF peak
    at Nz//3, so the pattern has no distinct OFOF bin to probe.
    """
    Nx, Ny, Nz = shape
    arr = perfect_oof_chain(shape, phase=2, direction="z")
    fft = order_params.chain_fft(arr)
    # DC component (k=0) is the mean = 1/3
    np.testing.assert_allclose(np.abs(fft[..., 0]), 1.0 / 3.0, atol=1e-12)
    # k=Nz/3 is the OOF component (non-zero)
    # k=Nz/2 (OFOF) should be zero for an OOF pattern
    np.testing.assert_allclose(fft[..., Nz // 2], 0, atol=1e-12)


@pytest.mark.parametrize("shape", OFOF_SHAPES)
def test_chain_fft_perfect_ofof_peaks_at_period_2(shape):
    """Perfect OFOF chain (Nz even): |phi| peaks at k = Nz/2."""
    Nx, Ny, Nz = shape
    arr = perfect_ofof_chain(shape, direction="z")
    fft = order_params.chain_fft(arr)
    k = Nz // 2
    # Mean is 1/2 so DC = 1/2. At k=Nz/2, pattern [0,1,0,1,...] gives 1/2.
    np.testing.assert_allclose(np.abs(fft[..., k]), 0.5, atol=1e-12)


def test_chain_fft_normalisation_period_4_in_N_12():
    """One F per period-4 at N=12: |phi_{N/4}| = 1/4."""
    N = 12
    p = 4
    arr = np.zeros((N, N, N), dtype=int)
    # F at i in {3, 7, 11}: one per period p=4.
    for i in range(N):
        if i % p == p - 1:
            arr[:, :, i] = 1
    fft = order_params.chain_fft(arr)
    np.testing.assert_allclose(np.abs(fft[..., 0]), 1.0 / p, atol=1e-12)
    np.testing.assert_allclose(np.abs(fft[..., N // p]), 1.0 / p, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_chain_fft_is_hermitian_for_real_input(shape):
    """|F_k| == |F_{Nz-k}| for real input. Catches any regression in the
    FFT axis or direction."""
    Nx, Ny, Nz = shape
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 2, size=(Nx, Ny, Nz))
    fft = order_params.chain_fft(arr)
    for k in range(1, Nz):
        np.testing.assert_allclose(
            np.abs(fft[..., k]), np.abs(fft[..., Nz - k]), atol=1e-12,
            err_msg=f"Hermitian symmetry broken at k={k}",
        )


def test_chain_fft_normalisation_period_3_in_N_9():
    """One F per period-3 at N=9: |phi_{N/3}| = 1/3."""
    N = 9
    p = 3
    arr = perfect_oof_chain(N, phase=2)
    fft = order_params.chain_fft(arr)
    np.testing.assert_allclose(np.abs(fft[..., 0]), 1.0 / p, atol=1e-12)
    np.testing.assert_allclose(np.abs(fft[..., N // p]), 1.0 / p, atol=1e-12)


@pytest.mark.parametrize("shape", ICC_SHAPES)
def test_motif_frequencies_perfect_oof_window_3(shape):
    """Perfect OOF chain of length Nz (Nz divisible by 3): the three rotation-
    distinct length-3 windows (0,0,1), (0,1,0), (1,0,0) each appear at
    frequency 1/3."""
    Nx, Ny, Nz = shape
    arr = perfect_oof_chain(shape, phase=2, direction="z")
    freqs = order_params.motif_frequencies(arr, window_length=3)
    expected = np.full((Nx, Ny), 1.0 / 3.0)
    for motif in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        assert motif in freqs, f"Expected motif {motif} in frequencies"
        np.testing.assert_allclose(freqs[motif], expected, atol=1e-12)
    assert set(freqs) == {(0, 0, 1), (0, 1, 0), (1, 0, 0)}


@pytest.mark.parametrize("shape", OFOF_SHAPES)
def test_motif_frequencies_perfect_ofof_window_2(shape):
    """Perfect OFOF chain of length Nz (Nz even): length-2 windows alternate
    between (0, 1) and (1, 0), each at frequency 1/2."""
    Nx, Ny, Nz = shape
    arr = perfect_ofof_chain(shape, direction="z")
    freqs = order_params.motif_frequencies(arr, window_length=2)
    expected = np.full((Nx, Ny), 0.5)
    assert set(freqs) == {(0, 1), (1, 0)}
    np.testing.assert_allclose(freqs[(0, 1)], expected, atol=1e-12)
    np.testing.assert_allclose(freqs[(1, 0)], expected, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_motif_frequencies_all_zero_chain(shape):
    """All-O chain: all length-3 windows are (0, 0, 0), frequency 1."""
    Nx, Ny, Nz = shape
    arr = np.zeros((Nx, Ny, Nz), dtype=int)
    freqs = order_params.motif_frequencies(arr, window_length=3)
    assert (0, 0, 0) in freqs
    np.testing.assert_allclose(freqs[(0, 0, 0)], np.ones((Nx, Ny)), atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_motif_frequencies_sum_to_one(shape):
    """Per-chain frequencies sum to 1 regardless of input."""
    Nx, Ny, Nz = shape
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 2, size=(Nx, Ny, Nz))
    freqs = order_params.motif_frequencies(arr, window_length=3)
    total = sum(freqs.values())
    np.testing.assert_allclose(total, np.ones((Nx, Ny)), atol=1e-12)


def test_motif_frequencies_invariant_under_chain_rotation():
    """Three chains that are cyclic rotations of each other share identical
    motif frequencies: per-chain, each rotation-distinct length-3 motif at 1/3."""
    N = 3
    arr = np.zeros((N, N, N), dtype=int)
    arr[0, 0, 0] = 1   # chain (0, 0): [1, 0, 0]
    arr[0, 1, 1] = 1   # chain (0, 1): [0, 1, 0]
    arr[0, 2, 2] = 1   # chain (0, 2): [0, 0, 1]
    freqs = order_params.motif_frequencies(arr, window_length=3)
    for motif in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        assert motif in freqs
        for (j, k) in [(0, 0), (0, 1), (0, 2)]:
            np.testing.assert_allclose(freqs[motif][j, k], 1.0 / 3.0, atol=1e-12)


@pytest.mark.parametrize("shape", ICC_SHAPES)
def test_motif_frequencies_window_length_1_is_single_site(shape):
    """window_length=1 gives single-site frequencies matching species fraction."""
    Nx, Ny, Nz = shape
    arr = perfect_oof_chain(shape, phase=2, direction="z")   # 1/3 of sites are F
    freqs = order_params.motif_frequencies(arr, window_length=1)
    assert set(freqs) == {(0,), (1,)}
    np.testing.assert_allclose(freqs[(0,)], np.full((Nx, Ny), 2.0 / 3.0), atol=1e-12)
    np.testing.assert_allclose(freqs[(1,)], np.full((Nx, Ny), 1.0 / 3.0), atol=1e-12)


@pytest.mark.parametrize("shape", ICC_SHAPES)
def test_motif_frequencies_window_length_equal_N_is_full_chain(shape):
    """window_length == Nz: the Nz sliding windows are the Nz cyclic rotations
    of the chain. For a period-3 chain, three distinct rotations appear, each
    at frequency 1/3."""
    Nx, Ny, Nz = shape
    arr = perfect_oof_chain(shape, phase=2, direction="z")
    freqs = order_params.motif_frequencies(arr, window_length=Nz)
    total = sum(freqs.values())
    np.testing.assert_allclose(total, np.ones((Nx, Ny)), atol=1e-12)
    assert len(freqs) == 3
    for motif_freq in freqs.values():
        np.testing.assert_allclose(motif_freq, np.full((Nx, Ny), 1.0 / 3.0), atol=1e-12)


def test_motif_frequencies_rejects_invalid_window_lengths():
    """Boundary cases must raise explicitly rather than silently alias."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    with pytest.raises(ValueError, match="window_length must be >= 1"):
        order_params.motif_frequencies(arr, window_length=0)
    with pytest.raises(ValueError, match="window_length must be >= 1"):
        order_params.motif_frequencies(arr, window_length=-3)
    with pytest.raises(ValueError, match="exceeds chain length"):
        order_params.motif_frequencies(arr, window_length=N + 1)
    with pytest.raises(TypeError, match="must be an integer"):
        order_params.motif_frequencies(arr, window_length=2.5)     # type: ignore[arg-type]


def test_motif_frequencies_rejects_non_integer_dtype():
    """Float occupation array: reject rather than silently truncate to 0/1."""
    N = 6
    arr = np.full((N, N, N), 0.7, dtype=np.float64)
    with pytest.raises(TypeError, match="integer dtype"):
        order_params.motif_frequencies(arr, window_length=3)


def test_motif_frequencies_per_chain_distinct_for_mixed_patterns():
    """Mixed OOF/OFOF chains: frequencies per (j, k) reflect each chain's pattern."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    # j in [0, 3): OOF (F at i == 2 mod 3)
    for i in range(N):
        if i % 3 == 2:
            arr[:3, :, i] = 1
    # j in [3, 6): OFOF (F at odd i)
    for i in range(N):
        if i % 2 == 1:
            arr[3:, :, i] = 1
    freqs = order_params.motif_frequencies(arr, window_length=3)
    # OOF chain (period 3): windows cycle through (0,0,1), (0,1,0), (1,0,0),
    # each at frequency 1/3. Other motifs absent.
    for motif in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        np.testing.assert_allclose(freqs[motif][:3, :], 1.0 / 3.0, atol=1e-12)
    # OFOF chain (period 2): windows alternate (0, 1, 0) and (1, 0, 1),
    # each at frequency 1/2.
    np.testing.assert_allclose(freqs[(0, 1, 0)][3:, :], 0.5, atol=1e-12)
    np.testing.assert_allclose(freqs[(1, 0, 1)][3:, :], 0.5, atol=1e-12)
    # Motifs from the other chain block must be zero where they don't occur.
    np.testing.assert_allclose(freqs[(0, 0, 1)][3:, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(freqs[(1, 0, 0)][3:, :], 0.0, atol=1e-12)
    # OFOF motif (1, 0, 1) should not appear in the OOF block.
    np.testing.assert_allclose(freqs[(1, 0, 1)][:3, :], 0.0, atol=1e-12)


@pytest.mark.parametrize("shape", ICC_SHAPES)
def test_along_chain_correlation_perfect_oof(shape):
    """Perfect OOF: g(0) = 2/9, g(3) = 2/9 (peaks), g(1) = g(2) = -1/9 (troughs)."""
    Nx, Ny, Nz = shape
    arr = perfect_oof_chain(shape, phase=2, direction="z")
    g = order_params.along_chain_correlation(arr)
    assert g.shape == (Nz,)
    np.testing.assert_allclose(g[0], 2.0 / 9.0, atol=1e-12)
    np.testing.assert_allclose(g[1], -1.0 / 9.0, atol=1e-12)
    np.testing.assert_allclose(g[2], -1.0 / 9.0, atol=1e-12)
    # Period-3 recurrence: g[3] = g[0] etc., but only defined for Nz >= 6.
    if Nz >= 6:
        np.testing.assert_allclose(g[3], 2.0 / 9.0, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_along_chain_correlation_all_zero(shape):
    """All-O: g(r) = 0 for all r (mean is 0, so no oscillation)."""
    Nx, Ny, Nz = shape
    arr = np.zeros((Nx, Ny, Nz), dtype=int)
    g = order_params.along_chain_correlation(arr)
    np.testing.assert_allclose(g, 0, atol=1e-12)


@pytest.mark.parametrize("shape", OFOF_SHAPES)
def test_along_chain_correlation_perfect_ofof(shape):
    """Perfect OFOF (mean 1/2): g(0) = g(2) = 1/4, g(1) = g(3) = -1/4."""
    Nx, Ny, Nz = shape
    arr = perfect_ofof_chain(shape, direction="z")
    g = order_params.along_chain_correlation(arr)
    # <s_i s_{i+0}> = <s^2> = <s> = 1/2 (binary). 1/2 - (1/2)^2 = 1/4.
    np.testing.assert_allclose(g[0], 0.25, atol=1e-12)
    # <s_i s_{i+1}> = 0 (alternating). 0 - 1/4 = -1/4.
    np.testing.assert_allclose(g[1], -0.25, atol=1e-12)
    # Period-2 recurrence: g[2] = g[0], g[3] = g[1]. Only defined for Nz >= 4.
    if Nz >= 4:
        np.testing.assert_allclose(g[2], 0.25, atol=1e-12)
        np.testing.assert_allclose(g[3], -0.25, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_along_chain_correlation_all_one(shape):
    """All-F (saturated): g(r) = 0 for all r (<s> = 1, so <s * s> - <s>^2 = 0)."""
    Nx, Ny, Nz = shape
    arr = np.ones((Nx, Ny, Nz), dtype=int)
    g = order_params.along_chain_correlation(arr)
    np.testing.assert_allclose(g, 0, atol=1e-12)


@pytest.mark.parametrize("shape", ICC_SHAPES)
def test_inter_chain_correlation_all_same_phase(shape):
    """All chains have identical OOF phase -> |G[dj, dk]| = 1 everywhere."""
    Nx, Ny, Nz = shape
    arr = perfect_oof_chain(shape, phase=2, direction="z")
    G = order_params.inter_chain_correlation(arr, period=3)
    assert G.shape == (Nx, Ny)
    np.testing.assert_allclose(np.abs(G), 1.0, atol=1e-12)


@pytest.mark.parametrize("shape", ICC_SHAPES)
def test_inter_chain_correlation_zero_lag_is_one(shape):
    """G[0, 0] = <exp(i * 0)> = 1 always (trivial)."""
    Nx, Ny, Nz = shape
    rng = np.random.default_rng(0)
    # Random OOF-like array (not strictly periodic, but OK for G[0,0] test)
    arr = rng.integers(0, 2, size=(Nx, Ny, Nz))
    G = order_params.inter_chain_correlation(arr, period=3)
    np.testing.assert_allclose(G[0, 0], 1.0, atol=1e-12)


def test_inter_chain_correlation_raises_when_N_not_divisible_by_period():
    """N=4 has no well-defined period-3 Fourier index."""
    N = 4
    arr = np.zeros((N, N, N), dtype=int)
    with pytest.raises(ValueError, match="divisible by period"):
        order_params.inter_chain_correlation(arr, period=3)


@pytest.mark.parametrize("shape", ICC_ROT_SHAPES)
def test_inter_chain_correlation_rotating_phase(shape):
    """Chains with OOF phase = a mod 3 give G[da, db] = exp(2j*pi*da/3).

    Analytical ground truth, independent of db. This catches sign flips,
    missing `conj`, and lateral-axis confusion that the uniform-phase test
    cannot distinguish from correct.
    """
    Nx, Ny, Nz = shape
    arr = np.zeros((Nx, Ny, Nz), dtype=int)
    for a in range(Nx):
        phase = a % 3
        for i in range(Nz):
            if i % 3 == phase:
                arr[a, :, i] = 1
    G = order_params.inter_chain_correlation(arr, period=3)
    for da in range(Nx):
        expected = np.exp(2j * np.pi * da / 3)
        np.testing.assert_allclose(
            G[da, :], np.full(Ny, expected), atol=1e-12,
            err_msg=f"G[{da}, :] mismatch for shape={shape}",
        )


@pytest.mark.parametrize("shape", ICC_SHAPES)
def test_inter_chain_correlation_random_input_small_off_peak(shape):
    """Random binary input: |G| ~ 0 off the origin (within statistical noise)."""
    Nx, Ny, Nz = shape
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 2, size=(Nx, Ny, Nz))
    G = order_params.inter_chain_correlation(arr, period=3)
    # G[0, 0] is exactly 1 by construction.
    np.testing.assert_allclose(G[0, 0], 1.0, atol=1e-12)
    # Off-origin: expected scale ~ 1/N for N-by-N independent complex samples,
    # so mean |G| should be well below 1. Use a generous bound that still
    # detects any gross formulation error.
    off_peak_mask = np.ones((Nx, Ny), dtype=bool)
    off_peak_mask[0, 0] = False
    assert np.abs(G[off_peak_mask]).mean() < 0.3


def test_inter_chain_correlation_lateral_shape_not_divisible_by_3():
    """Lateral plane can be any shape; only the chain-direction length
    must be divisible by 3."""
    # Chain-direction (last axis) = 6 (divisible by 3). Lateral plane is
    # (2, 4) -- neither divisible by 3.
    arr = np.zeros((2, 4, 6), dtype=int)
    for i in range(6):
        if i % 3 == 2:
            arr[:, :, i] = 1
    G = order_params.inter_chain_correlation(arr, period=3)
    assert G.shape == (2, 4)
    # All chains have identical OOF phase -> |G| = 1 everywhere.
    np.testing.assert_allclose(np.abs(G), 1.0, atol=1e-12)


def test_inter_chain_correlation_nan_on_zero_amplitude():
    """All-O (or all-F) input has |phi| = 0 everywhere; returns NaN."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    G = order_params.inter_chain_correlation(arr, period=3)
    assert G.shape == (N, N)
    assert G.dtype == np.complex128
    assert np.all(np.isnan(G))
    arr_full = np.ones((N, N, N), dtype=int)
    G_full = order_params.inter_chain_correlation(arr_full, period=3)
    assert G_full.shape == (N, N)
    assert G_full.dtype == np.complex128
    assert np.all(np.isnan(G_full))


def test_inter_chain_correlation_nan_on_period2_tile_at_period3():
    """Period-2 tile has zero Fourier weight at k=N/3; returns NaN at period=3."""
    arr = np.tile(np.array([1, 0]), (6, 6, 3))
    G = order_params.inter_chain_correlation(arr, period=3)
    assert G.shape == (6, 6)
    assert G.dtype == np.complex128
    assert np.all(np.isnan(G))


def test_inter_chain_correlation_accepts_arbitrary_period():
    """`period` targets any harmonic, not just period-3. A period-2 alternating
    chain with in-phase chains gives |G| = 1 everywhere when analysed at
    period=2."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    # Period-2 along the chain axis: F at every odd site, on every chain.
    for i in range(N):
        if i % 2 == 1:
            arr[:, :, i] = 1
    G = order_params.inter_chain_correlation(arr, period=2)
    assert G.shape == (N, N)
    np.testing.assert_allclose(np.abs(G), 1.0, atol=1e-12)


def test_inter_chain_correlation_rejects_invalid_period():
    """Non-positive or non-integer `period` raises ValueError."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    for bad_period in (0, -1, 2.5):
        with pytest.raises(ValueError, match="period must be a positive integer"):
            order_params.inter_chain_correlation(arr, period=bad_period)     # type: ignore[arg-type]


def test_structure_factor_takes_sublattice_occupation():
    """`structure_factor` accepts a single `SublatticeOccupation` and
    produces the same numerical output as the pre-refactor three-arg
    form would have on the same per-sublattice content."""
    Nx, Ny, Nz = 6, 4, 3
    shape = (Nx, Ny, Nz)
    ax = perfect_oof_chain(shape, phase=2, direction="x")
    occ = occupation_from_chain_arrays(shape, x=ax)

    F = order_params.structure_factor(occ)

    assert F.shape == (Nx, Ny, Nz)
    np.testing.assert_allclose(np.abs(F[0, 0, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[Nx // 3, 0, 0]), 1.0 / 3.0, atol=1e-12)


@pytest.mark.parametrize("shape", ICC_ROT_SHAPES)
def test_structure_factor_x_only_peaks_on_kx_axis(shape):
    """x-chains OOF (Nx divisible by 3), y/z empty: peaks on kx axis."""
    Nx, Ny, Nz = shape
    ax = perfect_oof_chain(shape, phase=2, direction="x")
    occ = occupation_from_chain_arrays(shape, x=ax)

    F = order_params.structure_factor(occ)

    assert F.shape == (Nx, Ny, Nz)
    np.testing.assert_allclose(np.abs(F[0, 0, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[Nx // 3, 0, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[2 * Nx // 3, 0, 0]), 1.0 / 3.0, atol=1e-12)
    # No period-3 peak on ky or kz axes.
    if Ny % 3 == 0:
        np.testing.assert_allclose(F[0, Ny // 3, 0], 0, atol=1e-12)
    if Nz % 3 == 0:
        np.testing.assert_allclose(F[0, 0, Nz // 3], 0, atol=1e-12)


def test_structure_factor_orthorhombic_x_only_normalisation():
    """Nx=6, Ny=4, Nz=3: Nx-only OOF gives correct per-axis phase and
    normalisation. Catches the /(N**3) vs /(Nx*Ny*Nz) regression."""
    Nx, Ny, Nz = 6, 4, 3
    shape = (Nx, Ny, Nz)
    ax = perfect_oof_chain(shape, phase=2, direction="x")
    occ = occupation_from_chain_arrays(shape, x=ax)

    F = order_params.structure_factor(occ)
    assert F.shape == (Nx, Ny, Nz)
    np.testing.assert_allclose(np.abs(F[Nx // 3, 0, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[0, 0, 0]), 1.0 / 3.0, atol=1e-12)


@pytest.mark.parametrize("shape", CUBIC_SHAPES)
def test_structure_factor_rotation_equivariance_about_z(shape):
    """Rotating the structure 90 deg about z moves peaks from kx to ky (Nx == Ny).

    Parametrised over CUBIC_SHAPES: every cubic shape in the test suite
    satisfies the two preconditions (Nx == Ny and Nx divisible by 3)
    without introducing a new shape list. The rotation property itself
    only requires Nx == Ny.
    """
    Nx, Ny, Nz = shape
    assert Nx == Ny, f"CUBIC_SHAPES must have Nx == Ny, got {shape}"
    ordered_x = perfect_oof_chain(shape, phase=2, direction="x")
    ordered_y = perfect_oof_chain(shape, phase=2, direction="y")

    # "Unrotated": x-chains carry the OOF pattern.
    F_unrot = order_params.structure_factor(
        occupation_from_chain_arrays(shape, x=ordered_x)
    )
    # "Rotated 90 deg about z": what were x-chains are now y-chains, so the
    # same occupation pattern (same phase) appears on y-chains.
    F_rot = order_params.structure_factor(
        occupation_from_chain_arrays(shape, y=ordered_y)
    )

    # Peaks move from (kx, 0, 0) to (0, ky, 0).
    for k in (Nx // 3, 2 * Nx // 3):
        np.testing.assert_allclose(np.abs(F_rot[0, k, 0]), 1.0 / 3.0, atol=1e-12)
        np.testing.assert_allclose(F_rot[k, 0, 0], 0, atol=1e-12)
    # DC is unaffected by rotation.
    np.testing.assert_allclose(F_unrot[0, 0, 0], F_rot[0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("shape", CUBIC_SHAPES)
def test_structure_factor_all_directions_ordered_has_cubic_symmetry(shape):
    """All three sublattices OOF, same phase: peaks on all three Cartesian axes."""
    Nx, Ny, Nz = shape
    assert Nx == Ny == Nz, f"CUBIC_SHAPES must be cubic, got {shape}"
    ax = perfect_oof_chain(shape, phase=2, direction="x")
    ay = perfect_oof_chain(shape, phase=2, direction="y")
    az = perfect_oof_chain(shape, phase=2, direction="z")
    occ = occupation_from_chain_arrays(shape, x=ax, y=ay, z=az)
    F = order_params.structure_factor(occ)
    # Period-3 peaks on each Cartesian axis, equal magnitude.
    np.testing.assert_allclose(np.abs(F[Nx // 3, 0, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[0, Nx // 3, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[0, 0, Nx // 3]), 1.0 / 3.0, atol=1e-12)
    # No mixed peaks (e.g. off the Cartesian axes).
    np.testing.assert_allclose(F[Nx // 3, Nx // 3, 0], 0, atol=1e-12)
    np.testing.assert_allclose(F[Nx // 3, 0, Nx // 3], 0, atol=1e-12)
    # DC = 1 (three sublattices each contributing 1/3).
    np.testing.assert_allclose(np.abs(F[0, 0, 0]), 1.0, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("sublattice", ["x", "y", "z"])
def test_structure_factor_single_anion_carries_sublattice_phase(shape, sublattice):
    """A single anion on sublattice `s` at the origin pins the phase factor
    exp(-i*pi*k_s/N_s) / (Nx*Ny*Nz), checking real AND imaginary parts."""
    Nx, Ny, Nz = shape
    ax = np.zeros((Ny, Nz, Nx), dtype=int)
    ay = np.zeros((Nx, Nz, Ny), dtype=int)
    az = np.zeros((Nx, Ny, Nz), dtype=int)

    if sublattice == "x":
        ax[0, 0, 0] = 1
        N_axis = Nx
    elif sublattice == "y":
        ay[0, 0, 0] = 1
        N_axis = Ny
    else:
        az[0, 0, 0] = 1
        N_axis = Nz
    occ = occupation_from_chain_arrays(shape, x=ax, y=ay, z=az)
    F = order_params.structure_factor(occ)

    for k in range(N_axis):
        expected = np.exp(-1j * np.pi * k / N_axis) / (Nx * Ny * Nz)
        if sublattice == "x":
            actual = F[k, 0, 0]
        elif sublattice == "y":
            actual = F[0, k, 0]
        else:
            actual = F[0, 0, k]
        np.testing.assert_allclose(
            actual, expected, atol=1e-12,
            err_msg=f"sublattice={sublattice}, shape={shape}, k={k}",
        )


def test_structure_factor_rejects_malformed_occupation_shape():
    """`.occupation` must be 4-D with leading axis of length 3."""
    # Leading axis wrong length.
    bad_leading = SublatticeOccupation(occupation=np.zeros((2, 4, 4, 4), dtype=int))
    with pytest.raises(ValueError, match="shape \\(3, Nx, Ny, Nz\\)"):
        order_params.structure_factor(bad_leading)

    # Wrong rank (3-D instead of 4-D).
    bad_rank = SublatticeOccupation(occupation=np.zeros((3, 4, 4), dtype=int))
    with pytest.raises(ValueError, match="shape \\(3, Nx, Ny, Nz\\)"):
        order_params.structure_factor(bad_rank)


def test_circulation_invariants_rejects_non_cubic():
    """The <111> 3-fold exists only for a cubic cell; non-cubic must raise."""
    occ = SublatticeOccupation(occupation=np.zeros((3, 2, 3, 4), dtype=int))
    with pytest.raises(ValueError, match="cubic"):
        order_params.circulation_invariants(occ, period=3)


def test_circulation_invariants_rejects_period_below_2():
    """period < 2 (and non-integer) is rejected before it can index off-grid."""
    occ = SublatticeOccupation(occupation=np.zeros((3, 6, 6, 6), dtype=int))
    for bad in (1, 0, -1, 2.5):
        with pytest.raises(ValueError, match="period must be an integer >= 2"):
            order_params.circulation_invariants(occ, period=bad)  # type: ignore[arg-type]


def test_circulation_invariants_rejects_period_not_dividing_N():
    """N=6 has no clean period-4 <111> harmonic."""
    occ = SublatticeOccupation(occupation=np.zeros((3, 6, 6, 6), dtype=int))
    with pytest.raises(ValueError, match="divisible by period"):
        order_params.circulation_invariants(occ, period=4)


def test_circulation_invariants_rejects_malformed_occupation():
    """.occupation must be 4-D with leading axis of length 3."""
    bad_leading = SublatticeOccupation(occupation=np.zeros((2, 4, 4, 4), dtype=int))
    with pytest.raises(ValueError, match=r"shape \(3, Nx, Ny, Nz\)"):
        order_params.circulation_invariants(bad_leading, period=3)
    bad_rank = SublatticeOccupation(occupation=np.zeros((3, 4, 4), dtype=int))
    with pytest.raises(ValueError, match=r"shape \(3, Nx, Ny, Nz\)"):
        order_params.circulation_invariants(bad_rank, period=3)


@pytest.mark.parametrize("shape", CUBIC_SHAPES)
def test_circulation_invariants_flips_under_reflection(shape):
    """The single-q <111> helix gives chirality = coherence = 1/4 at every N;
    its sense=-1 mirror, the opposite-handed helix, gives -1/4 and 1/4.
    """
    N = shape[0]
    ref = order_params.circulation_invariants(
        single_q_111(N, period=3, sense=1), period=3
    )
    mirror = order_params.circulation_invariants(
        single_q_111(N, period=3, sense=-1), period=3
    )
    np.testing.assert_allclose(ref.chirality, 0.25, atol=1e-10)
    np.testing.assert_allclose(ref.coherence, 0.25, atol=1e-10)
    np.testing.assert_allclose(mirror.chirality, -0.25, atol=1e-10)
    np.testing.assert_allclose(mirror.coherence, 0.25, atol=1e-10)


@pytest.mark.parametrize("shape", CUBIC_SHAPES)
def test_circulation_invariants_invariant_under_physical_ops(shape):
    """The parameter transforms as a pseudoscalar and a scalar under the 48
    physical cubic operations: chirality flips with the operation's determinant
    and coherence is invariant. The input is moved with the independent
    geometric action, not the production operation, so a wrong offset convention
    would break the relation rather than be hidden by the projector's own group
    closure. Asserted relative to the unmoved helix, so it pins the
    transformation contract, not the 1/4 normalisation.
    """
    N = shape[0]
    occ = single_q_111(N, period=3, sense=1)
    ref = order_params.circulation_invariants(occ, period=3)
    for perm, signs, det in order_params.CUBIC_OPS:
        moved = SublatticeOccupation(
            occupation=_apply_cubic_op_geometric(occ.occupation, perm, signs)
        )
        out = order_params.circulation_invariants(moved, period=3)
        np.testing.assert_allclose(
            out.coherence, ref.coherence, atol=1e-10,
            err_msg=f"coherence not invariant under det={det} perm={perm} signs={signs}",
        )
        np.testing.assert_allclose(
            out.chirality, det * ref.chirality, atol=1e-10,
            err_msg=f"chirality wrong under det={det} perm={perm} signs={signs}",
        )


def test_circulation_invariants_random_is_achiral():
    """Random occupancy is achiral: chirality ~ 0 at N=6, coherence
    non-negative. The tolerance is loose to absorb finite-size fluctuation,
    which at N=3 can exceed it."""
    N = 6
    rng = np.random.default_rng(0)
    occ = SublatticeOccupation(occupation=rng.integers(0, 2, size=(3, N, N, N)))
    out = order_params.circulation_invariants(occ, period=3)
    assert abs(out.chirality) < 0.05
    assert out.coherence >= 0.0


def test_circulation_invariants_centrosymmetric_is_achiral():
    """A structure invariant under the physical inversion has chirality exactly
    0: inversion is improper, so chirality(occ) = -chirality(occ). The
    centrosymmetric structure is built with the independent geometric inversion,
    not the production operation, so its centrosymmetry does not depend on the
    code under test.
    """
    N = 6
    rng = np.random.default_rng(1)
    half = rng.integers(0, 2, size=(3, N, N, N))
    centro = np.maximum(half, _apply_cubic_op_geometric(half, (0, 1, 2), (-1, -1, -1)))
    out = order_params.circulation_invariants(
        SublatticeOccupation(occupation=centro), period=3
    )
    np.testing.assert_allclose(out.chirality, 0.0, atol=1e-10)


def test_circulation_invariants_period_2_is_achiral_at_zone_boundary():
    """period = 2: the <111> wavevector is its own negative, so |E+| == |E-|
    and chirality is identically 0 for any input."""
    N = 6
    rng = np.random.default_rng(2)
    occ = SublatticeOccupation(occupation=rng.integers(0, 2, size=(3, N, N, N)))
    out = order_params.circulation_invariants(occ, period=2)
    np.testing.assert_allclose(out.chirality, 0.0, atol=1e-10)


def test_circulation_invariants_unequal_amplitudes():
    """Chirality at an asymmetric point: two sublattices carry the full helix
    and the third is empty, so the single-arm amplitudes are a = 1/3,
    b = (1/3) W^2, c = 0, giving q_chi = 1/3 and q_coh = 5/9; the projection
    scales a (1,1,1)-confined state by 1/4, so chirality = 1/12 and
    coherence = 5/36.
    """
    N = 6
    occ = single_q_111(N, period=3, sense=1).occupation.copy()
    occ[2] = 0
    out = order_params.circulation_invariants(
        SublatticeOccupation(occupation=occ), period=3
    )
    np.testing.assert_allclose(out.chirality, 1.0 / 12.0, atol=1e-10)
    np.testing.assert_allclose(out.coherence, 5.0 / 36.0, atol=1e-10)


def test_arm_ramp_matches_explicit_grid():
    """_arm_ramp(N, period) is exp(-2j pi (x + y + z) / period) on the grid."""
    N, period = 6, 3
    ramp = order_params._arm_ramp(N, period)
    expected = np.empty((N, N, N), dtype=complex)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                expected[x, y, z] = np.exp(-2j * np.pi * (x + y + z) / period)
    np.testing.assert_allclose(ramp, expected, atol=1e-12)


def _circulation_invariants_fft(occupation, *, period):
    """Full-FFT reference for circulation_invariants.

    Differs from production only in the arm read -- a full ``fftn`` bin rather
    than the single-k projection, the one step single-k changed -- so the
    equivalence sweep checks that read against an independent computation.
    """
    sub = occupation.occupation
    N = sub.shape[1]
    idx = (N // period,) * 3
    chirality = 0.0
    coherence = 0.0
    for perm, signs, det in order_params.CUBIC_OPS:
        moved = order_params._apply_cubic_op(sub, perm, signs)
        f = np.fft.fftn(moved.astype(float), axes=(1, 2, 3)) / N**3
        a, b, c = f[0][idx], f[1][idx], f[2][idx]
        e_plus = a + order_params._W * b + order_params._W * order_params._W * c
        e_minus = a + order_params._W * order_params._W * b + order_params._W * c
        chirality += det * (abs(e_plus) ** 2 - abs(e_minus) ** 2)
        coherence += abs(e_plus) ** 2 + abs(e_minus) ** 2
    return order_params.CirculationInvariants(
        chirality / len(order_params.CUBIC_OPS),
        coherence / len(order_params.CUBIC_OPS),
    )


@pytest.mark.parametrize("N, period", [(6, 3), (9, 3), (12, 3), (12, 4)])
def test_circulation_invariants_matches_fft_reference(N, period):
    """Single-k production equals the full-FFT reference within atol=1e-10 on
    random input, at several N and two periods."""
    rng = np.random.default_rng(0)
    occ = SublatticeOccupation(occupation=rng.integers(0, 2, size=(3, N, N, N)))
    single_k = order_params.circulation_invariants(occ, period=period)
    reference = _circulation_invariants_fft(occ, period=period)
    np.testing.assert_allclose(single_k.chirality, reference.chirality, atol=1e-10)
    np.testing.assert_allclose(single_k.coherence, reference.coherence, atol=1e-10)


def _apply_cubic_op_geometric(occ, perm, signs):
    """Independent reference for ``order_params._apply_cubic_op``, derived from
    the anion site geometry rather than transcribing the production logic.

    An anion on sublattice ``s`` in cell ``r`` sits at ``r + (1/2) e_s`` (an edge
    midpoint, half-integer along its own bond axis ``s``). The signed-permutation
    operation ``M`` sends it to ``M (r + (1/2) e_s) = r' + (1/2) e_s'``, where
    ``M e_s = sigma e_s'``; so the image bond axis is ``s'`` and ``r' = M r`` for
    ``sigma = +1`` or ``M r - e_s'`` for ``sigma = -1``, taken mod N. A forward
    scatter, where production is a backward gather -- so agreement across all 48
    ops is a genuine cross-check of the offset convention, not a restatement.
    """
    N = occ.shape[1]
    matrix = np.zeros((3, 3), dtype=int)
    for new_axis, old_axis in enumerate(perm):
        matrix[new_axis, old_axis] = signs[new_axis]
    image_axis = [perm.index(s) for s in range(3)]  # M e_s lies along image_axis[s]
    cells = np.array(list(np.ndindex(N, N, N)))      # (N^3, 3): every cell r
    out = np.zeros_like(occ)
    for s in range(3):
        s_new = image_axis[s]
        moved = cells @ matrix.T                       # M r
        if signs[s_new] == -1:
            moved[:, s_new] -= 1                        # the -e_s' half-cell correction
        moved %= N
        out[s_new, moved[:, 0], moved[:, 1], moved[:, 2]] = occ[
            s, cells[:, 0], cells[:, 1], cells[:, 2]
        ]
    return out


@pytest.mark.parametrize("N", [3, 4, 5])
def test_apply_cubic_op_matches_geometric_oracle(N):
    """The vectorised ``_apply_cubic_op`` equals the independent geometric
    reference for all 48 cubic point operations on random input, pinning the
    offset convention across the whole group.
    """
    rng = np.random.default_rng(0)
    occ = rng.integers(0, 5, size=(3, N, N, N))
    for perm, signs, _det in order_params.CUBIC_OPS:
        np.testing.assert_array_equal(
            order_params._apply_cubic_op(occ, perm, signs),
            _apply_cubic_op_geometric(occ, perm, signs),
            err_msg=f"perm={perm} signs={signs} N={N}",
        )


def test_apply_cubic_op_single_anion_destinations():
    """A single anion under three hand-derived ops, pinning the offset
    convention directly: a reflection maps the moved sublattice's own axis as
    i -> N-1-i and the perpendicular axes as i -> -i mod N. N = 4 so the two
    flavours differ.
    """
    N = 4
    # (sublattice, site, perm, signs, expected (sublattice, i, j, k))
    cases = [
        # Physical inversion: own-axis flip (1 -> 2) and perpendicular flip
        # (1 -> 3). Offset-naive would give (0, 3, 3, 0).
        (0, (1, 1, 0), (0, 1, 2), (-1, -1, -1), (0, 2, 3, 0)),
        # Pure transposition: relabel x-bond -> y-bond plus the axis swap, no
        # flips. The control with no offset subtlety.
        (0, (1, 2, 0), (1, 0, 2), (1, 1, 1), (1, 2, 1, 0)),
        # Relabel with a flip on the new sublattice's own axis: the N-1-i flip
        # follows the relabelled sublattice (src == s_old), not a fixed axis.
        (1, (2, 1, 0), (1, 0, 2), (-1, 1, 1), (0, 2, 2, 0)),
    ]
    for s, site, perm, signs, expected in cases:
        occ = np.zeros((3, N, N, N), dtype=int)
        occ[s][site] = 1
        out = order_params._apply_cubic_op(occ, perm, signs)
        found = tuple(int(x) for x in np.argwhere(out == 1)[0])
        assert found == expected, f"perm={perm} signs={signs}: {found} != {expected}"

