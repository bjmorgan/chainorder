import numpy as np
import pytest
from chainorder import order_params
from tests._fixtures import perfect_oof_chain, perfect_ofof_chain


def test_chain_fft_perfect_oof_peaks_at_period_3():
    """Perfect OOF chain (N divisible by 3): |phi| = 1/3 at k = N/3."""
    N = 6
    arr = perfect_oof_chain(N, phase=2)
    fft = order_params.chain_fft(arr)
    assert fft.shape == (N, N, N)
    k = N // 3
    # |phi| at k = N/3 should be 1/3 for every chain
    np.testing.assert_allclose(np.abs(fft[..., k]), 1.0 / 3.0, atol=1e-12)


def test_chain_fft_perfect_oof_other_components_zero():
    """Non-period-3 Fourier components should be zero (except DC)."""
    N = 6
    arr = perfect_oof_chain(N, phase=2)
    fft = order_params.chain_fft(arr)
    # DC component (k=0) is the mean = 1/3
    np.testing.assert_allclose(np.abs(fft[..., 0]), 1.0 / 3.0, atol=1e-12)
    # k=N/3 is the OOF component (non-zero)
    # k=N/2 (OFOF) should be zero
    np.testing.assert_allclose(fft[..., N // 2], 0, atol=1e-12)


def test_chain_fft_perfect_ofof_peaks_at_period_2():
    """Perfect OFOF chain (N even): |phi| peaks at k = N/2."""
    N = 6
    arr = perfect_ofof_chain(N)
    fft = order_params.chain_fft(arr)
    k = N // 2
    # Mean is 1/2 so DC = 1/2. At k=N/2, pattern is [0,1,0,1,0,1] so FFT gives 1/2.
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


def test_chain_fft_is_hermitian_for_real_input():
    """|F_k| == |F_{N-k}| for real input. Catches any regression in the
    FFT axis or direction."""
    N = 6
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 2, size=(N, N, N))
    fft = order_params.chain_fft(arr)
    for k in range(1, N):
        np.testing.assert_allclose(
            np.abs(fft[..., k]), np.abs(fft[..., N - k]), atol=1e-12,
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


def test_motif_counts_perfect_oof_window_3():
    """Perfect OOF chain: all length-3 windows are cyclic class (0, 0, 1),
    total N per chain."""
    N = 6
    arr = perfect_oof_chain(N, phase=2)
    counts = order_params.motif_counts(arr, window_length=3)
    # One F in every triplet -> canonical (0, 0, 1)
    assert (0, 0, 1) in counts
    np.testing.assert_array_equal(counts[(0, 0, 1)], np.full((N, N), N))
    # Other classes should be absent or zero
    for cls in [(0, 0, 0), (0, 1, 1), (1, 1, 1)]:
        assert cls not in counts or np.all(counts[cls] == 0)


def test_motif_counts_perfect_ofof_window_2():
    """Perfect OFOF chain: all length-2 windows are (0, 1); total N per chain."""
    N = 6
    arr = perfect_ofof_chain(N)
    counts = order_params.motif_counts(arr, window_length=2)
    assert (0, 1) in counts
    np.testing.assert_array_equal(counts[(0, 1)], np.full((N, N), N))


def test_motif_counts_all_zero_chain():
    """All-O chain: all length-3 windows are (0, 0, 0)."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    counts = order_params.motif_counts(arr, window_length=3)
    assert (0, 0, 0) in counts
    np.testing.assert_array_equal(counts[(0, 0, 0)], np.full((N, N), N))


def test_motif_counts_total_equals_N():
    """Sum of counts across classes equals N per chain, regardless of input."""
    N = 6
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 2, size=(N, N, N))
    counts = order_params.motif_counts(arr, window_length=3)
    total = sum(counts.values())
    np.testing.assert_array_equal(total, np.full((N, N), N))


def test_motif_counts_cyclic_rotations_collapse():
    """Three chains with F at positions 0, 1, 2 all canonicalise to (0, 0, 1)."""
    N = 3
    arr = np.zeros((N, N, N), dtype=int)
    arr[0, 0, 0] = 1   # chain (0, 0): [1, 0, 0]
    arr[0, 1, 1] = 1   # chain (0, 1): [0, 1, 0]
    arr[0, 2, 2] = 1   # chain (0, 2): [0, 0, 1]
    counts = order_params.motif_counts(arr, window_length=3)
    # Each of these chains has one F per period; every window canonicalises
    # to (0, 0, 1). Per-chain count should be N=3 for each of them.
    assert (0, 0, 1) in counts
    assert counts[(0, 0, 1)][0, 0] == N
    assert counts[(0, 0, 1)][0, 1] == N
    assert counts[(0, 0, 1)][0, 2] == N


def test_motif_counts_window_length_1_is_single_site():
    """window_length=1 counts single-site species; (0,) + (1,) sum to N per chain."""
    N = 6
    arr = perfect_oof_chain(N, phase=2)      # 1/3 of sites are F
    counts = order_params.motif_counts(arr, window_length=1)
    # (0,) is 2N/3 per chain, (1,) is N/3 per chain; they cover all N windows.
    assert set(counts) == {(0,), (1,)}
    np.testing.assert_array_equal(counts[(0,)], np.full((N, N), 2 * N // 3))
    np.testing.assert_array_equal(counts[(1,)], np.full((N, N), N // 3))


def test_motif_counts_window_length_equal_N_is_full_chain():
    """window_length == N: every rotation of the chain is in the same class."""
    N = 6
    arr = perfect_oof_chain(N, phase=2)      # chain [0,0,1,0,0,1]
    counts = order_params.motif_counts(arr, window_length=N)
    # All N sliding windows rotate through the same N-tuple cyclically, so
    # every window collapses to the same canonical form; counts sum to N.
    total = sum(counts.values())
    np.testing.assert_array_equal(total, np.full((N, N), N))
    assert len(counts) == 1


def test_motif_counts_rejects_invalid_window_lengths():
    """Boundary cases must raise explicitly rather than silently alias."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    with pytest.raises(ValueError, match="window_length must be >= 1"):
        order_params.motif_counts(arr, window_length=0)
    with pytest.raises(ValueError, match="window_length must be >= 1"):
        order_params.motif_counts(arr, window_length=-3)
    with pytest.raises(ValueError, match="exceeds chain length"):
        order_params.motif_counts(arr, window_length=N + 1)
    with pytest.raises(TypeError, match="must be an integer"):
        order_params.motif_counts(arr, window_length=2.5)     # type: ignore[arg-type]


def test_motif_counts_rejects_non_integer_dtype():
    """Float occupation array: reject rather than silently truncate to 0/1."""
    N = 6
    arr = np.full((N, N, N), 0.7, dtype=np.float64)
    with pytest.raises(TypeError, match="integer dtype"):
        order_params.motif_counts(arr, window_length=3)


def test_motif_counts_per_chain_distinct_for_mixed_patterns():
    """Mixed OOF/OFOF chains: counts per (j, k) reflect each chain's pattern."""
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
    counts = order_params.motif_counts(arr, window_length=3)
    # OOF: all N windows are (0, 0, 1); zero of (0, 1, 1).
    # OFOF chain [0,1,0,1,0,1]: windows (0,1,0) and (1,0,1) alternate, giving
    # N/2 of each canonical class (0, 0, 1) and (0, 1, 1).
    np.testing.assert_array_equal(counts[(0, 0, 1)][:3, :], N)
    np.testing.assert_array_equal(counts[(0, 0, 1)][3:, :], N // 2)
    np.testing.assert_array_equal(counts[(0, 1, 1)][:3, :], 0)
    np.testing.assert_array_equal(counts[(0, 1, 1)][3:, :], N // 2)


def test_along_chain_correlation_perfect_oof():
    """Perfect OOF: g(0) = 2/9, g(3) = 2/9 (peaks), g(1) = g(2) = -1/9 (troughs)."""
    N = 6
    arr = perfect_oof_chain(N, phase=2)
    g = order_params.along_chain_correlation(arr)
    assert g.shape == (N,)
    np.testing.assert_allclose(g[0], 2.0 / 9.0, atol=1e-12)
    np.testing.assert_allclose(g[3], 2.0 / 9.0, atol=1e-12)
    np.testing.assert_allclose(g[1], -1.0 / 9.0, atol=1e-12)
    np.testing.assert_allclose(g[2], -1.0 / 9.0, atol=1e-12)


def test_along_chain_correlation_all_zero():
    """All-O: g(r) = 0 for all r (mean is 0, so no oscillation)."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    g = order_params.along_chain_correlation(arr)
    np.testing.assert_allclose(g, 0, atol=1e-12)


def test_along_chain_correlation_perfect_ofof():
    """Perfect OFOF (mean 1/2): g(0) = g(2) = 1/4, g(1) = g(3) = -1/4."""
    N = 6
    arr = perfect_ofof_chain(N)
    g = order_params.along_chain_correlation(arr)
    # <s_i s_{i+0}> = <s^2> = <s> = 1/2 (binary). 1/2 - (1/2)^2 = 1/4.
    np.testing.assert_allclose(g[0], 0.25, atol=1e-12)
    # <s_i s_{i+1}> = 0 (alternating). 0 - 1/4 = -1/4.
    np.testing.assert_allclose(g[1], -0.25, atol=1e-12)
    # <s_i s_{i+2}> = 1/2 (shift by period). 1/2 - 1/4 = 1/4.
    np.testing.assert_allclose(g[2], 0.25, atol=1e-12)
    np.testing.assert_allclose(g[3], -0.25, atol=1e-12)


def test_along_chain_correlation_all_one():
    """All-F (saturated): g(r) = 0 for all r (<s> = 1, so <s * s> - <s>^2 = 0)."""
    N = 6
    arr = np.ones((N, N, N), dtype=int)
    g = order_params.along_chain_correlation(arr)
    np.testing.assert_allclose(g, 0, atol=1e-12)


def test_inter_chain_correlation_all_same_phase():
    """All chains have identical OOF phase -> |G[dj, dk]| = 1 everywhere."""
    N = 6
    arr = perfect_oof_chain(N, phase=2)     # every chain has phase 2
    G = order_params.inter_chain_correlation(arr)
    assert G.shape == (N, N)
    np.testing.assert_allclose(np.abs(G), 1.0, atol=1e-12)


def test_inter_chain_correlation_zero_lag_is_one():
    """G[0, 0] = <exp(i * 0)> = 1 always (trivial)."""
    N = 6
    rng = np.random.default_rng(0)
    # Random OOF-like array (not strictly periodic, but OK for G[0,0] test)
    arr = rng.integers(0, 2, size=(N, N, N))
    G = order_params.inter_chain_correlation(arr)
    np.testing.assert_allclose(G[0, 0], 1.0, atol=1e-12)


def test_inter_chain_correlation_raises_when_N_not_divisible_by_3():
    """N=4 has no well-defined period-3 Fourier index."""
    N = 4
    arr = np.zeros((N, N, N), dtype=int)
    with pytest.raises(ValueError, match="divisible by 3"):
        order_params.inter_chain_correlation(arr)


def test_inter_chain_correlation_rotating_phase():
    """Chains with OOF phase = a mod 3 give G[da, db] = exp(2j*pi*da/3).

    Analytical ground truth, independent of db. This catches sign flips,
    missing `conj`, and lateral-axis confusion that the uniform-phase test
    cannot distinguish from correct.
    """
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    for a in range(N):
        phase = a % 3
        for i in range(N):
            if i % 3 == phase:
                arr[a, :, i] = 1
    G = order_params.inter_chain_correlation(arr)
    for da in range(N):
        expected = np.exp(2j * np.pi * da / 3)
        np.testing.assert_allclose(
            G[da, :], np.full(N, expected), atol=1e-12,
            err_msg=f"G[{da}, :] mismatch",
        )


def test_inter_chain_correlation_random_input_small_off_peak():
    """Random binary input: |G| ~ 0 off the origin (within statistical noise)."""
    N = 12
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 2, size=(N, N, N))
    G = order_params.inter_chain_correlation(arr)
    # G[0, 0] is exactly 1 by construction.
    np.testing.assert_allclose(G[0, 0], 1.0, atol=1e-12)
    # Off-origin: expected scale ~ 1/N for N-by-N independent complex samples,
    # so mean |G| should be well below 1. Use a generous bound that still
    # detects any gross formulation error.
    off_peak_mask = np.ones((N, N), dtype=bool)
    off_peak_mask[0, 0] = False
    assert np.abs(G[off_peak_mask]).mean() < 0.3


def test_inter_chain_correlation_raises_on_non_finite_result():
    """Post-division np.isfinite guard catches NaN/inf in the normalised G.

    Reachable in practice only through a pathological `chain_fft` output
    (the exact-zero power is caught upstream; ordinary chain arrays never
    produce subnormal power in float64). Monkeypatch chain_fft to inject
    NaN into the period-3 slice so the guard fires.
    """
    N = 6

    def fake_chain_fft(arr, _orig=order_params.chain_fft):
        result = _orig(arr).astype(complex)
        result[0, 0, N // 3] = np.nan
        return result

    import chainorder.order_params as op
    original = op.chain_fft
    op.chain_fft = fake_chain_fft
    try:
        arr = np.zeros((N, N, N), dtype=int)
        arr[0, 0, 0] = 1
        with pytest.raises(ValueError, match="non-finite"):
            op.inter_chain_correlation(arr)
    finally:
        op.chain_fft = original


def test_inter_chain_correlation_raises_on_zero_amplitude():
    """All-O (or all-F) input has |phi| = 0 everywhere; correlation undefined."""
    N = 6
    arr = np.zeros((N, N, N), dtype=int)
    with pytest.raises(ValueError, match="undefined"):
        order_params.inter_chain_correlation(arr)
    arr_full = np.ones((N, N, N), dtype=int)
    with pytest.raises(ValueError, match="undefined"):
        order_params.inter_chain_correlation(arr_full)


def test_structure_factor_only_x_chains_ordered_peaks_on_kx_axis():
    """x-chains OOF, y/z empty: peaks only on the kx reciprocal axis."""
    N = 6
    ax = perfect_oof_chain(N, phase=2)
    zero = np.zeros((N, N, N), dtype=int)
    F = order_params.structure_factor(ax, zero, zero)
    assert F.shape == (N, N, N)
    # DC and period-3 harmonics along kx, all at magnitude 1/3.
    np.testing.assert_allclose(np.abs(F[0, 0, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[N // 3, 0, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[2 * N // 3, 0, 0]), 1.0 / 3.0, atol=1e-12)
    # No period-3 peak on the ky or kz axes (anisotropic input; no powder
    # averaging).
    np.testing.assert_allclose(F[0, N // 3, 0], 0, atol=1e-12)
    np.testing.assert_allclose(F[0, 0, N // 3], 0, atol=1e-12)


def test_structure_factor_rotation_equivariance_about_z():
    """Rotating the structure 90 deg about z moves peaks from the kx to the ky axis."""
    N = 6
    ordered = perfect_oof_chain(N, phase=2)
    zero = np.zeros((N, N, N), dtype=int)

    # "Unrotated": the x-chains carry the OOF pattern.
    F_unrot = order_params.structure_factor(ordered, zero, zero)
    # "Rotated 90 deg about z": what were x-chains are now y-chains, so the same
    # occupation pattern appears in the y-chain slot.
    F_rot = order_params.structure_factor(zero, ordered, zero)

    # Peaks move from (kx, 0, 0) to (0, ky, 0).
    for k in (N // 3, 2 * N // 3):
        np.testing.assert_allclose(np.abs(F_rot[0, k, 0]), 1.0 / 3.0, atol=1e-12)
        np.testing.assert_allclose(F_rot[k, 0, 0], 0, atol=1e-12)
    # DC is unaffected by rotation.
    np.testing.assert_allclose(F_unrot[0, 0, 0], F_rot[0, 0, 0], atol=1e-12)


def test_structure_factor_all_directions_ordered_has_cubic_symmetry():
    """All three sublattices OOF, same phase: peaks on all three Cartesian axes."""
    N = 6
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    F = order_params.structure_factor(ax, ay, az)
    # Period-3 peaks on each Cartesian axis, equal magnitude.
    np.testing.assert_allclose(np.abs(F[N // 3, 0, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[0, N // 3, 0]), 1.0 / 3.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(F[0, 0, N // 3]), 1.0 / 3.0, atol=1e-12)
    # No mixed peaks (e.g. off the Cartesian axes).
    np.testing.assert_allclose(F[N // 3, N // 3, 0], 0, atol=1e-12)
    np.testing.assert_allclose(F[N // 3, 0, N // 3], 0, atol=1e-12)
    # DC = 1 (three sublattices each contributing 1/3).
    np.testing.assert_allclose(np.abs(F[0, 0, 0]), 1.0, atol=1e-12)


@pytest.mark.parametrize("sublattice", ["x", "y", "z"])
def test_structure_factor_single_anion_carries_sublattice_phase(sublattice):
    """A single anion on sublattice `s` at the origin pins the phase factor
    exp(-i*pi*k_s/N) / N^3, checking real AND imaginary parts.

    The single-atom FFT is 1 at every frequency; the transpose to canonical
    (kx, ky, kz) leaves it 1; the only non-trivial contribution is the
    sublattice-offset phase exp(-i*pi*k_s/N) along the sublattice's own
    reciprocal axis. Parametrised over all three sublattices so a copy-
    paste bug in phase_y or phase_z is not masked by covering only x.
    """
    N = 6
    zero = np.zeros((N, N, N), dtype=int)
    ax = zero.copy()
    ay = zero.copy()
    az = zero.copy()

    if sublattice == "x":
        ax[0, 0, 0] = 1
    elif sublattice == "y":
        ay[0, 0, 0] = 1
    else:
        az[0, 0, 0] = 1
    F = order_params.structure_factor(ax, ay, az)

    # Pick the axis along which the phase varies, keep the other two at 0.
    for k in range(N):
        expected = np.exp(-1j * np.pi * k / N) / N ** 3
        if sublattice == "x":
            actual = F[k, 0, 0]
        elif sublattice == "y":
            actual = F[0, k, 0]
        else:
            actual = F[0, 0, k]
        np.testing.assert_allclose(
            actual, expected, atol=1e-12,
            err_msg=f"sublattice={sublattice}, k={k}",
        )


def test_structure_factor_raises_on_shape_mismatch():
    """Three individually cubic arrays with different N raise ValueError."""
    arr_n6 = np.zeros((6, 6, 6), dtype=int)
    arr_n4 = np.zeros((4, 4, 4), dtype=int)
    with pytest.raises(ValueError, match="same shape"):
        order_params.structure_factor(arr_n6, arr_n6, arr_n4)


def test_structure_factor_raises_on_non_cubic_shape():
    """Three equal-shaped but non-cubic inputs should still raise."""
    arr = np.zeros((3, 4, 5), dtype=int)
    with pytest.raises(ValueError, match="cubic 3D"):
        order_params.structure_factor(arr, arr, arr)


def test_structure_factor_raises_on_wrong_rank():
    """Non-3D inputs should raise before numpy's own broadcasting error."""
    arr = np.zeros((6, 6), dtype=int)    # 2D
    with pytest.raises(ValueError, match="cubic 3D"):
        order_params.structure_factor(arr, arr, arr)
