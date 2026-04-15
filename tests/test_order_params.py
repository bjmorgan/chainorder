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
