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
