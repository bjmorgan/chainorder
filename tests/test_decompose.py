import numpy as np
import pytest
from chainorder import decompose
from tests._fixtures import build_nbo2f, perfect_oof_chain


def test_decompose_returns_input_species_arrays():
    """Round-trip: build Atoms from known chain arrays, decompose, recover arrays."""
    N = 3
    ax_in = perfect_oof_chain(N, phase=2)   # all chains OOF at phase 2
    ay_in = perfect_oof_chain(N, phase=0)
    az_in = perfect_oof_chain(N, phase=1)
    atoms = build_nbo2f(N, ax_in, ay_in, az_in)

    ax_out, ay_out, az_out = decompose(atoms, N=N)

    np.testing.assert_array_equal(ax_out, ax_in)
    np.testing.assert_array_equal(ay_out, ay_in)
    np.testing.assert_array_equal(az_out, az_in)
