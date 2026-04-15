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


def test_decompose_with_half_origin():
    """Origin at (0.5, 0.5, 0.5): cation sits at half-integer positions, anions at
    integer + half-integer combinations."""
    N = 3
    ax_in = perfect_oof_chain(N, phase=1)
    ay_in = perfect_oof_chain(N, phase=2)
    az_in = perfect_oof_chain(N, phase=0)
    atoms = build_nbo2f(N, ax_in, ay_in, az_in, origin=(0.5, 0.5, 0.5))

    ax_out, ay_out, az_out = decompose(atoms, N=N, origin=(0.5, 0.5, 0.5))

    np.testing.assert_array_equal(ax_out, ax_in)
    np.testing.assert_array_equal(ay_out, ay_in)
    np.testing.assert_array_equal(az_out, az_in)


def test_decompose_tracks_alternative_species():
    """Default species='F' flags F. Passing species='O' flags O instead (inverted)."""
    N = 3
    ax_in = perfect_oof_chain(N, phase=2)
    ay_in = np.zeros((N, N, N), dtype=int)
    az_in = np.zeros((N, N, N), dtype=int)
    atoms = build_nbo2f(N, ax_in, ay_in, az_in)

    ax_o, ay_o, az_o = decompose(atoms, N=N, species="O")
    # With species="O", the anion arrays flag O as 1 and F as 0, so they should
    # be the complement of the F-flagged arrays.
    np.testing.assert_array_equal(ax_o, 1 - ax_in)
    np.testing.assert_array_equal(ay_o, 1 - ay_in)
    np.testing.assert_array_equal(az_o, 1 - az_in)
