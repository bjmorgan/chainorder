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


def test_decompose_raises_on_off_lattice_atom():
    """Atom at a non-(integer or half-integer) position should raise ValueError."""
    N = 3
    ax_in = perfect_oof_chain(N, phase=2)
    ay_in = perfect_oof_chain(N, phase=2)
    az_in = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax_in, ay_in, az_in)
    # Perturb one atom by 0.2 A - well off-lattice
    atoms.positions[10] += np.array([0.2, 0.0, 0.0])

    with pytest.raises(ValueError, match="not on-lattice"):
        decompose(atoms, N=N)


def test_decompose_raises_on_wrong_cation_count():
    """Structure with too few atoms should raise ValueError."""
    from ase import Atoms as AseAtoms
    # Build a structure with only 4 cations (not 27 = 3^3)
    atoms = AseAtoms(
        symbols=["Nb"] * 4,
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 3.9,
        cell=np.diag([3 * 3.9, 3 * 3.9, 3 * 3.9]),
        pbc=True,
    )
    with pytest.raises(ValueError, match="Wrong cation count"):
        decompose(atoms, N=3)


def test_decompose_raises_on_wrong_anion_count():
    """Structure with correct cation count but wrong anion count should raise.

    Exercises the anion-count check path (the cation check fires first if the
    cation count is also wrong, as in `test_decompose_raises_on_wrong_cation_count`).
    """
    N = 3
    ax_in = perfect_oof_chain(N, phase=2)
    ay_in = perfect_oof_chain(N, phase=2)
    az_in = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax_in, ay_in, az_in)
    # Remove one anion; cation count (N**3 = 27) stays correct, anion count drops.
    del atoms[N ** 3 + 5]
    with pytest.raises(ValueError, match="Wrong anion count"):
        decompose(atoms, N=N)


def test_decompose_rejects_invalid_N():
    """Non-integer or non-positive N raises before any structure checks."""
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    for bad_N in (0, -1, 2.5):
        with pytest.raises(ValueError, match="N must be a positive integer"):
            decompose(atoms, N=bad_N)     # type: ignore[arg-type]


def test_decompose_rejects_origin_out_of_range():
    """origin components outside [0, 1) raise rather than wrapping silently."""
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    for bad_origin in ((1.5, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.0, 1.0, 0.0)):
        with pytest.raises(ValueError, match=r"origin\[\d\] must lie in"):
            decompose(atoms, N=N, origin=bad_origin)


def test_decompose_rejects_origin_wrong_length():
    """origin with not exactly three components raises."""
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    with pytest.raises(ValueError, match="exactly 3 components"):
        decompose(atoms, N=N, origin=(0.0, 0.0))    # type: ignore[arg-type]


def test_decompose_raises_when_species_absent():
    """A species symbol absent from all anion sites raises with a list of present species."""
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    with pytest.raises(ValueError, match="species='Xe' not found"):
        decompose(atoms, N=N, species="Xe")


def test_decompose_raises_on_degenerate_cell():
    """Zero-diagonal cell raises on the finite/positive check."""
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    atoms.set_cell(np.diag([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="positive and finite"):
        decompose(atoms, N=N)


def test_decompose_raises_on_non_cubic_cell():
    """Orthorhombic but non-cubic cell raises (v1 scope)."""
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    new_cell = atoms.cell.array.copy()
    new_cell[0, 0] *= 1.1        # stretch x axis only
    atoms.set_cell(new_cell)
    with pytest.raises(ValueError, match="cubic"):
        decompose(atoms, N=N)


def test_decompose_raises_on_non_orthorhombic_cell():
    """Triclinic cell should raise ValueError (out of scope for v1)."""
    N = 3
    ax_in = perfect_oof_chain(N, phase=2)
    ay_in = perfect_oof_chain(N, phase=2)
    az_in = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax_in, ay_in, az_in)
    # Skew the cell
    new_cell = atoms.cell.array.copy()
    new_cell[0, 1] = 0.5     # introduce off-diagonal element
    atoms.set_cell(new_cell)

    with pytest.raises(ValueError, match="not orthorhombic"):
        decompose(atoms, N=N)


def test_decompose_caches_indices_across_calls(monkeypatch):
    """Calling decompose twice with the same positions/N/origin should only
    run _build_indices once."""
    import importlib
    dm = importlib.import_module("chainorder.decompose")

    call_count = {"n": 0}
    original_build = dm._build_indices

    def counting_build(*args, **kwargs):
        call_count["n"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(dm, "_build_indices", counting_build)
    # Clear any existing cache (from prior tests in the same session)
    dm._indices_cached.cache_clear()

    N = 3
    ax_in = perfect_oof_chain(N, phase=2)
    ay_in = perfect_oof_chain(N, phase=2)
    az_in = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax_in, ay_in, az_in)

    decompose(atoms, N=N)
    decompose(atoms, N=N)
    decompose(atoms, N=N)

    assert call_count["n"] == 1, f"Expected 1 build, got {call_count['n']}"


def test_decompose_rebuilds_when_n_changes(monkeypatch):
    """Different N should cause a rebuild (cache miss)."""
    import importlib
    dm = importlib.import_module("chainorder.decompose")

    call_count = {"n": 0}
    original_build = dm._build_indices

    def counting_build(*args, **kwargs):
        call_count["n"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(dm, "_build_indices", counting_build)
    dm._indices_cached.cache_clear()

    for N in (3, 6):
        ax_in = perfect_oof_chain(N, phase=2)
        ay_in = perfect_oof_chain(N, phase=2)
        az_in = perfect_oof_chain(N, phase=2)
        atoms = build_nbo2f(N, ax_in, ay_in, az_in)
        decompose(atoms, N=N)

    assert call_count["n"] == 2


def test_decompose_cache_hit_with_new_symbols_same_positions(monkeypatch):
    """Two Atoms with identical positions/cell but different F/O patterns:
    second call is a cache hit AND produces the correct per-symbols output."""
    import importlib
    dm = importlib.import_module("chainorder.decompose")

    call_count = {"n": 0}
    original_build = dm._build_indices

    def counting_build(*args, **kwargs):
        call_count["n"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(dm, "_build_indices", counting_build)
    dm._indices_cached.cache_clear()

    N = 3
    ax_first = perfect_oof_chain(N, phase=2)
    ay_first = perfect_oof_chain(N, phase=0)
    az_first = perfect_oof_chain(N, phase=1)
    atoms_first = build_nbo2f(N, ax_first, ay_first, az_first)

    ax_second = perfect_oof_chain(N, phase=0)      # different F/O pattern
    ay_second = perfect_oof_chain(N, phase=1)
    az_second = perfect_oof_chain(N, phase=2)
    atoms_second = build_nbo2f(N, ax_second, ay_second, az_second)
    # Positions and cell are identical (build_nbo2f uses the same lattice
    # geometry regardless of the chain arrays, only the species symbols
    # differ), so the index map should be reusable.

    out_first = decompose(atoms_first, N=N)
    out_second = decompose(atoms_second, N=N)

    # Index map built exactly once.
    assert call_count["n"] == 1, f"Expected 1 build, got {call_count['n']}"

    # Each output reflects its own atoms' species.
    np.testing.assert_array_equal(out_first.x, ax_first)
    np.testing.assert_array_equal(out_first.y, ay_first)
    np.testing.assert_array_equal(out_first.z, az_first)
    np.testing.assert_array_equal(out_second.x, ax_second)
    np.testing.assert_array_equal(out_second.y, ay_second)
    np.testing.assert_array_equal(out_second.z, az_second)
