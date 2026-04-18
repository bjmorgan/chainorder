import numpy as np
import pytest
from chainorder import SublatticeOccupation
from chainorder.decompose import Direction
from tests._fixtures import (
    build_nbo2f,
    perfect_oof_chain,
    oof_or_zero,
    dummy_chain_arrays,
    SHAPES,
)


def test_sublattice_occupation_exposes_primary_and_chain_views():
    """SublatticeOccupation holds a (3, Nx, Ny, Nz) xyz-coord array as its
    primary form. .x/.y/.z expose the chain-layout transpose views."""
    Nx, Ny, Nz = 2, 3, 4
    data = np.arange(3 * Nx * Ny * Nz).reshape(3, Nx, Ny, Nz)
    occ = SublatticeOccupation(occupation=data)

    # Primary form is the input array by identity.
    assert occ.occupation is data
    assert occ.occupation.shape == (3, Nx, Ny, Nz)

    # Chain-layout views: direction-specific shapes, last axis along-chain.
    assert occ.x.shape == (Ny, Nz, Nx)
    assert occ.y.shape == (Nx, Nz, Ny)
    assert occ.z.shape == (Nx, Ny, Nz)

    # Contents match the documented transposes.
    np.testing.assert_array_equal(occ.x, data[Direction.X].transpose(1, 2, 0))
    np.testing.assert_array_equal(occ.y, data[Direction.Y].transpose(0, 2, 1))
    np.testing.assert_array_equal(occ.z, data[Direction.Z])

    # __iter__ yields (x, y, z) for positional unpacking.
    ax, ay, az = occ
    np.testing.assert_array_equal(ax, occ.x)
    np.testing.assert_array_equal(ay, occ.y)
    np.testing.assert_array_equal(az, occ.z)


def test_sublattice_occupation_from_atoms_is_immutable():
    """The occupation array (and its transpose views) produced by
    `from_atoms` is read-only, so that `@dataclass(frozen=True)`'s
    immutability extends to the underlying buffer."""
    shape = (3, 3, 3)
    ax_in, ay_in, az_in = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)
    occ = SublatticeOccupation.from_atoms(atoms, N=shape, species="O")

    # The primary buffer is read-only: direct mutation raises.
    with pytest.raises(ValueError, match="read-only"):
        occ.occupation[0, 0, 0, 0] = 0

    # Chain-layout views are transpose views of the primary buffer, so
    # they inherit the read-only flag -- mutating .x/.y/.z raises too.
    with pytest.raises(ValueError, match="read-only"):
        occ.x[0, 0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        occ.y[0, 0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        occ.z[0, 0, 0] = 0


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_orthorhombic_round_trip(shape):
    """Round-trip for arbitrary orthorhombic shape."""
    ax_in = oof_or_zero(shape, phase=2, direction="x")
    ay_in = oof_or_zero(shape, phase=0, direction="y")
    az_in = oof_or_zero(shape, phase=1, direction="z")
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)

    out = SublatticeOccupation.from_atoms(atoms, N=shape, species="F")

    np.testing.assert_array_equal(out.x, ax_in)
    np.testing.assert_array_equal(out.y, ay_in)
    np.testing.assert_array_equal(out.z, az_in)


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_returns_input_species_arrays(shape):
    """Round-trip: build Atoms from known chain arrays, decompose, recover arrays."""
    ax_in = oof_or_zero(shape, phase=2, direction="x")
    ay_in = oof_or_zero(shape, phase=0, direction="y")
    az_in = oof_or_zero(shape, phase=1, direction="z")
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)

    ax_out, ay_out, az_out = SublatticeOccupation.from_atoms(atoms, N=shape, species="F")

    np.testing.assert_array_equal(ax_out, ax_in)
    np.testing.assert_array_equal(ay_out, ay_in)
    np.testing.assert_array_equal(az_out, az_in)


@pytest.mark.parametrize("shape", SHAPES)
def test_from_atoms_returns_sublattice_occupation_with_primary_form(shape):
    """`SublatticeOccupation.from_atoms(...)` returns a `SublatticeOccupation`;
    its primary `.occupation` field has shape `(3, Nx, Ny, Nz)` with
    per-layer content consistent with the chain-layout views."""
    Nx, Ny, Nz = shape
    ax_in = oof_or_zero(shape, phase=2, direction="x")
    ay_in = oof_or_zero(shape, phase=0, direction="y")
    az_in = oof_or_zero(shape, phase=1, direction="z")
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)

    result = SublatticeOccupation.from_atoms(atoms, N=shape, species="F")

    assert isinstance(result, SublatticeOccupation)
    assert result.occupation.shape == (3, Nx, Ny, Nz)
    np.testing.assert_array_equal(
        result.occupation[Direction.X].transpose(1, 2, 0), result.x
    )
    np.testing.assert_array_equal(
        result.occupation[Direction.Y].transpose(0, 2, 1), result.y
    )
    np.testing.assert_array_equal(result.occupation[Direction.Z], result.z)
    np.testing.assert_array_equal(result.x, ax_in)
    np.testing.assert_array_equal(result.y, ay_in)
    np.testing.assert_array_equal(result.z, az_in)


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_with_half_origin(shape):
    """Origin at (0.5, 0.5, 0.5): cation sits at half-integer positions, anions at
    integer + half-integer combinations."""
    ax_in = oof_or_zero(shape, phase=1, direction="x")
    ay_in = oof_or_zero(shape, phase=2, direction="y")
    az_in = oof_or_zero(shape, phase=0, direction="z")
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in, origin=(0.5, 0.5, 0.5))

    ax_out, ay_out, az_out = SublatticeOccupation.from_atoms(atoms, N=shape, origin=(0.5, 0.5, 0.5), species="F")

    np.testing.assert_array_equal(ax_out, ax_in)
    np.testing.assert_array_equal(ay_out, ay_in)
    np.testing.assert_array_equal(az_out, az_in)


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_tracks_alternative_species(shape):
    """Passing species='O' flags O sites as 1; the result is the complement
    of the same Atoms decomposed with species='F'."""
    ax_in = oof_or_zero(shape, phase=2, direction="x")
    _, ay_in, az_in = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)

    ax_o, ay_o, az_o = SublatticeOccupation.from_atoms(atoms, N=shape, species="O")
    # With species="O", the anion arrays flag O as 1 and F as 0, so they should
    # be the complement of the F-flagged arrays.
    np.testing.assert_array_equal(ax_o, 1 - ax_in)
    np.testing.assert_array_equal(ay_o, 1 - ay_in)
    np.testing.assert_array_equal(az_o, 1 - az_in)


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_raises_on_off_lattice_atom(shape):
    """Atom at a non-(integer or half-integer) position should raise ValueError."""
    ax_in, ay_in, az_in = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)
    # Perturb one atom by 0.2 A along x; deviation is on the x axis.
    atoms.positions[len(atoms) // 2] += np.array([0.2, 0.0, 0.0])

    # Match on the upgraded diagnostic content (axis letter) so a future
    # refactor can't silently strip the detail.
    with pytest.raises(ValueError, match=r"Atom \d+ is not on-lattice.*axis x"):
        SublatticeOccupation.from_atoms(atoms, N=shape, species="O")


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
        SublatticeOccupation.from_atoms(atoms, N=3, species="F")


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_raises_on_wrong_anion_count(shape):
    """Structure with correct cation count but wrong anion count should raise.

    Exercises the anion-count check path (the cation check fires first if the
    cation count is also wrong, as in `test_decompose_raises_on_wrong_cation_count`).
    """
    Nx, Ny, Nz = shape
    ax_in, ay_in, az_in = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)
    # Remove one anion; cation count (Nx*Ny*Nz) stays correct, anion count drops.
    del atoms[Nx * Ny * Nz + 5]
    with pytest.raises(ValueError, match="Wrong anion count"):
        SublatticeOccupation.from_atoms(atoms, N=shape, species="O")


def test_decompose_rejects_invalid_N():
    """Non-integer or non-positive N raises before any structure checks."""
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    for bad_N in (0, -1, 2.5):
        with pytest.raises(ValueError, match="N must be a positive integer"):
            SublatticeOccupation.from_atoms(atoms, N=bad_N, species="F")     # type: ignore[arg-type]


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_rejects_origin_out_of_range(shape):
    """origin components outside [0, 1) raise rather than wrapping silently."""
    ax, ay, az = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax, ay, az)
    for bad_origin in ((1.5, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.0, 1.0, 0.0)):
        with pytest.raises(ValueError, match=r"origin\[\d\] must lie in"):
            SublatticeOccupation.from_atoms(atoms, N=shape, origin=bad_origin, species="O")


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_raises_when_species_absent(shape):
    """A species symbol absent from all anion sites raises with a list of present species."""
    ax, ay, az = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax, ay, az)
    with pytest.raises(ValueError, match="species='Xe' not found"):
        SublatticeOccupation.from_atoms(atoms, N=shape, species="Xe")


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_raises_on_degenerate_cell(shape):
    """Zero-diagonal cell raises on the finite/positive check."""
    ax, ay, az = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax, ay, az)
    atoms.set_cell(np.diag([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="diagonal must be positive"):
        SublatticeOccupation.from_atoms(atoms, N=shape, species="O")


@pytest.mark.parametrize("row,col", [(0, 1), (0, 2), (1, 2), (2, 0)])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_decompose_raises_on_non_finite_in_cell(row, col, bad_value):
    """NaN or +/-inf anywhere in the cell matrix must raise.

    Parametrised over multiple (row, col) positions (on- and off-diagonal)
    and all three non-finite IEEE values, to guard against a narrowing
    refactor that only checks the diagonal or only checks NaN.
    """
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    new_cell = atoms.cell.array.copy()
    new_cell[row, col] = bad_value
    atoms.set_cell(new_cell)
    with pytest.raises(ValueError, match="finite"):
        SublatticeOccupation.from_atoms(atoms, N=N, species="F")


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_decompose_raises_on_non_finite_in_positions(axis, bad_value):
    """NaN or +/-inf anywhere in a position coordinate must raise.

    Without this check, `.astype(np.int64)` on a NaN-propagated scaled
    coordinate produces platform-dependent garbage that flows into the
    decomposition silently.
    """
    N = 3
    ax = perfect_oof_chain(N, phase=2)
    ay = perfect_oof_chain(N, phase=2)
    az = perfect_oof_chain(N, phase=2)
    atoms = build_nbo2f(N, ax, ay, az)
    # Corrupt one coordinate of one atom.
    atoms.positions[len(atoms) // 2, axis] = bad_value
    with pytest.raises(ValueError, match="Positions must contain only finite"):
        SublatticeOccupation.from_atoms(atoms, N=N, species="F")


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_raises_on_non_orthorhombic_cell(shape):
    """Triclinic cell should raise ValueError (out of scope for v1)."""
    ax_in, ay_in, az_in = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)
    # Skew the cell
    new_cell = atoms.cell.array.copy()
    new_cell[0, 1] = 0.5     # introduce off-diagonal element
    atoms.set_cell(new_cell)

    with pytest.raises(ValueError, match="not orthorhombic"):
        SublatticeOccupation.from_atoms(atoms, N=shape, species="O")


def test_decompose_rejects_permuted_N_with_structural_diagnostic():
    """A permuted N on a legitimately on-lattice orthorhombic structure
    must raise a "shape does not match cell" error, not a misleading
    "not on-lattice" error."""
    shape = (2, 3, 4)
    ax, ay, az = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax, ay, az)        # cell diag == (2a, 3a, 4a)
    # Permute: pass (3, 4, 2) for a (2, 3, 4) cell. Anion count is
    # permutation-invariant (3 * Nx * Ny * Nz), so without the aspect
    # ratio check this would slip past the count check and fail
    # misleadingly at the on-lattice test.
    with pytest.raises(ValueError, match="does not match cell"):
        SublatticeOccupation.from_atoms(atoms, N=(3, 4, 2), species="O")


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_caches_indices_across_calls(monkeypatch, shape):
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
    dm._clear_cache()

    ax_in, ay_in, az_in = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)

    SublatticeOccupation.from_atoms(atoms, N=shape, species="O")
    SublatticeOccupation.from_atoms(atoms, N=shape, species="O")
    SublatticeOccupation.from_atoms(atoms, N=shape, species="O")

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
    dm._clear_cache()

    for shape in [(3, 3, 3), (6, 6, 6), (2, 3, 4), (3, 3, 4)]:
        ax_in, ay_in, az_in = dummy_chain_arrays(shape)
        atoms = build_nbo2f(shape, ax_in, ay_in, az_in)
        SublatticeOccupation.from_atoms(atoms, N=shape, species="O")

    assert call_count["n"] == 4


def test_decompose_rebuilds_when_single_axis_changes(monkeypatch):
    """Shape tuples that differ in just one axis must miss the cache."""
    import importlib
    dm = importlib.import_module("chainorder.decompose")

    call_count = {"n": 0}
    original_build = dm._build_indices

    def counting_build(*args, **kwargs):
        call_count["n"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(dm, "_build_indices", counting_build)
    dm._clear_cache()

    for shape in [(3, 3, 3), (3, 3, 6), (3, 6, 3), (6, 3, 3)]:
        ax, ay, az = dummy_chain_arrays(shape)
        atoms = build_nbo2f(shape, ax, ay, az)
        SublatticeOccupation.from_atoms(atoms, N=shape, species="O")

    assert call_count["n"] == 4


def test_decompose_scalar_and_tuple_N_equivalent(monkeypatch):
    """SublatticeOccupation.from_atoms(atoms, N=3, species="F") and
    SublatticeOccupation.from_atoms(atoms, N=(3, 3, 3), species="F") produce
    bit-identical SublatticeOccupations and hit the same cache entry."""
    import importlib
    dm = importlib.import_module("chainorder.decompose")

    call_count = {"n": 0}
    original_build = dm._build_indices

    def counting_build(*args, **kwargs):
        call_count["n"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(dm, "_build_indices", counting_build)
    dm._clear_cache()

    N = 3
    ax_in = perfect_oof_chain(N, phase=2)
    ay_in = perfect_oof_chain(N, phase=0)
    az_in = perfect_oof_chain(N, phase=1)
    atoms = build_nbo2f(N, ax_in, ay_in, az_in)

    out_scalar = SublatticeOccupation.from_atoms(atoms, N=3, species="F")
    out_tuple = SublatticeOccupation.from_atoms(atoms, N=(3, 3, 3), species="F")

    np.testing.assert_array_equal(out_scalar.x, out_tuple.x)
    np.testing.assert_array_equal(out_scalar.y, out_tuple.y)
    np.testing.assert_array_equal(out_scalar.z, out_tuple.z)
    assert call_count["n"] == 1, (
        f"Scalar and tuple N should share one cache entry; "
        f"_build_indices was called {call_count['n']} times."
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_detects_in_place_position_mutation(monkeypatch, shape):
    """In-place mutation of `atoms.positions` must invalidate the cache.

    The cache stores a copy of positions on miss, not a reference, so
    that a later call with mutated positions compares as different and
    triggers a rebuild. Without that copy, the cached reference would
    be equal to itself and the rebuild would be skipped silently.
    """
    import importlib
    dm = importlib.import_module("chainorder.decompose")

    call_count = {"n": 0}
    original_build = dm._build_indices

    def counting_build(*args, **kwargs):
        call_count["n"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(dm, "_build_indices", counting_build)
    dm._clear_cache()

    ax_in, ay_in, az_in = dummy_chain_arrays(shape)
    atoms = build_nbo2f(shape, ax_in, ay_in, az_in)
    SublatticeOccupation.from_atoms(atoms, N=shape, species="O")

    # Shift every atom by one lattice vector along x. Positions differ
    # in value; under periodic wrap they still map to on-lattice sites,
    # so the second decomposition succeeds -- but from a distinct
    # positions buffer state, which the cache must notice.
    atoms.positions[:] += np.array([3.90, 0.0, 0.0])
    SublatticeOccupation.from_atoms(atoms, N=shape, species="O")

    assert call_count["n"] == 2, (
        f"In-place position mutation should trigger a rebuild; "
        f"_build_indices was called {call_count['n']} times."
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_decompose_cache_hit_with_new_symbols_same_positions(monkeypatch, shape):
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
    dm._clear_cache()

    ax_first, _, az_first = dummy_chain_arrays(shape)
    ay_first = oof_or_zero(shape, phase=0, direction="y")
    atoms_first = build_nbo2f(shape, ax_first, ay_first, az_first)

    ax_second, _, az_second = dummy_chain_arrays(shape)
    ay_second = oof_or_zero(shape, phase=1, direction="y")
    atoms_second = build_nbo2f(shape, ax_second, ay_second, az_second)
    # Positions and cell are identical (build_nbo2f uses the same lattice
    # geometry regardless of the chain arrays, only the species symbols
    # differ), so the index map should be reusable.

    # Use species="O" so that all-zero F patterns (when a shape axis is
    # not divisible by 3) still satisfy the species-present check.
    out_first = SublatticeOccupation.from_atoms(atoms_first, N=shape, species="O")
    out_second = SublatticeOccupation.from_atoms(atoms_second, N=shape, species="O")

    # Index map built exactly once.
    assert call_count["n"] == 1, f"Expected 1 build, got {call_count['n']}"

    # Each output reflects its own atoms' species (complement since species="O").
    np.testing.assert_array_equal(out_first.x, 1 - ax_first)
    np.testing.assert_array_equal(out_first.y, 1 - ay_first)
    np.testing.assert_array_equal(out_first.z, 1 - az_first)
    np.testing.assert_array_equal(out_second.x, 1 - ax_second)
    np.testing.assert_array_equal(out_second.y, 1 - ay_second)
    np.testing.assert_array_equal(out_second.z, 1 - az_second)
