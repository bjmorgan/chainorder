"""Decompose on-lattice ReO3-type supercells into a sublattice occupation."""
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols


_TOL = 1e-6


class Direction(IntEnum):
    """Named indices for the three Cartesian chain directions.

    Values are integers (`IntEnum`) so the members can be used directly as
    numpy indices (e.g. into the internal per-direction index map) or as
    the 0/1/2 axis labels used throughout this module.
    """

    X = 0
    Y = 1
    Z = 2


@dataclass(frozen=True)
class SublatticeOccupation:
    """Anion sublattice occupation of a ReO3-type supercell.

    The primary field `occupation` has shape ``(3, Nx, Ny, Nz)`` with
    axes ``(direction, i, j, k)``: layer ``direction`` holds the species
    flag (``1`` for `species`, ``0`` otherwise) of the anion at edge
    midpoint ``(i, j, k) + 1/2 e_direction``. The `Direction` IntEnum
    names the three layers (``X``, ``Y``, ``Z``).

    Chain-layout per-direction views (`.x`, `.y`, `.z`) are O(1)
    `transpose` views of `occupation` with the last axis brought to the
    along-chain position -- the shape convention the single-direction
    tools in `order_params` (``chain_fft``, ``motif_counts``,
    ``along_chain_correlation``, ``inter_chain_correlation``) expect:

    - ``x``: shape ``(Ny, Nz, Nx)``; ``x[j, k, i]`` is the x-chain at
      lateral position ``(j, k)`` in the ``(y, z)`` plane, site ``i``
      along x.
    - ``y``: shape ``(Nx, Nz, Ny)``; ``y[i, k, j]`` is the y-chain at
      lateral position ``(i, k)``, site ``j`` along y.
    - ``z``: shape ``(Nx, Ny, Nz)``; ``z[i, j, k]`` is the z-chain at
      lateral position ``(i, j)``, site ``k`` along z.

    For 3D operations (``structure_factor``) pass the `SublatticeOccupation`
    instance directly -- the 3D form is the primary representation and
    passing it whole avoids the chain-layout-and-back transpose round
    trip the old positional-array API imposed. For a cubic supercell
    all three chain-layout views reduce to the common shape ``(N, N, N)``.

    Supports positional unpacking (``ax, ay, az = SublatticeOccupation.from_atoms(...)``)
    via ``__iter__``, which yields ``(x, y, z)`` in direction order. A
    transposed unpacking (``az, ay, ax = ...``) would silently swap the
    x and z chain-layout arrays and produce a physically wrong
    downstream answer; prefer attribute access (``result.x``) where
    direction matters. Passing the whole ``SublatticeOccupation`` to
    3D-operation functions (``structure_factor``) removes this hazard
    at those call sites.
    """

    occupation: np.ndarray

    @property
    def x(self) -> np.ndarray:
        """x-chain sublattice, chain-layout shape ``(Ny, Nz, Nx)``."""
        return self.occupation[Direction.X].transpose(1, 2, 0)

    @property
    def y(self) -> np.ndarray:
        """y-chain sublattice, chain-layout shape ``(Nx, Nz, Ny)``."""
        return self.occupation[Direction.Y].transpose(0, 2, 1)

    @property
    def z(self) -> np.ndarray:
        """z-chain sublattice, chain-layout shape ``(Nx, Ny, Nz)``."""
        return self.occupation[Direction.Z]

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield the three chain-layout views in direction order."""
        yield self.x
        yield self.y
        yield self.z

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        *,
        N: int | tuple[int, int, int],
        species: str,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "SublatticeOccupation":
        """Decompose an on-lattice ReO3-type supercell into a sublattice occupation.

        Identifies each anion from its fractional coordinates (half-integer in
        exactly one axis, integer in the other two) and assigns it to a slot in
        a 3D sublattice occupation array of shape (3, Nx, Ny, Nz) -- axis 0
        indexes the three edge-midpoint sublattices, axes 1-3 the xyz grid.
        The decomposition map is cached on `(positions, cell, shape, origin)`,
        so repeated calls on frames with identical positions skip the mapping
        step; on typical supercell sizes this gives roughly an order of
        magnitude speedup per frame.
        Off-lattice MD trajectories (positions perturbed thermally) are out of
        scope; they would not pass the on-lattice check anyway.

        Args:
            atoms: On-lattice ASE `Atoms` supercell with anions at ideal edge
                midpoints of a simple-cubic cation sublattice.
            N: Supercell size. Scalar (cubic shorthand for `(N, N, N)`) or a
                3-tuple `(Nx, Ny, Nz)` of positive integers. The cell diagonal
                must match along each axis.
            origin: Fractional offset of the cation sub-lattice within each
                unit cell. Anions are assumed to sit at (cation position + 1/2)
                along one axis. Default `(0.0, 0.0, 0.0)` places cations at
                unit-cell corners and anions at edge midpoints;
                `(0.5, 0.5, 0.5)` corresponds to body-centred cations. Each
                component must lie in `[0.0, 1.0)`; values outside this range
                raise.
            species: Element symbol to flag as 1 in the output occupation.
                All other anion species are flagged 0.

        Returns:
            A SublatticeOccupation whose primary field .occupation has shape
            (3, Nx, Ny, Nz) with axes (direction, i, j, k). The .x, .y, .z
            properties expose chain-layout transpose views (shapes
            (Ny, Nz, Nx), (Nx, Nz, Ny), (Nx, Ny, Nz) respectively; reducing
            to (N, N, N) in the cubic case). Positional unpacking yields the
            three chain-layout views in direction order.

        Raises:
            ValueError: On any of the validation failures below -- invalid `N`
                (non-integer scalar, non-positive scalar, tuple of wrong
                length, non-integer tuple element, or non-positive tuple
                element); `origin` component outside `[0.0, 1.0)`; cell
                containing non-finite values, not orthorhombic, or
                non-positive on the diagonal; wrong cation or anion count
                for the given shape; any atom off-lattice (including the
                common case of passing the wrong `origin` -- e.g.
                `(0.0, 0.0, 0.0)` on a body-centred structure); `species`
                not a known chemical element symbol (e.g. `"Fluorine"` or
                `"f"` instead of `"F"`); `species` absent from all anion
                sites; or a slot collision during assignment.
        """
        shape = _validate_shape(N)
        origin = _validate_origin(origin)
        positions = np.ascontiguousarray(atoms.positions, dtype=np.float64)
        cell = np.ascontiguousarray(atoms.cell.array, dtype=np.float64)
        indices = _indices_cached(positions.tobytes(), cell.tobytes(), shape, origin)
        return _apply_indices(atoms.numbers, indices, species)


def _validate_shape(
    N: int | tuple[int, int, int],
) -> tuple[int, int, int]:
    """Normalise `N` to a canonical (Nx, Ny, Nz) tuple of positive ints.

    Accepts scalar (cubic shorthand) or a 3-tuple. Rejects wrong length,
    non-integer elements, or non-positive elements with `ValueError`.
    """
    if isinstance(N, (int, np.integer)):
        if N < 1:
            raise ValueError(f"N must be a positive integer, got {N!r}.")
        return (int(N), int(N), int(N))
    # Tuple path
    try:
        items = tuple(N)
    except TypeError:
        raise ValueError(
            f"N must be a positive integer or a 3-tuple of positive "
            f"integers, got {N!r}."
        )
    if len(items) != 3:
        raise ValueError(
            f"N tuple must have length 3 (Nx, Ny, Nz), got length "
            f"{len(items)}: {N!r}."
        )
    out: list[int] = []
    for axis, v in zip(("Nx", "Ny", "Nz"), items):
        if not isinstance(v, (int, np.integer)):
            raise ValueError(
                f"{axis} must be an integer, got {type(v).__name__} ({v!r})."
            )
        if v < 1:
            raise ValueError(f"{axis} must be a positive integer, got {v}.")
        out.append(int(v))
    return (out[0], out[1], out[2])


def _validate_origin(
    origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Normalise `origin` to a float 3-tuple with each component in `[0, 1)`.

    The float cast ensures `(0, 0, 0)` and `(0.0, 0.0, 0.0)` share an
    lru_cache key. Out-of-range values are rejected because they would
    wrap silently through the `frac % 1.0` operation in `_build_indices`;
    the caller should wrap deliberately if they want periodic equivalence.
    """
    a, b, c = (float(origin[0]), float(origin[1]), float(origin[2]))
    for name, v in (("origin[0]", a), ("origin[1]", b), ("origin[2]", c)):
        if not 0.0 <= v < 1.0:
            raise ValueError(
                f"{name} must lie in [0.0, 1.0), got {v}."
            )
    return (a, b, c)


@lru_cache(maxsize=1)
def _indices_cached(
    positions_bytes: bytes,
    cell_bytes: bytes,
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Cache the decomposition map for one ``(positions, cell, shape, origin)`` key.

    ``positions`` and ``cell`` are passed as raw bytes so that the key is
    hashable and compares by full-precision binary content. Single-entry cache
    (``maxsize=1``) is sufficient for trajectory analysis: repeated calls with
    identical inputs reuse the cached indices, and the first call after any
    change rebuilds.
    """
    positions = np.frombuffer(positions_bytes, dtype=np.float64).reshape(-1, 3)
    cell = np.frombuffer(cell_bytes, dtype=np.float64).reshape(3, 3)
    return _build_indices(positions, cell, shape, origin)


def _build_indices(
    positions: np.ndarray,
    cell: np.ndarray,
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Build the atom-to-chain-slot index mapping.

    Args:
        positions: Cartesian atom positions in Angstroms, shape (n_atoms, 3).
        cell: Orthorhombic cell matrix in Angstroms, shape (3, 3).
        shape: Supercell size per axis as a (Nx, Ny, Nz) tuple.
        origin: Cation position in unit-cell fractional coordinates.

    Returns:
        Integer array of shape (3, Nx, Ny, Nz) mapping (direction, i, j, k)
        to the index of the corresponding atom in `positions`. Each layer is
        indexed by the xyz-grid coordinate of the anion (not by chain-layout
        order); `_apply_indices` packages the species-masked layers into
        a `SublatticeOccupation`, whose chain-layout views transpose each
        layer as needed.

    Raises:
        ValueError: On non-orthorhombic cells, off-lattice atoms, wrong atom
            counts, or slot collisions.
    """
    Nx, Ny, Nz = shape
    N_per_axis = np.asarray(shape, dtype=np.float64)        # (3,)
    N_max = int(max(shape))
    # Finite check on the whole matrix: NaN anywhere in `cell` would propagate
    # silently through the orthorhombic threshold (`NaN > x` is False) and
    # the on-lattice check, giving platform-dependent int garbage downstream.
    if not np.all(np.isfinite(cell)):
        raise ValueError(
            f"Cell must contain only finite values, got {cell}."
        )

    # Orthorhombic check: off-diagonal elements must be zero (within tolerance
    # scaled by the largest cell-vector component).
    diag = np.diag(cell)
    off_diag = cell - np.diag(diag)
    if np.any(np.abs(off_diag) > _TOL * max(1.0, np.abs(cell).max())):
        raise ValueError(
            f"Cell is not orthorhombic (off-diagonal elements present): {off_diag}. "
            f"Non-orthorhombic cells are out of scope for this version."
        )

    # Diagonal must be positive: zero- or negative-length cell vectors would
    # produce degenerate (NaN) transforms even though the finite check passed.
    if np.any(diag <= 0):
        raise ValueError(
            f"Cell diagonal must be positive, got {diag}."
        )

    # Uniform lattice parameter check: `diag / N_per_axis` should be the
    # same per axis. A mismatch means the shape tuple doesn't correspond
    # to the cell -- most commonly a permuted N (e.g. passing (Nx, Nz, Ny)
    # for a cell with axis lengths Nx*a, Ny*a, Nz*a). Without this check
    # the permutation sails through to the on-lattice test, which then
    # reports "atom not on-lattice" even though the structure IS on-
    # lattice with the correct N.
    per_axis_a = diag / N_per_axis
    if not np.allclose(per_axis_a, per_axis_a[0], rtol=_TOL):
        raise ValueError(
            f"Shape {shape} does not match cell: per-axis lattice parameters "
            f"implied by diag/N are {tuple(per_axis_a)}, which disagree. The "
            f"cell diagonal {tuple(diag)} implies a single lattice parameter "
            f"only when N is proportional to the diagonal. Check that N is "
            f"passed in the (Nx, Ny, Nz) order matching the cell's "
            f"(x, y, z) diagonals."
        )

    inv_cell = np.linalg.inv(cell)
    # The origin offset is a unit-cell-fractional shift (how the cation
    # sits within each unit cell). Convert to supercell-fractional per
    # axis before subtracting.
    frac = positions @ inv_cell - np.asarray(origin, dtype=np.float64) / N_per_axis
    frac = frac % 1.0

    # Scale so integer grid is [0, N_axis) and half-integer grid is
    # [0.5, N_axis) along each axis.
    scaled = frac * N_per_axis                              # shape (n_atoms, 3)

    # Round to nearest half-integer: 2*scaled should be an integer.
    half_rounded = np.round(2 * scaled).astype(int)         # shape (n_atoms, 3)

    # Canonicalise half_rounded to [0, 2*N_axis) per axis. After %1.0 and
    # per-axis scaling, scaled is in [0, N_axis); half_rounded can be 0
    # or 2*N_axis at the wrap boundary.
    half_rounded = half_rounded % (2 * np.asarray(shape, dtype=np.int64))

    # Minimum image distance on the [0, N_axis) circle per axis (width
    # 2*N_axis for the half-integer grid).
    expected = half_rounded / 2
    deviation = np.minimum(
        np.abs(scaled - expected),
        np.abs(scaled - expected - N_per_axis),
    )
    deviation = np.minimum(deviation, np.abs(scaled - expected + N_per_axis))
    tol = _TOL * max(1.0, N_max)
    if np.any(deviation > tol):
        worst_per_atom = deviation.max(axis=1)
        bad = int(np.argmax(worst_per_atom))
        bad_axis = int(np.argmax(deviation[bad]))
        axis_label = "xyz"[bad_axis]
        raise ValueError(
            f"Atom {bad} is not on-lattice: axis {axis_label} deviation "
            f"{deviation[bad, bad_axis]:.3g} (tolerance {tol:.3g}). "
            f"Scaled fractional coords: {scaled[bad]}. Expected integer or "
            f"half-integer coordinates."
        )

    # Classify: coord is integer iff half_rounded is even; half-integer iff odd.
    is_half = (half_rounded % 2 == 1)      # shape (n_atoms, 3)
    n_half = is_half.sum(axis=1)           # shape (n_atoms,)

    if np.any(n_half > 1):
        bad = int(np.where(n_half > 1)[0][0])
        raise ValueError(
            f"Atom {bad} at scaled coords {scaled[bad]} has {n_half[bad]} "
            f"half-integer coordinates; expected 0 (cation) or 1 (anion)."
        )

    # Count checks
    n_cation = int((n_half == 0).sum())
    n_anion = int((n_half == 1).sum())
    n_cells = Nx * Ny * Nz
    expected_cation = n_cells
    expected_anion = 3 * n_cells
    if n_cation != expected_cation:
        raise ValueError(
            f"Wrong cation count: found {n_cation}, "
            f"expected {expected_cation} for shape={shape}."
        )
    if n_anion != expected_anion:
        raise ValueError(
            f"Wrong anion count: found {n_anion}, "
            f"expected {expected_anion} for shape={shape}."
        )

    # Build indices
    indices = np.full((3, Nx, Ny, Nz), -1, dtype=np.int64)
    anion_atoms = np.where(n_half == 1)[0]
    # Per-axis wrap of the integer part of scaled coords.
    coord = (half_rounded // 2) % np.asarray(shape, dtype=np.int64)

    # Each anion has exactly one half-integer axis (= its chain direction)
    # and writes to `indices[direction, i, j, k]`. Vectorised fancy-index
    # write: last-write-wins under duplicate indices matches the Python
    # loop's overwrite semantics, so a rounding-collision failure still
    # leaves some slot unfilled and is caught by the `-1` check below.
    directions = np.argmax(is_half[anion_atoms], axis=1)                # (n_anions,)
    ijk = coord[anion_atoms]                                            # (n_anions, 3)
    indices[directions, ijk[:, 0], ijk[:, 1], ijk[:, 2]] = anion_atoms

    if np.any(indices == -1):
        missing = np.argwhere(indices == -1)
        d, i, j, k = (int(x) for x in missing[0])
        axis_label = "xyz"[d]
        raise ValueError(
            f"Decomposition incomplete: slot (direction={axis_label}, "
            f"grid=({i}, {j}, {k})) is unfilled. "
            f"{len(missing)} slot(s) unfilled in total. This usually means "
            f"two anions mapped to the same slot (rounding collision) or the "
            f"supercell dimensions do not match the shape argument."
        )

    return indices


def _apply_indices(
    numbers: np.ndarray,
    indices: np.ndarray,
    species: str,
) -> SublatticeOccupation:
    """Apply a cached index map to produce a `SublatticeOccupation`.

    Args:
        numbers: Atomic numbers per atom, shape (n_atoms,).
        indices: Cached decomposition map from `_build_indices`,
            shape (3, Nx, Ny, Nz), xyz-coord indexed.
        species: Element symbol to flag as 1. Others are flagged 0.

    Returns:
        `SublatticeOccupation` whose `.occupation` field is the
        shape-(3, Nx, Ny, Nz) xyz-coord integer array. Chain-layout
        views are exposed via the `.x`, `.y`, `.z` properties.
    """
    try:
        target_z = atomic_numbers[species]
    except KeyError:
        raise ValueError(
            f"species={species!r} is not a known chemical element symbol."
        )
    anion_numbers = numbers[indices]
    is_species = (anion_numbers == target_z).astype(np.int64)
    if is_species.sum() == 0:
        present = sorted({chemical_symbols[int(z)] for z in anion_numbers.ravel()})
        raise ValueError(
            f"species={species!r} not found on any anion site. "
            f"Anion species present: {present}."
        )
    # Make the buffer read-only so that SublatticeOccupation's
    # @dataclass(frozen=True) immutability extends to the underlying
    # ndarray. Without this an accidental `result.occupation[...] = x`
    # (or a mutation through one of the transpose views) would silently
    # corrupt the decomposition. Transpose views share the base's
    # writeable flag, so .x/.y/.z inherit the read-only state.
    is_species.flags.writeable = False
    return SublatticeOccupation(occupation=is_species)
