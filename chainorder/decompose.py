"""Decompose on-lattice ReO3-type supercells into chain arrays."""
from enum import IntEnum
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from ase import Atoms


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


class ChainArrays(NamedTuple):
    """Three per-direction anion occupation arrays from a ReO3-type supercell.

    Each field is a shape-``(N, N, N)`` ``int64`` array with values in
    ``{0, 1}``: ``1`` at sites whose species matches the `species` argument
    passed to `decompose`, ``0`` elsewhere. The last axis is always
    along-chain; the first two indices identify the chain's lateral
    position in the sublattice:

    - ``x[j, k, i]``: x-chain at lateral position ``(j, k)``, site ``i`` along x.
    - ``y[i, k, j]``: y-chain at lateral position ``(i, k)``, site ``j`` along y.
    - ``z[i, j, k]``: z-chain at lateral position ``(i, j)``, site ``k`` along z.

    Supports positional unpacking (``ax, ay, az = decompose(...)``) as well
    as attribute access (``result.x``). The latter is safer for downstream
    use since it removes the risk of silently transposing the tuple.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


def decompose(
    atoms: Atoms,
    N: int,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    species: str = "F",
) -> ChainArrays:
    """Decompose an on-lattice ReO3-type supercell into three chain arrays.

    Identifies each anion from its fractional coordinates (half-integer in
    exactly one axis, integer in the other two) and assigns it to a slot in
    one of three (N, N, N) arrays -- one per chain direction. The assignment
    is cached across calls with identical `(positions, cell, N, origin)`,
    so re-analysing the same frame (e.g. with a different `species`) is
    free after the first call, and analysing an MC trajectory of occupation
    swaps -- where atom positions are identical frame-to-frame -- pays the
    decomposition cost only once. Off-lattice MD trajectories (positions
    perturbed thermally) are out of scope; they would not pass the
    on-lattice check anyway.

    Args:
        atoms: On-lattice ASE `Atoms` supercell with anions at ideal edge
            midpoints of a simple-cubic cation sublattice.
        N: Supercell size along each axis (cubic N*N*N).
        origin: Fractional offset of the cation sub-lattice within each
            unit cell. Anions are assumed to sit at (cation position + 1/2)
            along one axis. Default `(0.0, 0.0, 0.0)` places cations at
            unit-cell corners and anions at edge midpoints;
            `(0.5, 0.5, 0.5)` corresponds to body-centred cations. Each
            component must lie in `[0.0, 1.0)`; values outside this range
            raise.
        species: Element symbol to flag as 1 in the output arrays. Default
            `"F"`; all other anion species are flagged 0.

    Returns:
        A `ChainArrays` namedtuple with fields `x`, `y`, `z`, each a shape
        `(N, N, N)` integer array. For each array the first two indices
        identify the chain and the last index is position along the chain.
        Supports positional unpacking.

    Raises:
        TypeError: If `origin` is not a sequence of three numbers (e.g.
            `None`, an int, a string, or bytes).
        ValueError: On any of the validation failures below -- invalid `N`
            (non-integer or non-positive); `origin` of wrong length, or
            with a component outside `[0.0, 1.0)`; cell containing non-
            finite values, not orthorhombic, non-positive on the diagonal,
            or not cubic; wrong cation or anion count for the given `N`;
            any atom off-lattice (integer or half-integer) beyond
            tolerance; `species` absent from all anion sites; or a slot
            collision during assignment.
    """
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError(f"N must be a positive integer, got {N!r}.")
    N = int(N)
    origin = _validate_origin(origin)
    positions = np.ascontiguousarray(atoms.positions, dtype=np.float64)
    cell = np.ascontiguousarray(atoms.cell.array, dtype=np.float64)
    indices = _indices_cached(positions.tobytes(), cell.tobytes(), N, origin)
    symbols = np.array(atoms.get_chemical_symbols())
    return _apply_indices(symbols, indices, species)


def _validate_origin(
    origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Normalise and validate the `origin` argument to `decompose`.

    Casts each component to `float` (so `(0, 0, 0)` and `(0.0, 0.0, 0.0)`
    produce the same hashable cache key) and enforces that each component
    lies in `[0.0, 1.0)` -- values outside this range would wrap silently
    through the `frac % 1.0` operation in `_build_indices`.

    Raises:
        TypeError: If `origin` is not a sequence (e.g. `None`, `int`), or
            contains a non-numeric component. Strings and bytes are
            explicitly rejected even though they satisfy `len() == 3`,
            because per-character `float()` would accept e.g. `"012"`.
        ValueError: If `origin` has the wrong number of components, or if
            any component lies outside `[0.0, 1.0)`.
    """
    if isinstance(origin, (str, bytes)):
        raise TypeError(
            f"origin must be a sequence of three numbers, got {type(origin).__name__}."
        )
    try:
        length = len(origin)
    except TypeError as exc:
        raise TypeError(
            f"origin must be a sequence of three numbers, got "
            f"{type(origin).__name__}."
        ) from exc
    if length != 3:
        raise ValueError(
            f"origin must have exactly 3 components, got {length}."
        )
    components: list[float] = []
    for i, v in enumerate(origin):
        if not isinstance(v, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"origin[{i}] must be numeric (int or float), got "
                f"{type(v).__name__}."
            )
        components.append(float(v))
    a, b, c = components
    for name, v in (("origin[0]", a), ("origin[1]", b), ("origin[2]", c)):
        if not 0.0 <= v < 1.0:
            raise ValueError(
                f"{name} must lie in [0.0, 1.0), got {v}. Origin is "
                f"expressed in unit-cell fractional coordinates; values "
                f"outside this range are equivalent under periodicity and "
                f"must be wrapped by the caller."
            )
    return (a, b, c)


@lru_cache(maxsize=1)
def _indices_cached(
    positions_bytes: bytes,
    cell_bytes: bytes,
    N: int,
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Cache the decomposition map for one ``(positions, cell, N, origin)`` key.

    ``positions`` and ``cell`` are passed as raw bytes so that the key is
    hashable and compares by full-precision binary content. Single-entry cache
    (``maxsize=1``) is sufficient for trajectory analysis: repeated calls with
    identical inputs reuse the cached indices, and the first call after any
    change rebuilds.
    """
    positions = np.frombuffer(positions_bytes, dtype=np.float64).reshape(-1, 3)
    cell = np.frombuffer(cell_bytes, dtype=np.float64).reshape(3, 3)
    return _build_indices(positions, cell, N, origin)


def _build_indices(
    positions: np.ndarray,
    cell: np.ndarray,
    N: int,
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Build the atom-to-chain-slot index mapping.

    Args:
        positions: Cartesian atom positions in Angstroms, shape (n_atoms, 3).
        cell: Orthorhombic cell matrix in Angstroms, shape (3, 3).
        N: Supercell size per axis.
        origin: Cation position in unit-cell fractional coordinates.

    Returns:
        Integer array of shape (3, N, N, N) mapping (direction, j, k, i) to
        the index of the corresponding atom in `positions`.

    Raises:
        ValueError: On non-orthorhombic cells, off-lattice atoms, wrong atom
            counts, or slot collisions.
    """
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

    # Cubic requirement: N is a single scalar, so the three axes must be
    # equal. Non-cubic orthorhombic cells are out of scope for v1.
    if not np.allclose(diag, diag[0], rtol=_TOL):
        raise ValueError(
            f"Cell must be cubic (equal diagonal components), got diagonal "
            f"{diag}. Non-cubic orthorhombic cells are out of scope for this "
            f"version."
        )

    # Origin is a unit-cell-fractional offset (how the cation sits within
    # each unit cell); convert to supercell-fractional before subtracting.
    inv_cell = np.linalg.inv(cell)
    frac = positions @ inv_cell - np.asarray(origin, dtype=np.float64) / N
    frac = frac % 1.0

    # Scale so integer grid is [0, N) and half-integer grid is [0.5, N)
    scaled = frac * N

    # Round to nearest half-integer: 2*scaled should be an integer.
    half_rounded = np.round(2 * scaled).astype(int)   # shape (n_atoms, 3)

    # Check atoms are on-lattice (tolerance scaled by N)
    deviation = np.abs(scaled - half_rounded / 2)
    # Note: deviation can be ~N near the wrap boundary (scaled ~= N maps to 0).
    # After %1.0 and *N, scaled is in [0, N); half_rounded can be 0 or 2*N. We
    # canonicalise half_rounded by taking mod 2*N.
    half_rounded = half_rounded % (2 * N)
    # Recompute deviation after canonicalisation
    expected = half_rounded / 2
    # Use minimum image distance on the [0, N) circle (width 2N for half-grid)
    deviation = np.minimum(
        np.abs(scaled - expected),
        np.abs(scaled - expected - N),
    )
    deviation = np.minimum(deviation, np.abs(scaled - expected + N))
    if np.any(deviation > _TOL * max(1.0, N)):
        worst_per_atom = deviation.max(axis=1)
        bad = int(np.argmax(worst_per_atom))
        bad_axis = int(np.argmax(deviation[bad]))
        axis_label = "xyz"[bad_axis]
        raise ValueError(
            f"Atom {bad} is not on-lattice: axis {axis_label} deviation "
            f"{deviation[bad, bad_axis]:.3g} (tolerance {_TOL * N:.3g}). "
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
    expected_cation = N ** 3
    expected_anion = 3 * N ** 3
    if n_cation != expected_cation:
        raise ValueError(
            f"Wrong cation count: found {n_cation}, "
            f"expected {expected_cation} for N={N}."
        )
    if n_anion != expected_anion:
        raise ValueError(
            f"Wrong anion count: found {n_anion}, expected {expected_anion} for N={N}."
        )

    # Build indices
    indices = np.full((3, N, N, N), -1, dtype=np.int64)
    anion_atoms = np.where(n_half == 1)[0]
    coord = (half_rounded // 2) % N   # integer part of scaled coords, wrapped

    for atom_idx in anion_atoms:
        direction = Direction(int(np.argmax(is_half[atom_idx])))
        i, j, k = int(coord[atom_idx, 0]), int(coord[atom_idx, 1]), int(coord[atom_idx, 2])
        if direction is Direction.X:    # x-anion: chain (j, k), position i
            indices[Direction.X, j, k, i] = atom_idx
        elif direction is Direction.Y:  # y-anion: chain (i, k), position j
            indices[Direction.Y, i, k, j] = atom_idx
        else:                           # z-anion: chain (i, j), position k
            indices[Direction.Z, i, j, k] = atom_idx

    if np.any(indices == -1):
        missing = np.argwhere(indices == -1)
        d, a, b, c = (int(x) for x in missing[0])
        axis_label = "xyz"[d]
        raise ValueError(
            f"Decomposition incomplete: slot (direction={axis_label}, "
            f"lateral=({a}, {b}), along-chain={c}) is unfilled. "
            f"{len(missing)} slot(s) unfilled in total. This usually means "
            f"two anions mapped to the same slot (rounding collision) or the "
            f"supercell dimensions do not match N."
        )

    return indices


def _apply_indices(
    symbols: np.ndarray,
    indices: np.ndarray,
    species: str,
) -> ChainArrays:
    """Apply a cached index map to produce three (N, N, N) binary arrays.

    Args:
        symbols: Chemical symbols per atom, shape (n_atoms,).
        indices: Cached decomposition map from `_build_indices`,
            shape (3, N, N, N).
        species: Element symbol to flag as 1. Others are flagged 0.

    Returns:
        `ChainArrays` namedtuple of three integer arrays, each shape
        (N, N, N).
    """
    anion_symbols = symbols[indices]
    is_species = (anion_symbols == species).astype(np.int64)
    if is_species.sum() == 0:
        present = sorted(set(anion_symbols.ravel().tolist()))
        raise ValueError(
            f"species={species!r} not found on any anion site. "
            f"Anion species present: {present}."
        )
    return ChainArrays(
        x=is_species[Direction.X],
        y=is_species[Direction.Y],
        z=is_species[Direction.Z],
    )
