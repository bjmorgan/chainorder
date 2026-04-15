"""Decompose on-lattice ReO3-type supercells into chain arrays."""
from functools import lru_cache

import numpy as np
from ase import Atoms


_TOL = 1e-6


def decompose(
    atoms: Atoms,
    N: int,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    species: str = "F",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose an on-lattice ReO3-type supercell into three chain arrays.

    Identifies each anion from its fractional coordinates (half-integer in
    exactly one axis, integer in the other two) and assigns it to a slot in
    one of three (N, N, N) arrays — one per chain direction. The assignment
    is cached across calls with identical positions, cell, N, and origin, so
    analysing a trajectory only pays the decomposition cost on the first
    frame.

    Args:
        atoms: On-lattice ASE `Atoms` supercell with anions at ideal edge
            midpoints of a simple-cubic cation sublattice.
        N: Supercell size along each axis (cubic N*N*N).
        origin: Position of the cation within its unit cell, in unit-cell
            fractional coordinates. Default `(0, 0, 0)` puts the cation at
            the unit-cell corner. Pass `(0.5, 0.5, 0.5)` if the cation sits
            at the unit-cell body centre.
        species: Element symbol to flag as 1 in the output arrays. Default
            `"F"`; all other anion species are flagged 0.

    Returns:
        Three integer arrays `(anion_x, anion_y, anion_z)`, each of shape
        (N, N, N). For each array the first two indices identify the chain
        and the last index is position along the chain.

    Raises:
        ValueError: If the cell is not orthorhombic, the atom count is wrong
            for the given N, or any atom is off the expected on-lattice
            positions (within tolerance).
    """
    positions = np.ascontiguousarray(atoms.positions, dtype=np.float64)
    cell = np.ascontiguousarray(atoms.cell.array, dtype=np.float64)
    indices = _indices_cached(positions.tobytes(), cell.tobytes(), N, origin)
    symbols = np.array(atoms.get_chemical_symbols())
    return _apply_indices(symbols, indices, species)


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
    # Orthorhombic check: off-diagonal elements must be zero (within tolerance
    # scaled by the largest cell-vector component).
    off_diag = cell - np.diag(np.diag(cell))
    if np.any(np.abs(off_diag) > _TOL * max(1.0, np.abs(cell).max())):
        raise ValueError(
            f"Cell is not orthorhombic (off-diagonal elements present): {off_diag}. "
            f"Non-orthorhombic cells are out of scope for this version."
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
    # Note: deviation can be ~N near the wrap boundary (scaled ≈ N maps to 0).
    # After %1.0 and *N, scaled ∈ [0, N); half_rounded can be 0 or 2*N. We
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
        bad = int(np.argmax(deviation.max(axis=1)))
        raise ValueError(
            f"Atom {bad} at scaled fractional coords {scaled[bad]} is not on-lattice. "
            f"Expected integer or half-integer coordinates (tolerance {_TOL * N})."
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
        direction = int(np.argmax(is_half[atom_idx]))
        i, j, k = int(coord[atom_idx, 0]), int(coord[atom_idx, 1]), int(coord[atom_idx, 2])
        if direction == 0:    # x-anion: chain (j, k), position i
            indices[0, j, k, i] = atom_idx
        elif direction == 1:  # y-anion: chain (i, k), position j
            indices[1, i, k, j] = atom_idx
        else:                 # z-anion: chain (i, j), position k
            indices[2, i, j, k] = atom_idx

    if np.any(indices == -1):
        raise ValueError(
            "Decomposition incomplete: some chain slots unfilled. This usually "
            "means multiple anions mapped to the same slot (rounding collision) "
            "or the supercell dimensions don't match N."
        )

    return indices


def _apply_indices(
    symbols: np.ndarray,
    indices: np.ndarray,
    species: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a cached index map to produce three (N, N, N) binary arrays.

    Args:
        symbols: Chemical symbols per atom, shape (n_atoms,).
        indices: Cached decomposition map from `_build_indices`,
            shape (3, N, N, N).
        species: Element symbol to flag as 1. Others are flagged 0.

    Returns:
        Three integer arrays `(anion_x, anion_y, anion_z)`, each shape
        (N, N, N).
    """
    is_species = (symbols[indices] == species).astype(np.int64)
    return is_species[0], is_species[1], is_species[2]
