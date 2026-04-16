"""Test fixtures: hand-built on-lattice Atoms with known chain structure."""
import numpy as np
from ase import Atoms


def build_nbo2f(
    N: int,
    anion_x: np.ndarray,
    anion_y: np.ndarray,
    anion_z: np.ndarray,
    a: float = 3.90,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Atoms:
    """Build an on-lattice NbO2F ASE Atoms from three (N, N, N) chain arrays.

    Per-direction axis conventions match the build code below:

    - ``anion_x[j, k, i] == 1`` -> F, ``0`` -> O
    - ``anion_y[i, k, j] == 1`` -> F, ``0`` -> O
    - ``anion_z[i, j, k] == 1`` -> F, ``0`` -> O

    Atom ordering inside each block matches the decompose-test conventions
    (Nb first, then x-, y-, z-anion blocks), so tests that reference
    ``atoms.positions[i]`` at known indices continue to work.
    """
    ox, oy, oz = origin
    # idx0, idx1, idx2 sweep three axes in C order: flatten(N, N, N) is
    # outer-idx0, middle-idx1, inner-idx2 -- identical to the original
    # triple loop. The variable-name aliases below mark which lattice
    # index each axis represents for each atom block.
    idx0, idx1, idx2 = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing="ij"
    )

    # Cation: outer i, middle j, inner k. Position (i+ox, j+oy, k+oz) * a.
    cation_pos = np.stack(
        [(idx0 + ox) * a, (idx1 + oy) * a, (idx2 + oz) * a], axis=-1
    ).reshape(-1, 3)
    cation_sym = np.full(N ** 3, "Nb", dtype="<U2")

    # x-anion: outer j, middle k, inner i; at (i+0.5+ox, j+oy, k+oz) * a.
    j_x, k_x, i_x = idx0, idx1, idx2
    x_pos = np.stack(
        [(i_x + 0.5 + ox) * a, (j_x + oy) * a, (k_x + oz) * a], axis=-1
    ).reshape(-1, 3)
    x_sym = np.where(anion_x[j_x, k_x, i_x].astype(bool), "F", "O").reshape(-1)

    # y-anion: outer i, middle k, inner j; at (i+ox, j+0.5+oy, k+oz) * a.
    i_y, k_y, j_y = idx0, idx1, idx2
    y_pos = np.stack(
        [(i_y + ox) * a, (j_y + 0.5 + oy) * a, (k_y + oz) * a], axis=-1
    ).reshape(-1, 3)
    y_sym = np.where(anion_y[i_y, k_y, j_y].astype(bool), "F", "O").reshape(-1)

    # z-anion: outer i, middle j, inner k; at (i+ox, j+oy, k+0.5+oz) * a.
    i_z, j_z, k_z = idx0, idx1, idx2
    z_pos = np.stack(
        [(i_z + ox) * a, (j_z + oy) * a, (k_z + 0.5 + oz) * a], axis=-1
    ).reshape(-1, 3)
    z_sym = np.where(anion_z[i_z, j_z, k_z].astype(bool), "F", "O").reshape(-1)

    positions = np.concatenate([cation_pos, x_pos, y_pos, z_pos])
    symbols = np.concatenate([cation_sym, x_sym, y_sym, z_sym]).tolist()
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.diag([N * a, N * a, N * a]),
        pbc=True,
    )


def perfect_oof_chain(N: int, phase: int = 2) -> np.ndarray:
    """Shape-(N, N, N) binary array; every chain is OOF with F at i == phase (mod 3)."""
    assert N % 3 == 0, "N must be divisible by 3 for exact OOF"
    arr = np.zeros((N, N, N), dtype=int)
    for i in range(N):
        if i % 3 == phase:
            arr[:, :, i] = 1
    return arr


def perfect_ofof_chain(N: int) -> np.ndarray:
    """Shape-(N, N, N) binary array; every chain is OFOF (period 2, F at odd positions)."""
    assert N % 2 == 0, "N must be even for exact OFOF"
    arr = np.zeros((N, N, N), dtype=int)
    for i in range(N):
        if i % 2 == 1:
            arr[:, :, i] = 1
    return arr
