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

    anion_x[j, k, i] == 1 -> F, 0 -> O. Same for y and z.
    """
    positions = []
    symbols = []
    ox, oy, oz = origin

    # Nb atoms at integer grid positions
    for i in range(N):
        for j in range(N):
            for k in range(N):
                positions.append([(i + ox) * a, (j + oy) * a, (k + oz) * a])
                symbols.append("Nb")

    # x-anions at (i + 0.5, j, k)
    for j in range(N):
        for k in range(N):
            for i in range(N):
                positions.append([(i + 0.5 + ox) * a, (j + oy) * a, (k + oz) * a])
                symbols.append("F" if anion_x[j, k, i] else "O")

    # y-anions at (i, j + 0.5, k)
    for i in range(N):
        for k in range(N):
            for j in range(N):
                positions.append([(i + ox) * a, (j + 0.5 + oy) * a, (k + oz) * a])
                symbols.append("F" if anion_y[i, k, j] else "O")

    # z-anions at (i, j, k + 0.5)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                positions.append([(i + ox) * a, (j + oy) * a, (k + 0.5 + oz) * a])
                symbols.append("F" if anion_z[i, j, k] else "O")

    return Atoms(
        symbols=symbols,
        positions=np.array(positions),
        cell=np.diag([N * a, N * a, N * a]),
        pbc=True,
    )


def perfect_oof_chain(N: int, phase: int = 2) -> np.ndarray:
    """Shape-(N, N, N) binary array; every chain is OOF with F at i ≡ phase (mod 3)."""
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
