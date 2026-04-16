"""Test fixtures: hand-built on-lattice Atoms with known chain structure."""
import numpy as np
from ase import Atoms


SHAPES: list[tuple[int, int, int]] = [(3, 3, 3), (6, 6, 6), (2, 3, 4)]


def _normalise_shape(N: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Accept scalar N (cubic shorthand) or a 3-tuple and return (Nx, Ny, Nz)."""
    if isinstance(N, (int, np.integer)):
        return (int(N), int(N), int(N))
    Nx, Ny, Nz = N
    return (int(Nx), int(Ny), int(Nz))


def build_nbo2f(
    N: int | tuple[int, int, int],
    anion_x: np.ndarray,
    anion_y: np.ndarray,
    anion_z: np.ndarray,
    a: float = 3.90,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Atoms:
    """Build an on-lattice NbO2F ASE Atoms from three per-direction chain arrays.

    `anion_x[j, k, i] == 1` -> F, `0` -> O. Same for y and z. `N` may be a
    scalar (cubic shorthand for `(N, N, N)`) or a 3-tuple `(Nx, Ny, Nz)`.
    Atom ordering inside each block matches the decompose-test conventions
    (Nb first, then x-, y-, z-anion blocks), so tests that reference
    `atoms.positions[i]` at known indices continue to work.
    """
    Nx, Ny, Nz = _normalise_shape(N)
    ox, oy, oz = origin

    # Cation: sweep (i, j, k) over (Nx, Ny, Nz); position (i+ox, j+oy, k+oz) * a.
    ci, cj, ck = np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij"
    )
    cation_pos = np.stack(
        [(ci + ox) * a, (cj + oy) * a, (ck + oz) * a], axis=-1
    ).reshape(-1, 3)
    cation_sym = np.full(Nx * Ny * Nz, "Nb", dtype="<U2")

    # x-anion: sweep (j, k, i) over (Ny, Nz, Nx); at (i+0.5+ox, j+oy, k+oz) * a.
    xj, xk, xi = np.meshgrid(
        np.arange(Ny), np.arange(Nz), np.arange(Nx), indexing="ij"
    )
    x_pos = np.stack(
        [(xi + 0.5 + ox) * a, (xj + oy) * a, (xk + oz) * a], axis=-1
    ).reshape(-1, 3)
    x_sym = np.where(anion_x[xj, xk, xi].astype(bool), "F", "O").reshape(-1)

    # y-anion: sweep (i, k, j) over (Nx, Nz, Ny); at (i+ox, j+0.5+oy, k+oz) * a.
    yi, yk, yj = np.meshgrid(
        np.arange(Nx), np.arange(Nz), np.arange(Ny), indexing="ij"
    )
    y_pos = np.stack(
        [(yi + ox) * a, (yj + 0.5 + oy) * a, (yk + oz) * a], axis=-1
    ).reshape(-1, 3)
    y_sym = np.where(anion_y[yi, yk, yj].astype(bool), "F", "O").reshape(-1)

    # z-anion: sweep (i, j, k) over (Nx, Ny, Nz); at (i+ox, j+oy, k+0.5+oz) * a.
    zi, zj, zk = np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij"
    )
    z_pos = np.stack(
        [(zi + ox) * a, (zj + oy) * a, (zk + 0.5 + oz) * a], axis=-1
    ).reshape(-1, 3)
    z_sym = np.where(anion_z[zi, zj, zk].astype(bool), "F", "O").reshape(-1)

    positions = np.concatenate([cation_pos, x_pos, y_pos, z_pos])
    symbols = np.concatenate([cation_sym, x_sym, y_sym, z_sym]).tolist()
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.diag([Nx * a, Ny * a, Nz * a]),
        pbc=True,
    )


def perfect_oof_chain(
    N: int | tuple[int, int, int],
    phase: int = 2,
    direction: str = "z",
) -> np.ndarray:
    """Per-direction binary array with every chain OOF (F at i == phase mod 3).

    Shape is chain-layout for the given direction: `(Ny, Nz, Nx)` for x,
    `(Nx, Nz, Ny)` for y, `(Nx, Ny, Nz)` for z. Last axis is along-chain.
    """
    Nx, Ny, Nz = _normalise_shape(N)
    shape_by_direction = {
        "x": (Ny, Nz, Nx),
        "y": (Nx, Nz, Ny),
        "z": (Nx, Ny, Nz),
    }
    lateral0, lateral1, chain = shape_by_direction[direction]
    assert chain % 3 == 0, f"chain-direction length must be divisible by 3 for OOF, got {chain}"
    arr = np.zeros((lateral0, lateral1, chain), dtype=int)
    for i in range(chain):
        if i % 3 == phase:
            arr[:, :, i] = 1
    return arr


def perfect_ofof_chain(
    N: int | tuple[int, int, int],
    direction: str = "z",
) -> np.ndarray:
    """Per-direction binary array with every chain OFOF (F at odd i).

    Shape is chain-layout for the given direction. Last axis is along-chain.
    """
    Nx, Ny, Nz = _normalise_shape(N)
    shape_by_direction = {
        "x": (Ny, Nz, Nx),
        "y": (Nx, Nz, Ny),
        "z": (Nx, Ny, Nz),
    }
    lateral0, lateral1, chain = shape_by_direction[direction]
    assert chain % 2 == 0, f"chain-direction length must be even for OFOF, got {chain}"
    arr = np.zeros((lateral0, lateral1, chain), dtype=int)
    for i in range(chain):
        if i % 2 == 1:
            arr[:, :, i] = 1
    return arr
