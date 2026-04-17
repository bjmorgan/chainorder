# chainorder

Chain decomposition and order parameters for ReO3-type anion ordering.

## What this is

`chainorder` is a small NumPy + ASE library for analysing anion ordering
in MD or MC snapshots of ReO3-type solid solutions (NbO2F, TiOF2, and any
other cubic `MX3`-topology system). It takes an on-lattice ASE `Atoms`
object in an orthorhombic supercell, decomposes the three edge-midpoint
anion sublattices into per-direction chains, and exposes a small set of
order parameters computed either over the full 3D occupation (coherent
structure factor) or along individual chain directions (Fourier spectra,
pair correlation, cyclic motif counts, inter-chain correlation).

## Installation

Requires Python 3.11+, NumPy, and ASE.

Clone and install in editable mode:

```bash
git clone git@github.com:bjmorgan/chainorder.git
cd chainorder
pip install -e .
```

Or in one step:

```bash
pip install git+https://github.com/bjmorgan/chainorder.git
```

For development (adds `pytest` and `mypy`):

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from ase.io import read
from chainorder import SublatticeOccupation, order_params

atoms = read("frame.xyz")                                   # any ASE-readable format
occ = SublatticeOccupation.from_atoms(atoms, N=6, species="F")  # supercell size (cubic shorthand)

sf = order_params.structure_factor(occ)                      # 3D structure factor
spectrum = order_params.chain_fft(occ.x)                     # per x-chain FFT
g_r = order_params.along_chain_correlation(occ.x)            # g(r) along x-chains
counts = order_params.motif_counts(occ.x, window_length=3)   # cyclic motif tallies
G = order_params.inter_chain_correlation(occ.x)              # period-3 phase correlation
```

A full trajectory is just a loop: per frame, build a
`SublatticeOccupation` and call whichever order parameter(s) you need.

## Public API at a glance

`chainorder`:

- `SublatticeOccupation` -- frozen dataclass holding the decomposed anion
  occupation. Construct it from an ASE `Atoms` with
  `SublatticeOccupation.from_atoms(atoms, N=...)`. The primary field
  `.occupation` is a read-only `(3, Nx, Ny, Nz)` integer array;
  `.x`, `.y`, `.z` are chain-layout transpose views suited to the
  single-direction tools below.
- `order_params` -- submodule of order parameters:
  - `structure_factor(occ)` -- coherent 3D structure factor of the full
    anion sublattice.
  - `chain_fft(arr)` -- discrete Fourier transform along each chain.
  - `along_chain_correlation(arr)` -- pair correlation g(r) along chains,
    grand-averaged over the chain-plane.
  - `motif_counts(arr, window_length=N)` -- tallies of cyclic-equivalent
    length-`window_length` motifs per chain.
  - `inter_chain_correlation(arr)` -- spatial autocorrelation of the
    period-3 Fourier component across the chain plane.

`structure_factor` takes the whole `SublatticeOccupation`; the other
four take a single chain-layout array (`occ.x`, `occ.y`, or `occ.z`).

Each function's docstring covers shape conventions, normalisation, and
edge cases in detail.

## Related material

- `docs/concepts.md` -- what each order parameter measures and when to
  reach for it (forthcoming).
- `docs/tutorial.ipynb` -- end-to-end worked example on a synthetic
  ordered structure (forthcoming).
