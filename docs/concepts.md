# Concepts

`chainorder` analyses on-lattice ASE `Atoms` configurations of
ReO3-type MX3 solid solutions in orthorhombic supercells. This
document describes the problem domain, how configurations are
represented, what each order parameter measures, and a few common
analysis patterns.

## The problem

### ReO3-type topology

ReO3 is the parent structure for a family of `MX3` solids. The cations
`M` sit at the corners of a simple-cubic lattice; the anions `X` sit at
the midpoints of the cube edges. Every unit cell contains one cation and
three anions, one on each of the three mutually perpendicular edge
directions. Each anion is shared between two neighbouring cations, so
the anion-to-cation stoichiometry is 3:1. Extending the unit cell along
each axis gives an `Nx * Ny * Nz` supercell of `Nx * Ny * Nz` cations
and `3 * Nx * Ny * Nz` anions.

Each anion sits on the line joining two adjacent cations along a single
Cartesian axis, so repeating along that axis gives infinite
`...M-X-M-X-M...` chains. By cubic symmetry there are three equivalent
families of such chains -- one along each of x, y, and z -- and every
cation sits at the junction of one chain from each family. `chainorder`
decomposes the anion sublattice into these three chain families; the
decomposition makes chain-based structure -- species patterns along
individual chains, or correlations between neighbouring chains -- the
natural objects of analysis. An x-anion is one that sits on an
x-oriented chain, at a midpoint of the form `(i + 1/2, j, k) * a`,
where `(i, j, k)` is a cation lattice position and `a` is the cubic
lattice parameter; similarly for y-anions and z-anions.

### Anion ordering in solid solutions

In an MX3 solid solution the `X` sites are shared between two or more
anion species. NbO2F (2 O : 1 F per formula unit) and TiOF2 (1 O : 2 F)
are concrete examples; in both the cation sublattice is fixed (Nb or Ti
everywhere) and the structural question reduces to *which anion site
holds which species*.

## The data model

`chainorder` analyses individual configurations. Analysis proceeds in
two steps:

1. Convert the on-lattice structure to a binary occupation
   representation: for `species="F"`, a chain `O-F-F-O` becomes
   `0-1-1-0`, where `1` marks the chosen species and `0` marks
   anything else.
2. Pass the occupation representation to the order-parameter and
   structural-analysis functions.

Step 1 is a single call:

```python
from chainorder import SublatticeOccupation

occ = SublatticeOccupation.from_atoms(atoms, N=6, species="F")
```

`atoms` is an ASE `Atoms` object for the structure. `N` is the
supercell size, either a scalar (cubic) or a `(Nx, Ny, Nz)` tuple
(orthorhombic). `species` is the chemical symbol of the anion to
flag as `1`.

The `SublatticeOccupation` holds the binary occupation in two layouts;
each analysis function takes the one it needs.

### The per-direction views: `occ.x`, `occ.y`, `occ.z`

The properties `.x`, `.y`, `.z` give you the chains of one direction
at a time, each as a 3D NumPy array. The first two axes index a chain
by its position in the lateral plane; the last axis indexes sites
along that chain.

- `occ.x` has shape `(Ny, Nz, Nx)`: `Ny * Nz` chains in the (y, z)
  plane, each of length `Nx`.
- `occ.y` has shape `(Nx, Nz, Ny)`: `Nx * Nz` chains in the (x, z)
  plane, each of length `Ny`.
- `occ.z` has shape `(Nx, Ny, Nz)`: `Nx * Ny` chains in the (x, y)
  plane, each of length `Nz`.

These arrays are the inputs for the per-chain analysis functions --
`chain_fft`, `along_chain_correlation`, `motif_counts`, and
`inter_chain_correlation` -- described below. Each takes one
direction's chains at a time; to analyse chains of a different
direction, pass `occ.y` or `occ.z` in place of `occ.x`.

### The 3D view: `occ.occupation`

`occ.occupation` is a single array containing all three per-direction
layers, shape `(3, Nx, Ny, Nz)`. Axis 0 selects the direction (`0`
for x, `1` for y, `2` for z); the other three axes are the
`(i, j, k)` grid position of the anion site. This is the input for
`structure_factor`, also described below.

## Order parameters

### chain_fft

For every chain, `chain_fft` takes the discrete Fourier transform
of its species sequence, decomposing the chain into contributions
from different periods. `k = 0` is the chain's mean occupancy; a
nonzero coefficient at `k = N_chain / p` picks up any period-`p`
component; higher `k` captures finer variation.

- A chain in which every site carries the flagged species: magnitude
  `1` at `k = 0`, zero elsewhere.
- A period-3 ordered chain (`...O-O-F-O-O-F-...` with F flagged):
  magnitude `1/3` at `k = 0`, `k = N_chain / 3`, and
  `k = 2 N_chain / 3`, zero elsewhere. (The chain is real-valued, so
  the FFT is conjugate-symmetric: `|F(N_chain - k)| = |F(k)|`, and
  peaks away from `k = 0` come in pairs.)
- A period-2 alternating chain (`...O-F-O-F-...`): magnitude `1/2`
  at `k = 0` and `1/2` at `k = N_chain / 2`, zero elsewhere.

Output is a complex array of the same shape as the input. For a
chain at lateral position `(a, b)`, the last axis holds the Fourier
coefficients at `k = 0, 1, ..., N_chain - 1`. `k = N_chain / p` is
an integer only when `N_chain` is divisible by `p`.

### along_chain_correlation

For a pair of sites `r` apart along a chain, how correlated are
their species flags on average? `along_chain_correlation` returns
this as a function of `r`, averaged over all chain positions and
all chains of one direction.

- A random chain (each site independently 0 or 1 with mean `p`):
  `g(0) = p(1 - p)` (the variance), and `g(r) = 0` for all `r > 0`.
- A period-3 ordered chain (`...O-O-F-O-O-F-...` with F flagged):
  `g(r) = 2/9` at `r = 0, 3, 6, ...` (the flagged species recurs
  at those lags) and `g(r) = -1/9` at all other `r`.
- A period-2 alternating chain (`...O-F-O-F-...`): `g(r) = 1/4` at
  even `r` and `g(r) = -1/4` at odd `r`.

Writing `s_i` for the species flag at site `i` of a chain,

    g(r) = <s_i * s_{i+r}> - <s>^2

with the average over all chain positions `i` and all chains. The
`<s>^2` subtraction removes the mean-density baseline. Positive
`g(r)` at nonzero `r` means sites `r` apart tend to share the same
species (clustering); negative `g(r)` means they tend to differ.

Output is a real array of length `N_chain` giving `g(r)` for
`r = 0, 1, ..., N_chain - 1`, with periodic wrap.

### motif_frequencies

Slides a window of length `w` along each chain (with periodic wrap)
and returns the fraction of windows matching each distinct bit
pattern of length `w`. Each pattern is keyed by its bit tuple, e.g.
`(0, 1, 0)` for `OFO`.

Three concrete cases, all with `w = 3`:

- A period-3 ordered chain (`...O-O-F-O-O-F-...`): the three windows
  produced as the window slides are `(0, 0, 1)`, `(0, 1, 0)`, and
  `(1, 0, 0)`, each at frequency `1/3`.
- A period-2 alternating chain (`...O-F-O-F-...`, `N_chain` even):
  windows alternate between `(0, 1, 0)` and `(1, 0, 1)`, each at
  frequency `1/2`.
- A random chain: each pattern with `k_F` Fs has expected frequency
  `p_F^{k_F} * (1 - p_F)^{w - k_F}` (all patterns with the same
  number of Fs have equal expected frequency). Summing those
  frequencies over patterns of equal `k_F` recovers the binomial
  distribution `Binomial(w, p_F)` on F-count.

Returns a dictionary: keys are bit tuples of the patterns that
appear, values are float arrays of shape `(N_lat0, N_lat1)` giving
per-chain frequencies (each in `[0, 1]`). Patterns absent from the
input are not present in the dictionary. Every chain position is
the start of exactly one window, so per-chain frequencies sum to
`1` regardless of `w`.

`window_length` must satisfy `1 <= window_length <= min(N_chain, 62)`.

### inter_chain_correlation

`inter_chain_correlation` measures how the Fourier component at a
chosen period (selected by the `period` argument) of each chain is
aligned with the same component on other chains in the same lateral
plane. The output is a 2D complex array `G(da, db)` indexed by
lateral separation; `|G|` says how strongly the patterns line up at
that separation, and `arg G` says how they are offset.

Three concrete cases, all with `period=3`:

- All chains put the flagged species at the same position within
  each period (so the flagged species forms layers perpendicular to
  the chain direction): `|G(da, db)| ~ 1` and `arg G(da, db) ~ 0`
  for all `(da, db)`.
- Neighbouring chains along one lateral direction have their
  period-3 patterns shifted by a fixed amount along the chain:
  `|G|` stays near 1, and `arg G` picks up a linear gradient along
  that lateral direction.
- Chains are individually period-3 ordered but with no correlation
  between their phases: `|G(da, db)| ~ 0` for all nonzero
  `(da, db)`.

For the formal definition: each chain at lateral position `(a, b)`
has its own Fourier coefficient at the chosen period,
`phi(a, b) = chain_fft(arr)[a, b, N_chain // period]`, whose
amplitude and phase carry the strength and position of the
period-`p` component on that chain. `inter_chain_correlation`
returns the normalised spatial autocorrelation of `phi`:

    G(da, db) = < phi(a, b) * conj(phi(a + da, b + db)) > / < |phi|^2 >

Averages are over all `(a, b)` with periodic wrap; the
normalisation gives `G(0, 0) = 1`. Each chain's contribution is
weighted by `|phi|^2`, so chains that are themselves disordered
contribute little and a fully disordered input gives `|G| ~ 0` off
the origin.

Requires `N_chain` to be divisible by `period` and non-zero
amplitude at the chosen harmonic somewhere in the input.

### structure_factor

The 3D Fourier transform of the full anion sublattice. Peaks in
`|F|` at specific wavevectors correspond to periodic ordering
components of the structure; `|F|^2` is proportional to the
intensity a coherent diffraction experiment would measure at that
wavevector, with unit form factor (no chemical contrast weighting).

- A fully F-occupied anion sublattice: a single peak at `(0, 0, 0)`
  with `F(0, 0, 0) = 3` (three F per unit cell), zero elsewhere.
- Period-3 F ordering on the x-sublattice only (y- and z-sublattices
  contain no F): `F(0, 0, 0) = 1/3` (overall F fraction) and a peak
  of magnitude `1/3` at `(Nx/3, 0, 0)` (and at the conjugate
  `(2 Nx/3, 0, 0)`). No peaks on the `ky` or `kz` axes.
- Period-3 F ordering on all three sublattices, each chain in phase
  along its own direction: peaks of magnitude `1/3` on each
  reciprocal axis at `(Nx/3, 0, 0)`, `(0, Ny/3, 0)`, `(0, 0, Nz/3)`
  (and their conjugates), with `F(0, 0, 0) = 1` (one F per unit cell).

Takes a `SublatticeOccupation` and returns a complex array of shape
`(Nx, Ny, Nz)` in canonical `(kx, ky, kz)` coordinates. The
wavevector at index `(kx, ky, kz)` is `(kx / Nx, ky / Ny, kz / Nz)`
in reciprocal-lattice units.

The calculation is rotation-equivariant -- a lattice-symmetry
rotation of the input structure produces the correspondingly
rotated output. Anisotropy is preserved: chains ordered along one
direction give peaks on the matching reciprocal axis.

## Common patterns

### Trajectory loop

Analyse a sequence of snapshots by looping and building a fresh
`SublatticeOccupation` per frame:

```python
import numpy as np
from ase.io import iread
from chainorder import SublatticeOccupation, order_params

spectra = []
for atoms in iread("trajectory.xyz"):
    occ = SublatticeOccupation.from_atoms(atoms, N=6, species="F")
    spectra.append(order_params.chain_fft(occ.x))

spectra = np.array(spectra)          # (n_frames, Ny, Nz, Nx)
```

If the trajectory shares a fixed lattice (MC species-swap moves,
for example), `from_atoms` caches its position-to-sublattice
mapping and pays the decomposition cost only on the first frame.

### Iterating over directions

Any per-chain observable can be run on each of the three chain
directions by iterating over the views. A list comprehension is
usually the tidiest form:

```python
g_per_direction = [
    order_params.along_chain_correlation(view)
    for view in (occ.x, occ.y, occ.z)
]
# g_per_direction[0], [1], [2] are the per-direction g(r) arrays.
```

For an orthorhombic supercell with different chain lengths the
three outputs have different shapes and cannot be stacked directly;
a list sidesteps that. In the cubic case (`Nx = Ny = Nz`) the
shapes match and `np.stack(g_per_direction)` gives a `(3, N)`
array if you prefer.

