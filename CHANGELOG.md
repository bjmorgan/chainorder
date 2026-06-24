# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [0.2.0] - 2026-06-24

### Added

- `order_params.circulation_invariants`: the <111> circulation order
  parameter. Returns a `CirculationInvariants(chirality, coherence)` named
  tuple -- the configurational chirality (a pseudoscalar whose sign is the
  screw sense) and the coherence (ordering strength) of the <111> anion
  ordering of a `SublatticeOccupation`, summed over the four <111> arms.
  Requires a cubic supercell and a keyword-only `period` (an integer >= 2
  dividing N).

## [0.1.0] - 2026-04-15

### Added

- Initial release.
- `SublatticeOccupation.from_atoms`: decompose an on-lattice ReO3-type
  supercell (orthorhombic, ASE `Atoms` input) into a per-sublattice anion
  occupation.
- Chain-axis order parameters in `order_params`: `chain_fft`,
  `motif_frequencies`, `along_chain_correlation`, and
  `inter_chain_correlation`.
- The cross-sublattice `order_params.structure_factor`.
