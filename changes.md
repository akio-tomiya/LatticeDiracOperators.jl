# Changes

## v1.0.1

- Added `PseudofermionMDAction` and `refresh_pseudofermion!` so LDO fermion
  actions can be used directly by Gaugefields' deterministic MD driver,
  `MDActionSet`, and Sexton–Weingarten integrator.
- Made `PseudofermionMDAction` automatically honor stout smearing configured
  by `FermiAction(...; covneuralnet=smearing)` for pseudofermion refresh,
  action evaluation, and force evaluation. The force is pulled back to the
  thin links with the same chain rule as the legacy README HMC example.
- Corrected Z4 noise from eighth-circle phases to the exact roots
  `{1, im, -1, -im}`.
- Fixed the invalid two-dimensional Z4 field indexing.
- Added seeded, backend-independent Z4 generation for the standard Wilson,
  staggered, and domain-wall LatticeMatrices fields.
- Added explicit global-site RNG stream keywords to standard Gaussian
  pseudofermion refresh, giving reproducible fields across MPI decompositions
  and CPU/GPU backends.
- Raised the Gaugefields requirement to v1.0.5, where the public MD-provider
  interface is available.
