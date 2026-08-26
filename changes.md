# Changes

## v1.1.1

- Allow the LatticeMatrices HISQ wrapper to use its generic-color U(N)
  projection for SU(N), while retaining color-dimension consistency checks.
- Accept Wilsonloop.jl v1 in addition to v0.1.5.

## v1.1.0

- Made MPI.jl a weak dependency. Serial CPU and single-GPU applications can
  load and use LDO without installing MPI.jl.
- Added an MPI package extension. Loading MPI.jl activates
  `MPI.COMM_WORLD`; constructing the first MPI-backed fermion field lazily
  initializes MPI unless the application already initialized it explicitly.
- Kept one-rank serial execution selectable with `SerialCommunicator()` even
  after `using MPI`.
- Moved the hand-written legacy MPI fermion fields to
  `ext/deprecated/mpi/`. They are compatibility-only, load after `using MPI`,
  and are scheduled for removal in a future breaking release.
- Split CI coverage into MPI-free core tests and explicit MPI integration
  tests, including the MPI initialization lifecycle.
- Raised the minimum compatible versions to Gaugefields 1.1 and
  LatticeMatrices 1.2 so MPI remains optional throughout the dependency
  graph.

## v1.0.2

- Made `PseudofermionMDAction` automatically honor stout smearing configured
  by `FermiAction(...; covneuralnet=smearing)` for pseudofermion refresh,
  action evaluation, and force evaluation. The force is pulled back to the
  thin links with the same chain rule as the legacy README HMC example.
- Added paired README HMC examples for conventional explicit integration and
  the Gaugefields MD driver, both with and without stout smearing.

## v1.0.1

- Added `PseudofermionMDAction` and `refresh_pseudofermion!` so LDO fermion
  actions can be used directly by Gaugefields' deterministic MD driver,
  `MDActionSet`, and Sexton–Weingarten integrator.
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
