# Changes

## v1.1.3

### Native-smeared fermion HMC

- Added `SmearedFermiAction` for evaluating a fermion action on Gaugefields'
  native smeared links and analytically pulling its force back to the thin
  links.
- Supported stout/EXP, HEX, nHYP, and polar-projected APE/HYP, including
  repeated complete smearing steps. Bridge++-compatible MaxReTr APE/HYP
  remains available for forward calculations but is rejected by the HMC
  wrapper with an error directing users to `projection=:polar`.
- Added `NHYPSmearedFermiAction` and the corresponding
  `refresh_nhyp_pseudofermions!` convenience API. The generic
  `refresh_smeared_pseudofermions!` API handles all differentiable native
  smearings and reuses the workspace owned by `md_driver`.

### Analytic staggered and HISQ actions

- Added `StaggeredFermiAction(U, fermion; mass, Nf, discretization=...)` for
  LatticeMatrices-backed fields. `discretization=:staggered` selects the
  one-link operator and `:hisq` selects cached full HISQ.
- Added dedicated analytic one-link and HISQ forces. The HISQ force pulls
  back through both Fat7 levels, U(3) reunitarization, and the Naik term;
  neither action requires Enzyme.
- Supported `Nf=1`, `2`, `4`, and `8`, using RHMC for the rooted `Nf=1` and
  `Nf=2` cases.
- Added complete nHYP-staggered and `4^4` HISQ HMC examples, plus comparison
  and finite-difference regression tests.

### Compatibility and validation

- Raised the minimum compatible versions to Gaugefields 1.1.5 and
  LatticeMatrices 1.2.5, where the native smearing and pullback APIs are
  available.
- Declared the standard-library `Logging` dependency used by the MD-action
  regression as a test extra so `Pkg.test` works in an isolated environment.
- Passed 53 focused CPU checks for analytic staggered/HISQ actions, nHYP and
  generic native-smearing MD providers, and the complete HISQ HMC example
  against the registered Gaugefields 1.1.5 and LatticeMatrices 1.2.5.
- Passed the complete isolated CPU package suite (448/448 checks) with the
  same registered dependency versions.

## v1.1.2

- Added `reset_trajectory_state!(::PseudofermionMDAction)` as the public hook
  for clearing action-owned chronological solver guesses at trajectory
  boundaries. Restarting applications no longer need to inspect LDO action
  fields or Gaugefields temporary pools.
- Clear pseudofermion refresh destinations before sampling. This removes the
  dependence of standard and Möbius domain-wall refreshes on previous or
  uninitialized field contents and enables bitwise trajectory restart.
- Fixed leaked temporary-pool tokens in Wilson and domain-wall action
  constructors, eliminating repeated `All blocks are used` warnings and
  unintended growth of `PreallocatedArray` pools.
- Made pretabulated RHMC coefficient construction quiet by default. Pass
  `verbose=true` to `RHMC` to print the complete coefficient tables.

## v1.1.1

- Allow the LatticeMatrices HISQ wrapper to use its generic-color U(N)
  projection for SU(N), while retaining color-dimension consistency checks.
- Accept Wilsonloop.jl v1 in addition to v0.1.5.
- Keep the default full-package test focused on current CPU API regressions.
  The exploratory multi-trajectory legacy HMC suites now live in
  `test/runtests_legacy.jl` and run only in scheduled or manually dispatched
  CI jobs; Enzyme-dependent tests remain covered by the dedicated Enzyme job.

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
