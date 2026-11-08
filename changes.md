# Changes

## v1.1.5

### RHMC coefficients and AlgRemez startup

- Correct the swapped positive/negative one-eighth-power RHMC tables at
  order 10 and the negative one-eighth-power inverse-table selection at
  orders 10 and 15. This restores the intended rational approximation for
  the Nf=1 staggered molecular-dynamics force.
- Generate positive and negative half-power coefficients at order 15 with
  AlgRemez instead of referencing undefined tables.
- Fix AlgRemez startup on macOS by preserving the JLL executable environment.
  The previous wrapper retained only the executable path, dropping the
  library search environment needed to locate `libmpfr.6.dylib`.
- Run each coefficient generation in a private temporary directory so
  concurrent calls and MPI ranks cannot overwrite each other's `approx.dat`
  or `error.dat`, and existing files in the caller's directory are preserved.

### GeneralFermion vectors and Sakurai-Sugiura eigensystems

- Add global-vector conversion, bulk loading, linear component access, and
  `axpy!` for GeneralFermion. Field operations use JACC and the existing MPI
  communicator, supporting CPU, GPU, MPI, and MPI+GPU without adding a
  CUDA-specific library dependency. Global vector reads are collective;
  all ranks must supply identical global vectors or component writes.
  See the [GeneralFermion guide](docs/src/generalfermion.md#global-vectors-and-eigensystems)
  for the communication and halo-update contract.
- Support GeneralFermion in Sakurai-Sugiura eigenvector reconstruction.
  Dense moment matrices and the SVD remain on the host and are replicated
  on each MPI rank.
- Add a positive `residual_tolerance` keyword (default `1e-8`) for eigenpair
  acceptance. Honor the requested relative SVD cutoff `δ`, exclude rejected
  singular values, and center and scale contour moments to improve numerical
  conditioning. The tolerance controls acceptance, not eigenvector accuracy.
- Correct the shifted BiCG residual bound to use elementwise reciprocals,
  preventing premature convergence when different shifts converge unevenly.
  Defer internal GeneralFermion halo exchanges until operator application
  or solution return while preserving public vector-update behavior.
- Check GeneralFermion DdagD boundary phases directly, avoiding an invalid
  assumption about a wrapped `dirac` member and restoring the non-RHMC
  staggered solve path.
- Add regressions for coefficient tables and generation, concurrent calls,
  staggered sampling/actions against dense references, force finite
  differences, distributed vector operations, shifted-solve residuals,
  and eigensystems.

### WilsonClover migration

- Consolidate WilsonClover construction and forces on LatticeMatrices. The
  deprecated `hasclover=true` spelling routes MPILattice fields to the same
  implementation. Legacy gauge/fermion combinations fail before allocation
  with instructions for rebuilding the entire configuration; no hidden link
  copies are created. Historical Clover action objects are also rejected.
- Document explicit initialization and portable configuration migration in
  the [Clover migration guide](docs/src/wilson.md#clover-initialization-and-migration).
  Reject unsupported `r != 1` and the Wilson unit-diagonal even/odd solver
  rather than silently applying an incompatible Clover operator.
- Add SU(2)/SU(3) migration, solver, repeated-update and force regressions for
  cSW = -1, 0, +1 and antiperiodic/zero boundaries.

### Wilson/Clover boundary conditions

- Accept `0` alongside `+1` and `-1` in Wilson/Clover MPILattice boundary
  conditions with the corresponding LatticeMatrices zero-phase support.
  A zero phase suppresses hopping across that global boundary in both
  directions while retaining the original gauge links and Clover term.
  Extend independent stencil and solver regressions to temporal walls and
  intersecting spatial/temporal walls; field/operator mismatches still fail
  before output mutation or scratch acquisition.

- Forward Wilson field initialization keywords to the MPILattice constructor,
  preserving explicitly requested periodic or antiperiodic boundary conditions.
  Wilson and WilsonClover operators inherit the field phases when no boundary
  condition is specified and reject inconsistent parameters, input fields, or
  output fields before multiplication or solving. Invalid boundary signs are
  rejected without borrowing solver workspaces. Correct the `nowing` assertion
  message. Regression tests compare forward, adjoint, normal operators, and
  solves with an independent position-space stencil across boundary choices.

### Temporary-field cleanup

- Fix temporary-field ownership gaps identified while reviewing PR #80 and
  PR #81, without importing the historical Clover implementation from PR #80.
  BiCG and BiCGStab now release partially acquired scratch fields when setup,
  operator application, or finalization throws. Wilson/Clover action evaluation
  and force accumulation release their borrowed fermion and gauge fields on
  errors and solver nonconvergence, including failed batch acquisitions.
  Failed calls can be retried with the same pools; fields already borrowed by
  callers remain reserved. Regression tests cover failure injection, bounded
  pool reuse, and successful solves and force evaluations.

### Legacy Wilson gamma5 fixes

- Incorporates [PR #77](https://github.com/akio-tomiya/LatticeDiracOperators.jl/pull/77)
  to correct the color index in the legacy 2D Wilson `mul_γ5x!` and `apply_γ5!` helpers.
  Both gamma5 operations now apply the expected spin-dependent sign to every
  color component without an undefined-index error. Regression tests cover all
  stored components, including halos, and are included in the package and
  isolated core test runners.
- Incorporates [PR #79](https://github.com/akio-tomiya/LatticeDiracOperators.jl/pull/79)
  to correct the color index in the legacy 4D wing Wilson `apply_γ5!` helper.
  The in-place gamma5 operation now applies the expected spin-dependent sign to
  every color component without an undefined-index error. Regression tests cover all
  stored components, including halos, and are included in the package and
  isolated core test runners.

### Generalized domain-wall sources and gradients

- Incorporates [PR #105](https://github.com/akio-tomiya/LatticeDiracOperators.jl/pull/105)
  for generalized domain-wall physical sources and residual-mass coefficient
  gradients on CPU, GPU, and MPI, including multi-GPU calculations.
- Extended physical-source import and propagator helpers from Shamir to
  scalar Mobius and slice-dependent generalized coefficients, including
  nonuniform fifth-direction scaling.
- Added `domainwall_residual_mass_gradient` to compute derivatives with
  respect to real, nonuniform `bs` and `cs` for even `L5`. Adjoint solves
  include the coefficient dependence of the physical source, enabling
  gradient-based coefficient tuning for a residual-mass ratio over selected
  time slices.
- Added coefficient-aware `Renew` methods for generalized domain-wall
  operators and actions. Updates copy coefficient vectors, reuse gauge
  links and workspaces, and rebuild physical and Pauli-Villars EO caches.
- Fixed generic CG temporary-buffer cleanup on successful convergence,
  nonconvergence, and exceptions.
- Added regression coverage for coefficient renewal, physical sources,
  finite-difference gradients, Grid reference correlators, and CG cleanup
  to the standard, core, and MPI test runners.

## v1.1.4

### Generalized domain-wall even/odd preconditioning

- Added opt-in four-dimensional even/odd (EO) preconditioning for
  LatticeMatrices-backed `GeneralizedDomainwall` operators through
  `"evenodd" => true`, including nonuniform fifth-direction coefficients
  `as`, `bs`, and `cs`.
- Implemented the normalized even Schur operator and its adjoint, diagonal
  block inverses, and full-solution reconstruction for forward and adjoint
  solves with CG, BiCG, and BiCGStab. Public operator multiplication retains
  the full domain-wall operator. CG is also supported for non-EO solves.
- Used the same Schur determinant ratio for pseudofermion refresh, action
  evaluation, analytic gauge forces, and momentum updates. The omitted
  diagonal determinant factors are independent of the gauge field.
- Reduced intermediate halo synchronization in EO hopping and Krylov vector
  operations while keeping public matvec and successful solve outputs
  synchronized. MPI parity handling supports odd local extents when the
  global physical extents are even.
- Retained full field storage and the existing default without EO. EO
  requires even global extents in all four physical directions, a fully
  local fifth direction, and a fifth-direction boundary phase of one.

### Regression tests and validation

- Consolidated CPU, GPU, and MPI numerical checks in `test/domainwall_eo.jl`,
  covering block inverses, Schur operators, solver reconstruction,
  pseudofermion actions, force finite differences, and halo correctness.
  MPI tests compare distributed results with an undistributed reference.
- Added 44 independent dense-reference assertions for single-process runs,
  covering the Schur operator and its adjoint, full forward/adjoint
  solutions, the action, and gauge independence of the omitted determinant
  factors for colors 1 and 3, fifth extents 1 and 2, nonuniform coefficients,
  and a complex boundary phase.
- Added `test/eo/run_cuda.jl` to require a functional CUDA backend, disable
  scalar indexing, verify GPU storage, and report the selected device after
  MPI rank-to-device assignment.
- Passed 570/570 assertions with the integrated CPU runner and passed the
  common runner on both CPU MPI ranks. Verified that the CUDA entry point
  rejects the Threads backend.
- Earlier H100 NVL validation of the same production implementation passed
  534/534 single-GPU assertions and 44/44 separate dense-reference
  assertions. Two MPI ranks sharing one H100 passed 489/489 and 465/465
  assertions with host-staged communication. These results predate the
  consolidated CUDA entry point; distinct-GPU and device-direct MPI
  communication were not exercised.
- These focused checks do not cover the full package suite, long HMC
  trajectories, or production performance. See
  [the EO test documentation](test/eo/README.md) for configurations,
  commands, numerical errors, and validation dates.

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
