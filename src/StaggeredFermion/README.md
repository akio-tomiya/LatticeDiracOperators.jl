# Staggered implementation layout

The v1 standard path is the four-dimensional MPILattice implementation backed
by LatticeMatrices:

- `StaggeredFermion_4D_MPILattice.jl`: the staggered field wrapper around
  `LatticeMatrices.LatticeMatrix` (`NC × 1` at each site).
- `StaggeredDiracOperator_4D_MPILattice.jl`: the LDO wrapper around
  `LatticeMatrices.StaggeredDiracOperator4D`.
- `HISQDiracOperator_4D_MPILattice.jl`: the LDO wrapper around the complete
  cached LatticeMatrices HISQ construction and stencil.
- `StaggeredFermion.jl`: the small include/dispatch layer shared by the
  standard and compatibility implementations.
- `deprecated/`: the old 2D, wing, nowing, hand-written MPI, operator, and
  inactive accelerator implementations.

Files under `deprecated/` are still included where they were included before,
so existing type and function names continue to work. They are compatibility
implementations, not v1's default backend.

## Standard construction

Gaugefields v1's high-level API creates `Gaugefields_4D_MPILattice` by default.
The existing LDO API now uses its LatticeMatrices storage directly:

```julia
U = gauge_configuration(
    (8, 8, 8, 8);
    colors=3, halo=1, start=:cold, process_grid=(1, 1, 1, 1),
)
x = Initialize_pseudofermion_fields(U[1], "staggered")

parameters = Dict(
    "Dirac_operator" => "staggered",
    "mass" => 0.1,
    "eps_CG" => 1e-10,
)
D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)
mul!(y, D', x)
DdagD = DdagD_operator(U, x, parameters)
```

Public function names and the string/Dict selection API are unchanged. The
resulting field is `StaggeredFermion_4D_MPILattice`, and `D.D` is the
LatticeMatrices operator.

The historical `Initialize_Gaugefields` API keeps its legacy default in
Gaugefields v1. It reaches the same standard path when called with
`isMPILattice=true`.

The gauge field owns the halo width, process grid, communicator, element type,
and link storage. The staggered field inherits those settings. The historical
`nowing` keyword remains accepted for source compatibility; for this standard
path, `U[1].NDW` determines whether the LatticeMatrix has a halo. Boundary
phases are stored in the fermion LatticeMatrix and must agree with the Dirac
operator's `boundarycondition` parameter.

## Action and force

`FermiAction`, `evaluate_FermiAction`, and `calc_UdSfdU` retain their existing
interfaces. For MPILattice gauge and staggered fields, the force specialization
uses LatticeMatrices shifts, staggered links, matrix products, and halo
management. It does not use the old per-site wing/nowing stencil.

The regression test covers halo widths 0 and 1, forward and adjoint operations,
`D†D`, operator rebuilding with replacement links, one- and two-rank MPI, and
the action/force path. On one rank it also compares the deprecated nowing
operator and its force kernel with the standard LatticeMatrices path, using
the same input solution vector so the comparison is independent of iterative
solver convergence.

## HISQ

HISQ uses the same `StaggeredFermion_4D_MPILattice` field.  It is selected as
a different Dirac discretization rather than as a separate fermion-field type:

```julia
U = gauge_configuration(
    (8, 8, 8, 8);
    colors=3, halo=3, start=:cold, process_grid=(1, 1, 1, 1),
)
x = Initialize_pseudofermion_fields(U[1], "staggered")

parameters = Dict(
    "Dirac_operator" => "HISQ",
    "mass" => 0.1,
    "naik_epsilon" => 0.0,
    "eps_CG" => 1e-10,
)
D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)
mul!(y, D', x)

action = FermiAction(D, Dict("Nf" => 4))
Sf = evaluate_FermiAction(action, U, x)
force = calc_UdSfdU(action, U, x)
```

LatticeMatrices owns the two Fat7 levels, U(N) reunitarization, Lepage
correction, Naik links, Dirac stencil, derived-link cache, and the analytic
`hisq_link_pullback!`. LDO owns the standard operator/action API and converts
the thin-link gradient to its existing `U * (dS/dU)'` force convention. The
ordinary one-link staggered force is never used for a HISQ operator.

The operator supports any `NC>=2` and `halo=0` as a serial/fallback stencil,
but dynamical HISQ force evaluation requires `halo>=3`. The operator cache is
deliberately shared by the lightweight
objects returned from `D(U)` and is intended for serial use within one action;
create a separate Dirac operator when applying the same action concurrently.

`naik_epsilon` belongs to the quark operator and can differ by species.  A
physical rooted `2+1` or `2+1+1` simulation should therefore compose separate
light, strange, and optional charm action terms.  The existing
`examples/HISQ_HMC_4x4.jl` remains an unrooted `GeneralFermionAction` smoke
test and a regression oracle for the AD path, not a physical rooted ensemble
generator.
