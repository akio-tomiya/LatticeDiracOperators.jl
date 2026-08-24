# Public v1 API index

This page collects docstrings for the recommended LDO v1 entry points. The
package intentionally retains a much larger exported compatibility surface;
those names are not part of the v1 backend contract.

## Standard fields and operators

```@docs
Initialize_pseudofermion_fields
WilsonFermion_4D_MPILattice
StaggeredFermion_4D_MPILattice
DomainwallFermion_5D_MPILattice
Dirac_operator
DdagD_operator
```

## Solvers

```@docs
solve_DinvX!
SolverDiagnostics
```

Gaussian initialization is performed with
`gauss_distribution_fermion!(field)`. The method is defined for each standard
field so it can respect the field's storage and halo representation.

## Actions and forces

```@docs
FermiAction
evaluate_FermiAction
gauss_sampling_in_action!
sample_pseudofermions!
calc_UdSfdU
calc_UdSfdU!
PseudofermionMDAction
refresh_pseudofermion!
```

## User-defined operators

```@docs
GeneralFermion
DdagDgeneral
GeneralFermionAction
```

`mul_AshiftB!`, `mul_shiftAshiftB!`, `γ1`, `γ2`, `γ3`, and `γ4` are
re-exported from LatticeMatrices for callback construction. Their low-level
field semantics follow the LatticeMatrices API.

## Domain-wall measurements

```@docs
solve_domainwall_physical_propagator!
domainwall_physical_point_propagators
domainwall_residual_mass_correlator
```

See [Legacy API and examples](howtouse.md) for the pre-v1 concrete types and
low-level exported helpers that remain available for compatibility.
