# v1 API and compatibility

LatticeDiracOperators v1 uses Gaugefields' `Gaugefields_4D_MPILattice` and
LatticeMatrices storage as its standard backend.  The existing high-level
function names remain the primary API:

```julia
x = Initialize_pseudofermion_fields(U[1], family; kwargs...)
D = Dirac_operator(U, x, parameters)
DdagD = DdagD_operator(U, x, parameters)
action = FermiAction(D, action_parameters)
```

| `Dirac_operator` value | Standard field | LatticeMatrices implementation |
| --- | --- | --- |
| `"Wilson"` | `WilsonFermion_4D_MPILattice` | `WilsonDiracOperator4D` |
| `"WilsonClover"` | `WilsonFermion_4D_MPILattice` | `WilsonDiracCloverOperator4D` |
| `"staggered"` | `StaggeredFermion_4D_MPILattice` | `StaggeredDiracOperator4D` |
| `"HISQ"` | `StaggeredFermion_4D_MPILattice` | cached HISQ construction and stencil |
| `"Domainwall"`, `"MobiusDomainwall"`, `"GeneralizedDomainwall"` | `DomainwallFermion_5D_MPILattice` | five-dimensional domain-wall operators |

`GeneralFermion` and `GeneralFermionAction` are the extension point for a
user-defined `apply_D`/`apply_Ddag` pair. Their automatic force uses the
Enzyme extension. Standard Wilson--clover and HISQ forces use analytic
LatticeMatrices pullbacks.

## Compatibility implementations

Files below each fermion family's `deprecated/` directory remain included in
v1.  Historical concrete types and function names, including
`WilsonFermion_4D_wing` and `Wilson_Dirac_operator_faster`, therefore remain
callable.  They are not selected for MPILattice input and do not define the
v1 backend contract.  New functionality and backend work should target the
standard types in the table above.

No compatibility implementation is removed by the v1 directory reorganization.
The directory boundary records implementation status; it is not a change to
the public function-name contract.
