# Quick start

This example uses the recommended LDO v1 path. Gaugefields creates the links,
and LDO derives the fermion field's LatticeMatrices geometry from `U[1]`.

## Initialize the backend and links

Initialize JACC before loading Gaugefields and LDO so the same script can be
used with a different execution backend later.

```julia
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

U = gauge_configuration(
    (4, 4, 4, 4);
    colors=3,
    halo=1,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)
```

`U` contains the four links. `process_grid=(1, 1, 1, 1)` is an ordinary
single-process run; no separate serial fermion type is needed.

## Construct and apply Wilson D

```julia
x = Initialize_pseudofermion_fields(U[1], "Wilson")
Random.seed!(101)
gauss_distribution_fermion!(x)

parameters = Dict{String,Any}(
    "Dirac_operator" => "Wilson",
    "κ" => 0.12,
    "eps_CG" => 1e-10,
    "MaxCGstep" => 3000,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)
mul!(y, D', x)
```

`D(Unew)` returns the same operator configuration rebuilt for replacement
links. This is the form used internally when actions receive an updated gauge
configuration.

## Solve D†D and evaluate the action

```julia
DdagD = DdagD_operator(U, x, parameters)
diagnostics = solve_DinvX!(y, DdagD, x)

action = FermiAction(D, Dict("Nf" => 2))
action_value = evaluate_FermiAction(action, U, x)
force = calc_UdSfdU(action, U, x)

@assert isfinite(action_value)
@assert all(link -> isfinite(real(dot(link.U, link.U))), force)
```

The Wilson action and force are analytic and do not require Enzyme. See
[Actions, forces, and solvers](actions_forces.md) for the common action API and
[High-level API parameters](highlevelapi.md) for all standard selectors.

## Choose another formulation

The initialization family names and operator selectors are deliberately
separate for HISQ and Wilson--clover because those formulations reuse the
Wilson or staggered field storage:

| Formulation | Field initialization | Operator selector |
| --- | --- | --- |
| Wilson | `"Wilson"` | `"Wilson"` |
| Wilson--clover | `"Wilson"` | `"WilsonClover"` |
| Staggered | `"staggered"` | `"staggered"` |
| HISQ | `"staggered"` | `"HISQ"` |
| Shamir domain wall | `"Domainwall"` | `"Domainwall"` |
| Möbius domain wall | `"MobiusDomainwall"` | `"MobiusDomainwall"` |
| Generalized domain wall | `"GeneralizedDomainwall"` | `"GeneralizedDomainwall"` |
