# LatticeDiracOperators

[![CI](https://github.com/akio-tomiya/LatticeDiracOperators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/LatticeDiracOperators.jl/actions/workflows/CI.yml)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://akio-tomiya.github.io/LatticeDiracOperators.jl/dev/)

LatticeDiracOperators.jl provides lattice Dirac operators, pseudofermion
actions, solvers, and fermion forces for lattice QCD. Version 1 uses
[Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl) v1 and
[LatticeMatrices.jl](https://github.com/cometscome/LatticeMatrices.jl) v1.1
as its standard backend.

The package supports Julia 1.11 and 1.12, threaded CPU execution, MPI domain
decomposition, and the GPU backends provided by JACC and LatticeMatrices.

## What's new in v1

Compared with the 0.6 release line, v1:

- uses the LatticeMatrices-backed MPILattice fields as the standard path;
- provides Wilson, Wilson--clover, staggered, HISQ, Shamir domain-wall,
  Möbius domain-wall, and generalized domain-wall operators;
- keeps historical wing/nowing/accelerator implementations in
  family-specific `deprecated/` directories for source compatibility;
- implements the standard HISQ force analytically, without Enzyme;
- provides Wilson--clover and user-defined callback forces through an optional
  Enzyme extension;
- supports user-defined `apply_D!` and `apply_Ddag!` callbacks through
  `GeneralFermionAction`; and
- tests the core, Enzyme, and two-rank MPI paths separately on Julia 1.11 and
  1.12.

New code should create gauge links with Gaugefields'
`gauge_configuration`. The historical `Initialize_Gaugefields` and LDO
concrete field names remain available, but select the compatibility path when
given legacy gauge fields.

## Install

In Julia package mode:

```text
pkg> add Gaugefields LatticeDiracOperators JACC
```

Enzyme is optional. Add it only for Wilson--clover forces or automatic
differentiation of user-defined `GeneralFermionAction` callbacks:

```text
pkg> add Enzyme
```

Add `MPI` as a direct application dependency when the application itself
imports the MPI API or launches MPI-specific helper code.

## Recommended high-level API

The common workflow is:

1. create gauge links with `gauge_configuration`;
2. create a pseudofermion field with
   `Initialize_pseudofermion_fields`;
3. construct `Dirac_operator` or `DdagD_operator`; and
4. construct `FermiAction` when an action or force is needed.

### Wilson and Wilson--clover

This complete example applies and solves the Wilson operator, evaluates the
pseudofermion action and force, and applies the Wilson--clover operator:

```julia
# README_V1_WILSON
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

DdagD = DdagD_operator(U, x, parameters)
solve_DinvX!(y, DdagD, x)

action = FermiAction(D, Dict("Nf" => 2))
action_value = evaluate_FermiAction(action, U, x)
force = calc_UdSfdU(action, U, x)

@assert isfinite(action_value)
@assert all(link -> isfinite(real(dot(link.U, link.U))), force)

clover_parameters = merge(
    parameters,
    Dict{String,Any}(
        "Dirac_operator" => "WilsonClover",
        "cSW" => 1.17,
    ),
)
Dclover = Dirac_operator(U, x, clover_parameters)
mul!(y, Dclover, x)
@assert isfinite(real(dot(y, y)))
```

Wilson--clover application and inversion do not require Enzyme. Calling
`calc_UdSfdU` for a Wilson--clover action does require `using Enzyme`;
without it, LDO raises an error explaining the optional dependency.

### HISQ without Enzyme

HISQ uses the staggered MPILattice field and requires a halo width of at least
three. Its standard fermion force is an analytic LatticeMatrices pullback:

```julia
# README_V1_HISQ_NO_ENZYME
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

U = gauge_configuration(
    (4, 4, 4, 4);
    colors=3,
    halo=3,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)

x = Initialize_pseudofermion_fields(U[1], "staggered")
Random.seed!(102)
gauss_distribution_fermion!(x)

parameters = Dict{String,Any}(
    "Dirac_operator" => "HISQ",
    "mass" => 0.4,
    "naik_epsilon" => -0.083,
    "eps_CG" => 1e-8,
    "MaxCGstep" => 2000,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)

action = FermiAction(D, Dict("Nf" => 4))
action_value = evaluate_FermiAction(action, U, x)
force = calc_UdSfdU(action, U, x)

@assert isfinite(action_value)
@assert all(link -> isfinite(real(dot(link.U, link.U))), force)
```

The one-link staggered operator uses the same field with
`"Dirac_operator" => "staggered"` and a `"mass"` parameter.

### Domain-wall fermions

Shamir, Möbius, and generalized domain-wall operators share one
`DomainwallFermion_5D_MPILattice` storage type backed by a
`LatticeMatrix{5}`:

```julia
# README_V1_DOMAIN_WALL
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

U = gauge_configuration(
    (2, 2, 2, 2);
    colors=3,
    halo=1,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)

L5 = 2
x = Initialize_pseudofermion_fields(U[1], "Domainwall"; L5)
Random.seed!(103)
gauss_distribution_fermion!(x)

parameters = Dict{String,Any}(
    "Dirac_operator" => "Domainwall",
    "mass" => 0.1,
    "L5" => L5,
    "M" => -1.0,
    "eps_CG" => 1e-8,
    "MaxCGstep" => 1000,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)

action = FermiAction(D, Dict())
action_value = evaluate_FermiAction(action, U, x)

@assert x isa DomainwallFermion_5D_MPILattice
@assert isfinite(real(dot(y, y)))
@assert isfinite(action_value)
```

Use `"MobiusDomainwall"` with scalar `b` and `c`, or
`"GeneralizedDomainwall"` with fifth-coordinate vectors `as`, `bs`,
and `cs`.

## Standard operators

| `Dirac_operator` value | Standard field | Force implementation |
|---|---|---|
| `"Wilson"` | `WilsonFermion_4D_MPILattice` | analytic |
| `"WilsonClover"` | `WilsonFermion_4D_MPILattice` | Enzyme extension |
| `"staggered"` | `StaggeredFermion_4D_MPILattice` | analytic |
| `"HISQ"` | `StaggeredFermion_4D_MPILattice` | analytic, no Enzyme |
| `"Domainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |
| `"MobiusDomainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |
| `"GeneralizedDomainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |

All standard fields store their data in LatticeMatrices. The MPILattice name
also covers a single process: use `process_grid=(1, 1, 1, 1)` for a normal
single-process run.

## User-defined Dirac operators

`GeneralFermionAction` accepts user-defined `apply_D!` and
`apply_Ddag!` callbacks. The examples progress from a minimal definition to
a four-direction stencil:

- [Quick start: define and use `apply_D!`](examples/GeneralFermion_Quickstart.jl)
- [Shift-based operator with automatic differentiation](examples/GeneralFermion_Shift_AD.jl)
- [Operator using `U1`, `U2`, `U3`, and `U4`](examples/GeneralFermion_AllDirections.jl)

The callback force requires Enzyme. Built-in standard operators should use
`Dirac_operator` and `FermiAction` directly unless a custom composition is
needed.

## MPI and GPU execution

The field and operator APIs above do not change with the execution backend.
Choose the JACC backend before constructing fields, and choose the domain
decomposition through `process_grid`. For example, a two-rank decomposition
along the first direction uses `process_grid=(2, 1, 1, 1)`.

Gauge initialization, rank-to-device mapping, and backend-specific setup are
documented by Gaugefields and LatticeMatrices:

- [Gaugefields four-dimensional and multi-GPU tutorial](https://github.com/akio-tomiya/Gaugefields.jl/blob/main/docs/src/tutorial4d.md)
- [Gaugefields randomness and reproducibility guide](https://github.com/akio-tomiya/Gaugefields.jl/blob/main/docs/src/randomness.md)

## Documentation

- [v1 API and compatibility boundary](docs/src/v1_api.md)
- [Operator, action, and historical HMC examples](docs/src/howtouse.md)
- [Wilson implementation notes](src/WilsonFermion/README.md)
- [Staggered and HISQ implementation notes](src/StaggeredFermion/README.md)
- [Domain-wall implementation notes](src/DomainwallFermion/README.md)
- [GeneralFermion callback guide](src/GeneralFermion/README.md)
- [v1 implementation audit](V1_AUDIT.md)

The compatibility files under `deprecated/` are still included in v1, so
historical function and concrete type names remain callable. They are not the
backend contract for new code.

## Questions

Please use this repository's issue tracker or the
[JuliaQCD discussion board](https://github.com/orgs/JuliaQCD/discussions).
Questions in Japanese are welcome.

## Acknowledgment

If you use this package in a paper, please cite:

```bibtex
@article{Nagai:2024yaf,
    author = "Nagai, Yuki and Tomiya, Akio",
    title = "{JuliaQCD: Portable lattice QCD package in Julia language}",
    eprint = "2409.03030",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    month = "9",
    year = "2024"
}
```

The paper is available as [arXiv:2409.03030](https://arxiv.org/abs/2409.03030).
