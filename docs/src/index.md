```@meta
CurrentModule = LatticeDiracOperators
```

# LatticeDiracOperators.jl v1

LatticeDiracOperators.jl provides lattice Dirac operators, pseudofermion
actions, iterative solvers, and fermion forces for lattice QCD. Version 1 uses
Gaugefields.jl v1 and LatticeMatrices.jl v1.1 as its standard backend.

## Start here

Begin with the [quick start](quickstart.md). It creates Gaugefields v1 links,
constructs the standard LatticeMatrices-backed Wilson field and operator,
applies and solves the operator, and evaluates a pseudofermion action and
force.

~~~julia
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra

U = gauge_configuration(
    (4, 4, 4, 4);
    colors=3,
    halo=1,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)

x = Initialize_pseudofermion_fields(U[1], "Wilson")
parameters = Dict{String,Any}(
    "Dirac_operator" => "Wilson",
    "κ" => 0.12,
    "eps_CG" => 1e-10,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)
~~~

## Standard formulations

| Operator selector | Standard field | Force implementation |
| --- | --- | --- |
| `"Wilson"` | `WilsonFermion_4D_MPILattice` | analytic |
| `"WilsonClover"` | `WilsonFermion_4D_MPILattice` | analytic |
| `"staggered"` | `StaggeredFermion_4D_MPILattice` | analytic |
| `"HISQ"` | `StaggeredFermion_4D_MPILattice` | analytic |
| `"Domainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |
| `"MobiusDomainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |
| `"GeneralizedDomainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |

The MPILattice field is also the standard single-process field. CPU threads,
GPUs, MPI, and multi-GPU execution use the same public operator API; JACC and
`process_grid` select the execution mode.

## Manual structure

- [Wilson and Wilson--clover](wilson.md) describes the Wilson-family wrappers
  and the analytic clover force.
- [Staggered and HISQ](staggered_hisq.md) covers both staggered operators and
  the analytic HISQ pullback.
- [Domain-wall fermions](domainwall.md) covers Shamir, Möbius, generalized
  coefficients, and physical propagator helpers.
- [User-defined operators](generalfermion.md) introduces callback-defined
  `apply_D!` and `apply_Ddag!` operators.
- [Actions, forces, and solvers](actions_forces.md) documents the common
  workflow across formulations.
- [MPI, GPU, and multi-GPU](mpi_gpu.md) explains backend and process-grid
  selection.
- [High-level API parameters](highlevelapi.md) is the dictionary and keyword
  reference.
- [Public v1 API index](publicapi.md) collects the recommended docstrings.
- [Citing LDO](references.md) lists the JuliaQCD and LLVM-level automatic
  differentiation papers.

## Installation

In Julia package mode:

~~~julia
pkg> add Gaugefields LatticeDiracOperators JACC
~~~

Enzyme is optional. Add it for automatic forces for user-defined
`GeneralFermionAction` callbacks:

~~~julia
pkg> add Enzyme
~~~

!!! warning "Pre-v1 compatibility API"
    Historical wing, nowing, hand-written MPI, and accelerator types remain
    callable in v1, but they are compatibility implementations. New programs
    should use Gaugefields' `gauge_configuration` and the standard MPILattice
    fields. See [v1 compatibility boundary](v1_api.md) and
    [legacy API and examples](howtouse.md).
