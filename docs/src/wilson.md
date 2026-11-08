# Wilson and Wilson--clover

The v1 Wilson path uses `WilsonFermion_4D_MPILattice`, whose spinor data is a
four-dimensional `LatticeMatrix` with four spin components. Both Wilson and
Wilson--clover operators reuse this field.

## Wilson operator

```julia
x = Initialize_pseudofermion_fields(U[1], "Wilson")
parameters = Dict{String,Any}(
    "Dirac_operator" => "Wilson",
    "κ" => 0.12,
    "eps_CG" => 1e-10,
    "MaxCGstep" => 3000,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
DdagD = DdagD_operator(U, x, parameters)

y = similar(x)
mul!(y, D, x)
mul!(y, D', x)
solve_DinvX!(y, DdagD, x)
```

For Gaugefields v1 MPILattice links, `D` wraps
`LatticeMatrices.WilsonDiracOperator4D`. The boundary condition defaults to
periodic spatial directions and antiperiodic time,
`[1, 1, 1, -1]`.

## Wilson--clover

Select clover improvement with the operator dictionary; the field remains a
Wilson field.

```julia
clover_parameters = Dict{String,Any}(
    "Dirac_operator" => "WilsonClover",
    "κ" => 0.12,
    "cSW" => 1.17,
    "eps_CG" => 1e-10,
    "MaxCGstep" => 3000,
    "verbose_level" => 0,
)

Dclover = Dirac_operator(U, x, clover_parameters)
mul!(y, Dclover, x)
mul!(y, Dclover', x)
```

LatticeMatrices owns the Wilson stencil, clover field strength, cached
derived data, and adjoint application. Calling `Dclover(Unew)` rebuilds the
wrapper for new links. In-place link updates are detected through the
LatticeMatrices cache epoch; direct low-level mutation of link storage must
follow the LatticeMatrices halo/cache contract.

The force uses LatticeMatrices' analytic `wilson_clover_link_pullback!`, which
propagates through both the Wilson hopping term and the cached four-leaf
clover field strength:

```julia
clover_action = FermiAction(Dclover, Dict("Nf" => 2))
clover_force = calc_UdSfdU(clover_action, U, x)
```

Forward application, adjoint application, `D†D`, inversion, and force
evaluation therefore use the same LatticeMatrices backend without requiring
an automatic-differentiation package.

## Compatibility boundary

The historical `WilsonFermion_4D_wing`, no-wing, hand-written MPI, and
accelerator implementations remain included from `src/WilsonFermion/deprecated`.
They are not selected for Gaugefields v1 MPILattice input and should not be
used as the basis for new backend work.

## Clover initialization and migration

WilsonClover uses the MPILattice representation on CPU, GPU, MPI and GPU+MPI.
Select the JACC backend before allocating fields. MPI applications load and
initialize MPI first and supply their communicator and process grid to
`Gaugefields.gauge_configuration`. A single-rank application can explicitly
use `Gaugefields.SerialCommunicator()`.

```julia
using Gaugefields, LatticeDiracOperators

U = Gaugefields.gauge_configuration((4, 4, 4, 8); colors=3, halo=1)
# For an existing ensemble, load links before constructing dependent objects:
# Gaugefields.load_configuration!(U, "configuration.jld2"; format=:jld2)
bc = [1, 1, 1, -1]  # [1, 1, 1, 0] selects the supported temporal hopping wall
x = Initialize_pseudofermion_fields(U[1], "Wilson"; boundarycondition=bc)
D = Dirac_operator(U, x, Dict{String,Any}(
    "Dirac_operator" => "WilsonClover", "κ" => 0.12, "cSW" => 1.0,
    "boundarycondition" => bc, "method_CG" => "bicgstab",
    "verbose_level" => 0,
))
action = FermiAction(D, Dict("Nf" => 2))
p = Gaugefields.gauge_momenta(U)
force = [similar(link) for link in U]
```

For legacy serial configurations, write a portable file with
`Gaugefields.save_configuration("configuration.jld2", old_U)`, then load it
into the new links as above. Existing Bridge/ILDG ensembles can be loaded
with their corresponding format. Every rank participates in distributed
loads. Recreate fermions, actions, momenta and force workspaces from the new
links before resuming HMC; old runtime objects are not compatible with the
new representation. This is a configuration migration, not an exact replay
of a trajectory checkpoint from the historical Clover implementation.

`Dirac_operator` does not implicitly convert links, since subsequent updates
must remain visible to the operator, Clover cache and force calculation.
Legacy field inputs now report this migration requirement before allocating
an operator. `hasclover=true` is deprecated; on MPILattice inputs it selects
the same implementation as `Dirac_operator="WilsonClover"`.

The supported Wilson parameter is `r=1`. Use `bicg` or `bicgstab` for direct
solves, or the normal operator for CG. The historical
`preconditiond_bicgstab` implementation assumes a unit on-site Wilson block
and is not a Clover Schur solver; it is rejected for Clover. No new Clover
even/odd preconditioner is introduced by this migration.
