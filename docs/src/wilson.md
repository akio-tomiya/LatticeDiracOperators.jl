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

## CG action evaluation and solver choice

With `"method_CG" => "cg"`, Wilson and WilsonClover action evaluation solves
`(D' * D) * X = phi` and returns `real(dot(phi, X))`. This uses the same normal
operator as the force calculation. CG is not applied directly to D or D';
use `"bicg"` or `"bicgstab"` for those non-Hermitian solves.

The Wilson/Clover Krylov helpers defer vector-update halo exchanges until
an operator reads the field. Successful solves still return synchronized
solution halos, including with MPI and JACC accelerator storage.

## Point sources, vectors, and Z2 noise

The public v1 Wilson APIs work through JACC on CPU and GPU and use the field's
MPI communicator. For a point source, all ranks execute the same writes:

```julia
source = Initialize_pseudofermion_fields(U[1], "Wilson")
clear_fermion!(source)
# value, color, x, y, z, t, spin (global, one-based coordinates)
setvalue_fermion!(source, 1.0, 1, 1, 1, 1, 1, 1)
set_wing_fermion!(source)
```

Component writes mark halos dirty. Batch any number of writes, then call
`set_wing_fermion!` collectively before using a stencil. The assignment
`source[color, x, y, z, t, spin] = value` follows the same contract.

`substitute_fermion!(source, values)` loads a replicated global vector and
refreshes halos. `convert_to_normalvector(source)` collects the interior into
a host vector on every rank. Both operations are collective and exclude
halos. Wilson vectors use `(color, x, y, z, t, spin)` order, preserving the
historical Wilson convention; GeneralFermion vectors instead use
`(color, spin, x, y, z, t)` order. For example:

```julia
values = zeros(ComplexF64, size(source))
values[1, 1, 1, 1, 1, 1] = 1
substitute_fermion!(source, vec(values))
@assert convert_to_normalvector(source) == vec(values)
Z2_distribution_fermion!(source; seed=0x1234, sweep=0)
```

Z2 noise fills the interior with real `-1` or `1` and refreshes halos. An
explicit seed and sweep produce the same global field on every backend and
MPI decomposition. Without a seed, rank zero draws and broadcasts a fresh one.

Legacy domain-wall fields also accept
`setvalue_fermion!(field, value, color, x, y, z, t, spin, fifth)`.
Its coordinates follow the underlying legacy Wilson slice (local coordinates
on legacy MPI fields); call `set_wing_fermion!` after component writes.
