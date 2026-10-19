# Domain-wall implementation layout

The v1 standard path uses one five-dimensional
`LatticeMatrices.LatticeMatrix` for all three domain-wall variants:

- `DomainwallFermion_5D_MPILattice.jl`: the common field and the
  LatticeMatrix-specific algebra required by the LDO action.
- `DomainwallFermion.jl`: the small compatibility entry point.
- `../MobiusDomainwallFermion/`: the Shamir/Möbius operator wrapper.
- `../GeneralizedDomainwallFermion/GeneralizedDomainwallFermion_5D_MPILattice.jl`:
  the generalized-coefficient wrapper.
- each family's `deprecated/` directory: historical `w[s]`, wing/no-wing,
  and hand-written MPI implementations.

Files under `deprecated/` remain included, so the historical type and
function names continue to work. They are compatibility implementations and
are not the v1 default for `Gaugefields_4D_MPILattice` input.

## Standard construction

Gaugefields v1's high-level API creates `Gaugefields_4D_MPILattice` by
default. The existing LDO string/Dict API selects the LatticeMatrices path
without a backend flag:

```julia
U = gauge_configuration(
    (8, 8, 8, 8);
    colors=3, halo=1, start=:cold, process_grid=(1, 1, 1, 1),
)
L5 = 8
x = Initialize_pseudofermion_fields(U[1], "Domainwall"; L5)

parameters = Dict(
    "Dirac_operator" => "Domainwall",
    "mass" => 0.01,
    "L5" => L5,
    "M" => -1.0,
    "eps_CG" => 1e-10,
)
D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)
mul!(y, D', x)

action = FermiAction(D, Dict())
Sf = evaluate_FermiAction(action, U, x)
force = calc_UdSfdU(action, U, x)
```

The resulting field is `DomainwallFermion_5D_MPILattice` and owns one
`LatticeMatrix{5}`. The compatibility name
`MobiusDomainwallFermion_5D_MPILattice` is an alias of the same type.
The gauge field supplies the first four dimensions' process grid,
communicator, precision, and halo width. The fifth coordinate currently must
use one MPI partition.

The historical `Initialize_Gaugefields` API reaches this path with
`isMPILattice=true`. The old `is5D` and `"improved gpu"` switches are not
needed for MPILattice fields; `"improved gpu" => true` with a legacy field is
rejected rather than silently selecting a mismatched implementation.

## Generalized domain-wall even/odd preconditioning

The standard generalized wrapper supports four-dimensional even/odd (EO)
preconditioning, with the complete fifth direction stored locally:

```julia
L5 = 4
x = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5)
gauss_distribution_fermion!(x; seed=41)
parameters = Dict{String,Any}(
    "Dirac_operator" => "GeneralizedDomainwall",
    "L5" => L5, "mass" => 0.1, "M" => -1.0,
    "as" => ones(L5), "bs" => fill(1.5, L5), "cs" => fill(0.5, L5),
    "evenodd" => true, "method_CG" => "bicgstab",
    "eps_CG" => 1e-18, "MaxCGstep" => 3000, "verbose_level" => 0,
)
D = Dirac_operator(U, x, parameters)
solution = similar(x)
solve_DinvX!(solution, D.D5DW, x)    # Full solution, reconstructed from EO.
solve_DinvX!(solution, D.D5DW', x)   # Adjoint solve.
solve_DinvX!(solution, D, x)        # Full Pauli–Villars-composed operator.

action = FermiAction(D, Dict())
phi, noise = similar(x), similar(x)
provider = PseudofermionMDAction(action, phi)
refresh_pseudofermion!(provider, U, noise; seed=42)
Sf = evaluate_FermiAction(action, U, phi)
force = calc_UdSfdU(action, U, phi)
```

For `D = [E B; C O]` the normalized even Schur operator is
`S = I - E⁻¹ B O⁻¹ C`. Raw `mul!` still applies the full five-dimensional
operator. `solve_DinvX!` uses `S` and reconstructs both parities; `"bicg"`,
`"bicgstab"`, and `"cg"` (normal equations) are supported. Solver diagnostics
refer to the reduced system, so check a full-system residual when needed.
`eps_CG` is an absolute **squared** residual tolerance.

The same `"method_CG" => "cg"` setting is accepted with `"evenodd" => false`,
including the raw forward/adjoint and PV-composed solves. The public name `cg`
covers CGNR for forward solves (`D†D x = D†b`) and CGNE for adjoint solves
(`D†D z = b`, followed by `x = D z`). With EO, these normal equations apply
to the Schur operator before reconstruction. Their stopping residuals differ
from the original full-system residual.

`test/domainwall_eo.jl` compares all three
solvers with and without EO, checks the original-system residuals and the
agreement of their solutions, and reports median solve times and
`non_eo_time / eo_time`. Run it after preparing the environment in `test/eo/README.md`:

```sh
julia --project=test/eo test/domainwall_eo.jl
```

The comparison defaults to a hot `4^4` lattice, `L5=4`, and five samples per
solve. Set `LDO_EO_BENCH_N`, `LDO_EO_BENCH_L5`, or `LDO_EO_BENCH_REPEATS` to
change these values. It uses the active project's JACC backend. Set
`LDO_TEST_MPI=true` under an MPI launcher to split a physical direction across
ranks (`LDO_EO_BENCH_AXIS=1` by default). `LDO_EO_BENCH_N` is the global extent.
Timings synchronize backend work, use an MPI barrier before each sample, and
report the maximum rank time. They exclude operator construction and warmup.
A speed ratio below one means that EO is slower on the measured configuration.

With EO enabled, the action, pseudofermion refresh, gauge derivative, and
momentum update all use the same Schur determinant ratio. Pseudofermions
occupy even sites. Gauge-independent diagonal determinant factors are
omitted from the action. The action's normal equations use CG regardless of
the raw solver selection. Nonuniform `as`, `bs`, and `cs` are supported;
reconstruct the operator when changing these coefficients or masses.

All four global physical extents must be even, `PEs[5]` must be 1, and the
fifth boundary phase must be 1. The implementation retains full field storage
but evaluates hopping only on the requested output parity. SU(3) uses the
v1 half-spinor hopping kernels directly, with a two-stage adjoint that reuses
Wilson hopping across fifth slices. Other colour counts use a projected
source and a parity-restricted generic stencil. Schur applications use two
intermediate fields (plus adjoint/generic hopping scratch), fuse the final
subtraction, and update intermediate halos only before hopping reads.
On one rank, SU(3) hopping wraps the fifth coordinate directly and copies
only the four-dimensional boundary faces it reads. Partial halo updates leave
the full halo marked dirty, so subsequent public stencils still synchronize
all required ghosts. Multiple ranks retain the standard halo exchange.
EO Krylov operations also defer halo synchronization during vector algebra
and internal matvecs; public matvecs and successful solves return clean halos.
The shared solver recurrences and non-EO field operations are unchanged.
This is not a compressed-storage or specialized GPU implementation. The common
`test/domainwall_eo.jl` entry point runs numerical, action/force, halo, and solve
comparisons on the chosen JACC backend and compares decomposed fields with
an undistributed reference. Single-process runs also include an independent
dense-matrix reference for Schur operators, solutions, actions, and determinants.
See [`test/eo/README.md`](../../test/eo/README.md) for CPU, MPI, GPU, and MPI+GPU
commands, the CUDA entry point, and the H100 validation scope. Production
trajectory performance and communication between distinct GPUs require
separate validation.

Omitting `"evenodd"` (or setting it to `false`) retains the existing path.
This switch currently applies to the LatticeMatrices-backed
`"GeneralizedDomainwall"` wrapper.

## Physical point propagators and residual mass

The v1 valence API imports a four-dimensional source onto the Shamir walls,
solves the raw five-dimensional operator, and exports the physical solution.
It deliberately does not call the Pauli--Villars-composed outer operator used
by the pseudofermion action:

```julia
L5 = 4
x5 = Initialize_pseudofermion_fields(U[1], "Domainwall"; L5)
D = Dirac_operator(U, x5, Dict(
    "Dirac_operator" => "Domainwall",
    "mass" => 0.1,
    "L5" => L5,
    "M" => -1.0,                 # Grid M5=1 convention
    "eps_CG" => 1e-28,
    "MaxCGstep" => 100_000,
    "method_CG" => "bicg",
    "verbose_level" => 0,
    "boundarycondition" => [1, 1, 1, -1],
))

propagators = domainwall_physical_point_propagators(
    D, x5; source_position=(1, 1, 1, 1))
correlators = domainwall_residual_mass_correlator(
    propagators.five_dimensional; origin=(1, 1, 1, 1))

correlators.PP
correlators.J5qP
correlators.ratio                 # J5qP ./ PP on each timeslice
```

`ratio` is the raw per-configuration timeslice ratio. A residual-mass result
still requires an ensemble average and a stated plateau fit window. The
definition follows the midpoint axial Ward identity of Furman and Shamir,
[Nucl. Phys. B439 (1995) 54--78](https://doi.org/10.1016/0550-3213(95)00031-M).

The independent regression in `test/domainwall_grid_reference.jl` uses a
`4^4`, `L5=4`, non-cold one-link SU(3) field and compares all timeslices of
`PP`, Grid's `ContractJ5q`, and `J5qP/PP` with Grid commit
`0ac72cb6a30ccdc41d664e7e0759f0c8833078f1`. The same frozen values pass on
threaded CPU, two MPI ranks, and an NVIDIA H100 through JACC/CUDA. The complete
Grid driver and recorded output are under `test/references/grid/`.

## Variants

The public function names and operator strings are unchanged:

- `"Domainwall"`: Shamir domain wall, implemented as the Möbius preset
  `b=1`, `c=1`.
- `"MobiusDomainwall"`: scalar Möbius coefficients `b` and `c`.
- `"GeneralizedDomainwall"`: fifth-coordinate vectors `as`, `bs`, and `cs`.

LatticeMatrices defines the generalized operator as

```math
D_5=A\left[I-F_m+D_W(B+C F_m)\right].
```

Thus `as=1`, `bs=(b+c)/2`, and `cs=(b-c)/2` reproduce the scalar Möbius
operator. The LDO regression test checks this identity for the operator,
action, and analytic gauge force. Slice-dependent `as`, `bs`, and `cs` are
also covered by the action/force test.

## Compatibility boundary

The standard forward, adjoint, `D†D`, action, and `calc_UdSfdU` paths operate
on the whole five-dimensional LatticeMatrix and do not access `w[s]`.
Historical concrete fields and the unused direct-momentum force helpers still
use the old slice representation and remain in the compatibility layer.

The focused regression test passes on one and two MPI ranks. It checks field
selection, boundary/partition metadata, Gaussian initialization, Shamir and
Möbius equivalence, adjointness, `D†D`, all three actions, generalized
coefficients, complex boundary phases, fifth-direction shifts, and finite
analytic forces.
