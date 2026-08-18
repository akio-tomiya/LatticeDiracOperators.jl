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
