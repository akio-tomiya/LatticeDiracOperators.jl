# Domain-wall fermions

Shamir, Möbius, and generalized domain-wall operators use one standard field,
`DomainwallFermion_5D_MPILattice`. It stores all fifth-coordinate slices in a
single five-dimensional `LatticeMatrix`; the historical `w[s]` representation
is confined to the compatibility layer.

## Shamir domain wall

```julia
U = gauge_configuration(
    (2, 2, 2, 2);
    colors=3,
    halo=1,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)

L5 = 2
x = Initialize_pseudofermion_fields(U[1], "Domainwall"; L5)
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
value = evaluate_FermiAction(action, U, x)
force = calc_UdSfdU(action, U, x)
```

The gauge field determines the first four process-grid dimensions,
communicator, precision, and halo width. The fifth coordinate is currently
not MPI-distributed.

## Möbius and generalized coefficients

Use `"MobiusDomainwall"` for scalar `b` and `c` coefficients:

```julia
mobius_parameters = merge(
    parameters,
    Dict{String,Any}(
        "Dirac_operator" => "MobiusDomainwall",
        "b" => 1.5,
        "c" => 0.5,
    ),
)
```

Use `"GeneralizedDomainwall"` for fifth-coordinate vectors `as`, `bs`, and
`cs`. Each vector must have length `L5`.

LatticeMatrices defines the generalized five-dimensional operator as

```math
D_5=A\left[I-F_m+D_W(B+C F_m)\right].
```

Thus `as=1`, `bs=(b+c)/2`, and `cs=(b-c)/2` reproduce the scalar Möbius
operator. All three standard domain-wall forces are analytic and do not
require Enzyme.

## Physical propagators

The current physical-source helpers support Shamir coefficients (`b=1`,
`c=1`) on the standard MPILattice field. They import a four-dimensional
source, solve the raw five-dimensional equation, and export the physical
boundary solution.

```julia
propagators = domainwall_physical_point_propagators(
    D,
    x;
    source_position=(1, 1, 1, 1),
)

residual = domainwall_residual_mass_correlator(
    propagators.five_dimensional,
)
```

`residual` contains `PP`, `J5qP`, and their timeslice ratio. Plateau fitting
belongs in the measurement layer rather than the operator helper.
