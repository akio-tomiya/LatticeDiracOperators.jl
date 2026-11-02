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

The physical-source helpers support Shamir, scalar Möbius, and slice-dependent
generalized coefficients on the standard MPILattice field. They embed a
four-dimensional source `eta`, apply `A*(I-C*D_W)` to the embedded source,
solve the raw five-dimensional equation, and export the physical boundary
solution. All `as` must be nonzero. The row factor `A` cancels from the solution.
Only Shamir (`bs=1, cs=0`, or scalar `b=c=1`) has `Dminus=I`.

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

The source convention follows the equations in Grid's
[Cayley implementation](https://github.com/paboyle/Grid/blob/0ac72cb6a30ccdc41d664e7e0759f0c8833078f1/Grid/qcd/action/fermion/implementation/CayleyFermion5DImplementation.h):
`ImportPhysicalFermionSource`, `Dminus`, and `ContractJ5q`. The existing fifth
coordinate and gamma-matrix conventions of LatticeMatrices are retained.

## Updating coefficients and differentiating residual mass

`Renew` copies new coefficient vectors while sharing the original gauge links
and scratch pools. Both the physical and Pauli--Villars EO inverses are rebuilt.
Old and renewed objects cannot be used concurrently because their scratch is shared.

```julia
generalized = Dirac_operator(U, x, merge(parameters, Dict(
    "Dirac_operator" => "GeneralizedDomainwall",
    "as" => ones(L5), "bs" => fill(1.5, L5), "cs" => fill(0.5, L5),
    "evenodd" => true,
)))
action = FermiAction(generalized, Dict())

# Select the measurement window explicitly; index 1 is zero separation.
measurement = domainwall_residual_mass_gradient(generalized, x; timeslices=[2])
rate = 1e-3
updated = Renew(generalized,
    generalized.bs - rate * measurement.db,
    generalized.cs - rate * measurement.dc)
updated_action = Renew(action, updated)
```

The differentiated scalar is the ratio of sums over the specified time window:

```math
R=\frac{\sum_{t\in T}J_{5q}P(t)}{\sum_{t\in T}PP(t)}.
```

An adjoint solve differentiates both `D5` and `A*(I-C*D_W)*eta`. In particular,
the `cs` derivative includes the source derivative. There are twelve primal
point-source solves and twelve adjoint solves, independent of `L5`. The method
supports real, nonuniform `bs/cs`, even `L5`, CPU threads, CUDA through JACC,
and MPI. It returns `value`, `db`, `dc`, the timeslice `correlators`, and solver
diagnostics. It performs no optimizer update or HMC accept/reject decision.

For the objective `L5*R^2/2`, multiply both gradients by `L5*R` before passing
them to an optimizer. If `bs-cs` is constrained to be fixed, the derivative
for a shared increment of `bs[s]` and `cs[s]` is `db[s]+dc[s]`.
Apply learned coefficients between trajectories and keep them fixed throughout
each HMC trajectory. This point-source ratio differs from the legacy stochastic
`N/D-mass` objective; old measurements and timings cannot be interpreted as
measurements of this new observable.
