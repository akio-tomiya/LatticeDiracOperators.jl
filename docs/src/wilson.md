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
