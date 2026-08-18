# Staggered and HISQ

The one-link staggered and HISQ operators share
`StaggeredFermion_4D_MPILattice`, a LatticeMatrices field with one internal
fermion component per color. HISQ is an operator/action choice, not a separate
field type.

## One-link staggered operator

```julia
x = Initialize_pseudofermion_fields(U[1], "staggered")
parameters = Dict{String,Any}(
    "Dirac_operator" => "staggered",
    "mass" => 0.1,
    "eps_CG" => 1e-10,
    "MaxCGstep" => 3000,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)
mul!(y, D', x)
```

The fermion field and operator boundary phases must agree. To use a
non-default phase, pass `boundarycondition` while initializing the field and
constructing the operator.

## HISQ

HISQ requires `NC=3`. Use a halo of at least three for dynamical force
calculations:

```julia
U_hisq = gauge_configuration(
    (4, 4, 4, 4);
    colors=3,
    halo=3,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)

x_hisq = Initialize_pseudofermion_fields(U_hisq[1], "staggered")
hisq_parameters = Dict{String,Any}(
    "Dirac_operator" => "HISQ",
    "mass" => 0.4,
    "naik_epsilon" => -0.083,
    "eps_CG" => 1e-8,
    "MaxCGstep" => 2000,
    "verbose_level" => 0,
)

D_hisq = Dirac_operator(U_hisq, x_hisq, hisq_parameters)
y_hisq = similar(x_hisq)
mul!(y_hisq, D_hisq, x_hisq)

action_hisq = FermiAction(D_hisq, Dict("Nf" => 4))
value_hisq = evaluate_FermiAction(action_hisq, U_hisq, x_hisq)
force_hisq = calc_UdSfdU(action_hisq, U_hisq, x_hisq)
```

The standard HISQ force uses the analytic LatticeMatrices pullback.
LatticeMatrices owns both Fat7 levels, U(3) reunitarization, the Lepage
correction, Naik links, the cached stencil, and the thin-link pullback. LDO
converts the thin-link gradient to its standard force convention.

`halo=0` is supported for serial/fallback operator application, but the
dynamical HISQ force requires `halo>=3`. The derived-link cache is shared by
lightweight operators returned from `D_hisq(Unew)`; construct separate Dirac
operators for concurrent actions.

## Flavors and rooting

`FermiAction(D, Dict("Nf" => Nf))` selects the staggered flavor count. The
direct paths for `Nf=4` and `Nf=8` avoid rational approximation; other counts
use RHMC coefficients.

`naik_epsilon` belongs to a quark species. A physical rooted `2+1` or `2+1+1`
setup should compose separate light, strange, and optional charm action terms
rather than treating one example action as a complete ensemble definition.
