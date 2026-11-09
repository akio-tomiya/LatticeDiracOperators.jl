# User-defined Dirac operators

`GeneralFermionAction` is the extension point for a Dirac operator that is
most naturally expressed as field operations. The callback may wrap a
LatticeMatrices operator, but it may also define `D` directly from shifts,
link products, spin/flavor matrices, and field additions.

## Global vectors and eigensystems

`length(x)` for a `GeneralFermion` counts the global interior components,
excluding halos. `convert_to_normalvector(x)` gathers a host vector on every
rank in `(color, column, x1, ..., xD)` order. All ranks in the field communicator
must call it together. `substitute_fermion!(x, values)` loads a replicated global
vector into the owning local blocks and refreshes halos; every rank must supply
the same values. Both operations support JACC accelerator storage.

Linear reads `x[i]` are collective and gather the global vector. Linear writes
`x[i] = value` must use the same global index and value on all ranks; only the
owning rank updates its storage through JACC. Call `set_wing_fermion!(x)` after
a sequence of component writes. Use bulk conversion/loading for dense tools
instead of component loops.

`eigensystem(A, x, radius, origin, z2!, randomize!; ...)` supports these fields.
The operator must provide `mul!` and its adjoint, and the randomization callbacks
must produce consistent distributed fields. The method uses distributed JACC
operator applications and vector algebra, with replicated dense moment matrices
and a host SVD. Its dense storage therefore scales with global vector size.
`residual_tolerance` controls eigenpair acceptance (default `1e-8`); `δ` controls
relative singular-value truncation. Contour moments use coordinates centered
at `origin` and scaled by the positive `radius`. Increasing the residual tolerance accepts
less accurate eigenvectors and does not improve their accuracy.

LatticeDiracOperators does not provide a public LOBPCG solver. The unused,
incomplete source prototype was removed in v1.1.6; `eigensystem` remains the
supported contour-based interface described above.

## Callback contract

Define an application function with this signature:

```julia
function apply_D!(
    result,
    U1,
    U2,
    U3,
    U4,
    source,
    fermion_temps,
    gauge_temps,
)
    # Overwrite result here.
    return result
end
```

Define a separate `apply_Ddag!` for a non-Hermitian operator. The two
temporary vectors are preallocated and passed explicitly so the callback can
remain allocation-free and differentiable.

## Minimal shift-defined operator

The following Hermitian example uses only the first link direction:

```julia
const MASS = 1.0
const KAPPA = 0.1

function apply_D!(result, U1, U2, U3, U4, source, fermion_temps, gauge_temps)
    work = fermion_temps[1]

    clear_fermion!(result)
    add_fermion!(result, MASS, source)

    mul_AshiftB!(work, U1, source, (1, 0, 0, 0))
    add_fermion!(result, -KAPPA, work)

    mul_shiftAshiftB!(
        work,
        adjoint(U1),
        source,
        (-1, 0, 0, 0),
        (-1, 0, 0, 0),
    )
    add_fermion!(result, -KAPPA, work)
    return result
end
```

Construct a compatible field and action:

```julia
global_size = (2, 2, 2, 2)
process_grid = (1, 1, 1, 1)

U = gauge_configuration(
    global_size;
    colors=2,
    halo=1,
    start=:cold,
    process_grid,
)

x = GeneralFermion(
    2,
    1,
    global_size,
    process_grid;
    nw=1,
    numtemps=4,
)
gauss_distribution_fermion!(x)
set_wing_fermion!(x)

action = GeneralFermionAction(
    U,
    x,
    apply_D!,
    apply_D!; # This example is Hermitian.
    numtemp=2,
    num=4,
    numg=4,
    numcg=10,
    eps_CG=1e-12,
    verbose_level=0,
)

y = similar(x)
mul!(y, action.DdagD, x)
value = evaluate_FermiAction(action, U, x)
```

Forward callback application and action evaluation do not require Enzyme.
Load Enzyme before asking LDO to differentiate the callback for a gauge force:

```julia
using Enzyme

force = calc_UdSfdU(action, U, x)
```

## Complete examples

The repository contains three executable examples:

- [minimal one-direction callback](https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/examples/GeneralFermion_Quickstart.jl),
- [callback using U1, U2, U3, and U4](https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/examples/GeneralFermion_AllDirections.jl), and
- [Wilson-like spin operator assembled from shifts](https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/examples/GeneralFermion_Shift_AD.jl).

The last example defines distinct forward and adjoint callbacks and compares
the shift-defined route with the standard Wilson structure in regression
tests. It never constructs `WilsonDiracOperator4D` inside the callback.
