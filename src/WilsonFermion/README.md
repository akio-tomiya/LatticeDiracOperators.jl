# Wilson implementation layout

The v1 standard path is:

- `WilsonFermion_4D_MPILattice.jl`: the `LatticeMatrices.jl`-backed field.
- `WilsonFermion_improved.jl`: the wrapper around
  the LatticeMatrices Wilson-family operators.
- `WilsonFermion_clover.jl`: the standard wrapper around
  `LatticeMatrices.WilsonDiracCloverOperator4D`.
- `WilsonFermion_4D.jl`: common four-dimensional Wilson interfaces.

Implementations outside this path are kept in `deprecated/` and are still
included for public Julia API compatibility.

## Wilson--clover

For an MPILattice gauge field, the existing string/Dict API selects the
LatticeMatrices-backed clover operator without changing public function names:

```julia
fermion = Initialize_pseudofermion_fields(U[1], "Wilson")
parameters = Dict(
    "Dirac_operator" => "WilsonClover",
    "κ" => 0.12,
    "cSW" => 1.17,
)
D = Dirac_operator(U, fermion, parameters)
mul!(result, D, fermion)
mul!(result, D', fermion)
DdagD = DdagD_operator(U, fermion, parameters)
```

The wrapper uses the explicit-link cached LatticeMatrices entry points, so an
in-place gauge update refreshes the clover field on the next application.
Direct mutations of a link's `.A` storage must follow the LatticeMatrices
contract and call `mark_halo_dirty!`.

The standard force calls LatticeMatrices' analytic
`wilson_clover_link_pullback!` for the explicit-link cached clover application
and returns the same `U * (dS/dU)'` convention as the other LDO actions. The
action-level regression test compares this force with a Lie-direction finite
difference of the complete pseudofermion action. Forward, adjoint, `D†D`, and
force calculations all work without loading an automatic-differentiation
package.

## `linearalgebra_4D.jl`

This is a transitional collection of `LatticeMatrix` methods used by the
MPILattice wrapper. LatticeMatrices v1.1 already provides overlapping generic
implementations for `axpby!`, `dot`, scalar and matrix `mul!`, shifted
operations, and the `Oneγ` spin operators.

The exact duplicates of `axpby!`, scalar `mul!`, and the four-dimensional
`Oneγμ` kernels are kept commented during the v1 transition.  Their active
call sites use the corresponding LatticeMatrices implementations.  The local
code still needed here includes the three-field `add_matrix!` operation, the
LDO-only γ₅ projector, `MMatrix` helpers for `LatticeMatrices.Oneγ`, and
several spinor outer-product/right-multiplication methods.  These should be
replaced by existing LatticeMatrices APIs or moved upstream before this file is
removed.
