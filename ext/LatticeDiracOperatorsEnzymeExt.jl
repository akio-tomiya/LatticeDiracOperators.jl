module LatticeDiracOperatorsEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices: toann, DiffArg, NoDiffArg, enzyme_duplicated, fold_halo_to_core_grad!, dSFdU
using LatticeDiracOperators
using PreallocatedArrays
import Gaugefields.AbstractGaugefields_module: Gaugefields_4D_MPILattice
import Gaugefields.Temporalfields_module: get_temp, unused!

import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn, Active, Annotation
const ER = Enzyme.EnzymeRules
import LatticeMatrices: LatticeMatrix, Shifted_Lattice, Adjoint_Lattice, delinearize, shiftindices, kernel_clear_4D!, kernel_add_4D!, mul_AshiftB!, mul_shiftAshiftB!, clear_matrix!, add_matrix!
import LatticeDiracOperators.Dirac_operators: WilsonFermion_4D_MPILattice,
    _general_fermion_derivative!

include("fallbackmacro.jl")
using .EnzymeBFallback
EnzymeBFallback.@gen_enzyme_fallback_for_B WilsonFermion_4D_MPILattice 10

include("AD_generalfermion.jl")

@inline _enzyme_workspace(x) = x
@static if VERSION >= v"1.12"
    # Match the Julia 1.12 mixed-activity ABI used by LatticeMatrices v1.1.
    # Tuples keep the workspace shape immutable while their lattice storage is
    # paired with its shadow by `enzyme_duplicated`.
    @inline _enzyme_workspace(x::AbstractVector) = Tuple(x)
end

function _general_fermion_derivative!(
    func,
    U1::T,
    U2::T,
    U3::T,
    U4::T,
    dfdU1::T,
    dfdU2::T,
    dfdU3::T,
    dfdU4::T, args...;
    temp=nothing,
    dtemp=nothing,
    phitemp=nothing,
    dphitemp=nothing
) where {T<:Gaugefields_4D_MPILattice}
    Enzyme.API.strictAliasing!(false)

    # LatticeMatrices owns the Julia-version-specific Enzyme ABI. In
    # particular, Julia 1.12 needs MixedDuplicated for immutable composites.
    annU1 = enzyme_duplicated(U1, dfdU1)
    annU2 = enzyme_duplicated(U2, dfdU2)
    annU3 = enzyme_duplicated(U3, dfdU3)
    annU4 = enzyme_duplicated(U4, dfdU4)

    ann_args = map(toann, args)

    (temp === nothing) == (dtemp === nothing) ||
        throw(ArgumentError("temp and dtemp must either both be set or both be nothing"))
    (phitemp === nothing) == (dphitemp === nothing) ||
        throw(ArgumentError("phitemp and dphitemp must either both be set or both be nothing"))

    ann_phitemp = phitemp === nothing ? () : (
        enzyme_duplicated(
            _enzyme_workspace(phitemp),
            _enzyme_workspace(dphitemp),
        ),
    )
    ann_temp = temp === nothing ? () : (
        enzyme_duplicated(
            _enzyme_workspace(temp),
            _enzyme_workspace(dtemp),
        ),
    )

    result = Enzyme.autodiff(
        Reverse,
        Enzyme.Const(func),
        Active,
        annU1,
        annU2,
        annU3,
        annU4,
        ann_args...,
        ann_phitemp...,
        ann_temp...,
    )

    # Halo values are constrained to core values; fold halo gradients back to core.
    fold_halo_to_core_grad!(dfdU1.U)
    fold_halo_to_core_grad!(dfdU2.U)
    fold_halo_to_core_grad!(dfdU3.U)
    fold_halo_to_core_grad!(dfdU4.U)

    return result
end

function g(χ, U1, U2, U3, U4, η, p, apply, phitemp, temp)
    phitemp1 = phitemp[end]
    apply(phitemp1, U1, U2, U3, U4, η, p, phitemp, temp)
    #Dmul!(phitemp1, U1, U2, U3, U4, D, η)
    #s = -2 * real(dot(_lm_primal(χ), _lm_primal(phitemp1)))
    #s = -2 * real(dot(χ.f, phitemp1.f))
    s = -2 * real(dot(χ, phitemp1))


    return s
end



end
