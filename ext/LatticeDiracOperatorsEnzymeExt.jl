module LatticeDiracOperatorsEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices: Wiltinger_derivative!, toann, DiffArg, NoDiffArg, Enzyme_derivative!, fold_halo_to_core_grad!, dSFdU
using LatticeDiracOperators
#import LatticeDiracOperators.Dirac_operators: General_Dirac_operator, DgagD_General_Dirac_operator
using PreallocatedArrays
import Gaugefields.AbstractGaugefields_module: Gaugefields_4D_MPILattice

import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn, Active, Annotation
const ER = Enzyme.EnzymeRules
import LatticeMatrices: LatticeMatrix, Shifted_Lattice, Adjoint_Lattice, delinearize, shiftindices, kernel_clear_4D!, kernel_add_4D!, mul_AshiftB!, mul_shiftAshiftB!, clear_matrix!, add_matrix!
import LatticeDiracOperators.Dirac_operators: WilsonFermion_4D_MPILattice

include("fallbackmacro.jl")
using .EnzymeBFallback
EnzymeBFallback.@gen_enzyme_fallback_for_B WilsonFermion_4D_MPILattice 10

include("AD_generalfermion.jl")

Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, temp, dtemp, args...) =
    Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, args...; temp=temp, dtemp=dtemp)

Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, temp, dtemp, phitemp, dphitemp, args...) =
    Enzyme_derivative!(func, U1, U2, U3, U4, dfdU1, dfdU2, dfdU3, dfdU4, args...; temp=temp, dtemp=dtemp, phitemp=phitemp, dphitemp=dphitemp)

function Enzyme_derivative!(
    func,
    U::Vector{T},
    dfdU, args...;
    temp=nothing,
    dtemp=nothing
) where T
    # NOTE: Vector U input is not supported. Define a function with U1,U2,U3,U4 args for autodiff.
    error("Enzyme_derivative! does not support Vector U input. Please define a function that takes U1, U2, U3, U4 as separate arguments and run autodiff on that.")
end

function Enzyme_derivative!(
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
    #println("Enzyme_derivative! in LatticeMatrices.jl")
    Enzyme.API.strictAliasing!(false)
    # Primary variables: always differentiated
    annU1 = Enzyme.Duplicated(U1, dfdU1)
    annU2 = Enzyme.Duplicated(U2, dfdU2)
    annU3 = Enzyme.Duplicated(U3, dfdU3)
    annU4 = Enzyme.Duplicated(U4, dfdU4)

    # Convert additional arguments
    ann_args = map(toann, args)

    if phitemp !== nothing && dphitemp === nothing
        error("phitemp is set but dphitemp is nothing")
    end

    # Call Enzyme
    if temp === nothing && phitemp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU1,
            annU2,
            annU3,
            annU4,
            ann_args...
        )
    else
        extra_args = Any[]
        if phitemp !== nothing
            push!(extra_args, Duplicated(phitemp, dphitemp))
        end
        if temp !== nothing
            push!(extra_args, Duplicated(temp, dtemp))
        end
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU1,
            annU2,
            annU3,
            annU4,
            ann_args...,
            extra_args...
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    # Halo values are constrained to core values; fold halo gradients back to core.
    fold_halo_to_core_grad!(dfdU1.U)
    fold_halo_to_core_grad!(dfdU2.U)
    fold_halo_to_core_grad!(dfdU3.U)
    fold_halo_to_core_grad!(dfdU4.U)

    # Gradients of Active scalar arguments are returned by Enzyme
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
