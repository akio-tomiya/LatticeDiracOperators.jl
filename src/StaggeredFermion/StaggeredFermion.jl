import Gaugefields.Temporalfields_module: Temporalfields, unused!, get_temp

abstract type Staggered_Dirac_operators{Dim} <: Dirac_operator{Dim} end

include("./deprecated/StaggeredFermion_4D_wing.jl")
include("./deprecated/StaggeredFermion_4D_nowing.jl")
include("./deprecated/StaggeredFermion_2D_wing.jl")
include("./deprecated/StaggeredFermion_2D_nowing.jl")
include("./deprecated/StaggeredFermion_4D_nowing_mpi.jl")
include("./StaggeredFermion_4D_MPILattice.jl")
include("./deprecated/StaggeredDiracOperator_legacy.jl")

struct DdagD_Staggered_operator{Dim,T,fermion,TF} <: DdagD_operator
    dirac::TF
end

function DdagD_Staggered_operator(
    U::Array{T,1},
    x,
    parameters,
) where {T<:AbstractGaugefields}
    return DdagD_Staggered_operator(Staggered_Dirac_operator(U, x, parameters))
end

function DdagD_Staggered_operator(
    D::Staggered_Dirac_operator{Dim,T,fermion},
) where {Dim,T,fermion}
    return DdagD_Staggered_operator{Dim,T,fermion,typeof(D)}(D)
end


function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:DdagD_Staggered_operator,T3<:AbstractFermionfields}
    temps = A.dirac._temporary_fermi
    temp, it_temp = get_temp(temps)
    @assert typeof(temp) == typeof(x) "DdagD temporary field type must match the source"

    mul!(temp, A.dirac, x)
    mul!(y, A.dirac', temp)

    unused!(temps, it_temp)
    return
end

include("./StaggeredDiracOperator_4D_MPILattice.jl")
include("./HISQDiracOperator_4D_MPILattice.jl")

#include("./deprecated/notused/StaggeredFermion_4D_accelerator.jl")
#include("./deprecated/notused/kernelfunctions/Staggered_jacc.jl")
#include("./deprecated/notused/kernelfunctions/linearalgebra_mul_jacc.jl")
