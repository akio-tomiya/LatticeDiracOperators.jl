import Gaugefields:AbstractGaugefields
using LinearAlgebra

abstract type Operator <: AbstractMatrix{ComplexF64}
end 
        
abstract type Dirac_operator{Dim}  <: Operator
end

abstract type DdagD_operator  <: Operator
end


abstract type Adjoint_Dirac_operator <: Operator
end

function Base.adjoint(A::Dirac_operator{Dim} ) where Dim
    error("Base.adjoint(A::T)  is not implemented in type $(typeof(A))")
end


include("./AbstractFermions.jl")
include("./StaggeredFermion/StaggeredFermion.jl")
include("./WilsonFermion/WilsonFermion.jl")

function Dirac_operator(U::Array{<: AbstractGaugefields{NC,Dim},1},x,parameters) where {NC,Dim} 
    @assert haskey(parameters,"Dirac_operator") "parameters should have Dirac_operator keyword!"
    if parameters["Dirac_operator"] == "staggered"
        Staggered_Dirac_operator(U,x,parameters)
    elseif parameters["Dirac_operator"] == "Wilson"
        Wilson_Dirac_operator(U,x,parameters)
    else
        error("$(parameters["Dirac_operator"]) is not supported")
    end
end

function DdagD_operator(U::Array{<: AbstractGaugefields{NC,Dim},1},x,parameters) where {NC,Dim} 
    @assert haskey(parameters,"Dirac_operator") "parameters should have Dirac_operator keyword!"
    if parameters["Dirac_operator"] == "staggered"
        DdagD_Staggered_operator(U,x,parameters)
    else
        error("$(parameters["Dirac_operator"]) is not supported")
    end
end

function LinearAlgebra.mul!(y::T1,A::T2,x::T3) where {T1,T2 <:  Dirac_operator, T3}
    error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")
end

function LinearAlgebra.mul!(y::T1,A::T2,x::T3) where {T1,T2 <:  Operator, T3}
    error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")
end

function LinearAlgebra.mul!(y::T1,A::T,x::T2)  where {T <: DdagD_operator,T1 <: AbstractFermionfields,T2 <: AbstractFermionfields} #y = A*x
    temp = A.dirac._temporary_fermi[5]
    mul!(temp,A.dirac,x)
    mul!(y,A.dirac',temp)

    return
end



function check_parameters(parameters,key,initial)
    if haskey(parameters,key)
        value = key
    else
        value = initial
    end
    return value
end

