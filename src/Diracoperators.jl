import Gaugefields:AbstractGaugefields
import Gaugefields:Verbose_level,Verbose_3,Verbose_2,Verbose_1,println_verbose3
using LinearAlgebra

abstract type Operator#<: AbstractMatrix{ComplexF64}
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

Base.adjoint(A::Adjoint_Dirac_operator) = A.parent

const default_eps_CG = 1e-19
const default_MaxCGstep = 3000


include("./AbstractFermions.jl")
include("./StaggeredFermion/StaggeredFermion.jl")
include("./WilsonFermion/WilsonFermion.jl")
include("./action/FermiAction.jl")

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
    elseif parameters["Dirac_operator"] == "Wilson"
        DdagD_Wilson_operator(U,x,parameters)
    else
        error("$(parameters["Dirac_operator"]) is not supported")
    end
end


function solve_DinvX!(y::T1,A::T2,x::T3) where {T1 <: AbstractFermionfields,T2 <:  Operator, T3 <: AbstractFermionfields}
    error("solve_DinvX!(y,A,x) (y = A^{-1} x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")
end

function solve_DinvX!(y::T1,A::T2,x::T3) where {T1 <: AbstractFermionfields,T2 <:  Dirac_operator, T3 <: AbstractFermionfields}
    bicg(y,A,x;eps=A.eps_CG,maxsteps = A.MaxCGstep,verbose = set_verbose(A.verbose_level)) 
    set_wing_fermion!(y,A.boundarycondition)
end

function solve_DinvX!(y::T1,A::T2,x::T3) where {T1 <: AbstractFermionfields,T2 <:  Adjoint_Dirac_operator, T3 <: AbstractFermionfields}
    bicg(y,A,x;eps=A.parent.eps_CG,maxsteps = A.parent.MaxCGstep,verbose = set_verbose(A.parent.verbose_level)) 
    set_wing_fermion!(y,A.parent.boundarycondition)
end

using InteractiveUtils

function solve_DinvX!(y::T1,A::T2,x::T3) where {T1 <: AbstractFermionfields,T2 <:  DdagD_operator, T3 <: AbstractFermionfields}
    cg(y,A,x;eps=A.dirac.eps_CG,maxsteps = A.dirac.MaxCGstep,verbose = set_verbose(A.dirac.verbose_level))  
    set_wing_fermion!(y,A.dirac.boundarycondition)
end

function LinearAlgebra.mul!(y::T1,A::T2,x::T3) where {T1 <: AbstractFermionfields,T2 <:  Operator, T3 <: AbstractFermionfields}
    error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")
end

function LinearAlgebra.mul!(y::AbstractFermionfields{NC,Dim},A::T,x::AbstractFermionfields{NC,Dim})  where {T <: DdagD_operator,NC,Dim} #y = A*x
    temp = A.dirac._temporary_fermi[5]

    mul!(temp,A.dirac,x)
    mul!(y,A.dirac',temp)

    return
end



function check_parameters(parameters,key,initial)
    if haskey(parameters,key)
        value = parameters[key]
    else
        value = initial
    end
    return value
end

function set_verbose(verbose_level)
    if verbose_level == 1 
        verbose = Verbose_1()
    elseif verbose_level == 2
        verbose = Verbose_2()
    elseif verbose_level == 3
        verbose = Verbose_3()
    else
        error("verbose_level = $verbose_level is not supported")
    end 
    return verbose
end