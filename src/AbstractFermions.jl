
abstract type Abstractfermion# <: AbstractVector{ComplexF64}
end

abstract type AbstractFermionfields{NC,Dim}  <: Abstractfermion
end


abstract type Adjoint_fermion{T} <: Abstractfermion
end

struct Adjoint_fermionfields{T} <: Adjoint_fermion{T}
    parent::T
end

function Base.adjoint(F::T) where T <: Abstractfermion
    Adjoint_fermionfields{T}(F)
end

abstract type Shifted_fermionfields{NC,Dim} <: Abstractfermion
end

const default_boundaryconditions = (nothing,[1,-1],nothing,[1,1,1,-1])

include("./AbstractFermions_4D.jl")


function Initialize_pseudofermion_fields(u::AbstractGaugefields{NC,Dim},Dirac_operator::String) where {NC,Dim}
    mpi = u.mpi
    if mpi
        error("mpi = $mpi is not supported")
    else
        if Dim == 4
            if Dirac_operator == "staggered"
                x = Initialize_StaggeredFermion(u)
            elseif Dirac_operator == "Wilson"
                x = Initialize_WilsonFermion(u)
            else
                error("Dirac_operator = $Dirac_operator is not supported")
            end
        else
            error("Dim = $Dim is not supported")
        end
    end

    return x
end

function Initialize_pseudofermion_fields(u::AbstractGaugefields{NC,Dim},parameters) where {NC,Dim}
    mpi = u.mpi
    if mpi
        error("mpi = $mpi is not supported")
    else
        if Dim == 4
            if Dirac_operator == "staggered"
                x = Initialize_StaggeredFermion(u)
            elseif parameters["Dirac_operator"] == "Wilson"
                x = WilsonFermion_4D_wing(parameters,u.NC,u.NX,u.NY,u.NZ,u.NT)
                #x = Initialize_WilsonFermion(u)
            else
                error("Dirac_operator = $Dirac_operator is not supported")
            end
        else
            error("Dim = $Dim is not supported")
        end
    end

    return x
end

function clear_fermion!(F::T) where T <: AbstractFermionfields
    error("clear_fermion! is not implemented in type $(typeof(F)) ")
end

function Base.similar(F::T) where T <: AbstractFermionfields
    error("Base.similar is not implemented in type $(typeof(F)) ")
end

function gauss_distribution_fermion!(F::T) where T <: AbstractFermionfields
    error("gauss_distribution_fermi! is not implemented in type $(typeof(F)) ")
end

function set_wing_fermion!(F::Vector{<: AbstractFermionfields{NC,Dim}},boundarycondition) where {NC,Dim}
    for μ=1:Dim
        set_wing_fermion!(F[μ],boundarycondition)
    end
end

function set_wing_fermion!(F::T,boundarycondition) where T <: AbstractFermionfields
    error("set_wing_fermion!(F,boundarycondition) is not implemented in type $(typeof(F)) ")
end

function set_wing_fermion!(F::AbstractFermionfields{NC,Dim}) where {NC,Dim}
    set_wing_fermion!(F,default_boundaryconditions[Dim])
end

function Dx!(temp::T,U,x,temps)  where T <: AbstractFermionfields
    error("Dx! is not implemented in type $(typeof(temp)) ")
end


function LinearAlgebra.axpby!(a::Number, X::T1, b::Number, Y::T2) where {T1 <: Abstractfermion, T2 <: Abstractfermion}
    error("LinearAlgebra.axpby(! (Y <- a*X + b*Y ) is not implemented in type X:$T1,Y:$T2")
end

function LinearAlgebra.dot(a::T1,b::T2) where {T1 <: Abstractfermion, T2 <: Abstractfermion}
    error("LinearAlgebra.dot is not implemented in type X:$T1,Y:$T2")
end

function get_origin(a::T1) where T1 <: AbstractFermionfields
    error(" get_origin is not implemented in type $(typeof(a)) ")
end