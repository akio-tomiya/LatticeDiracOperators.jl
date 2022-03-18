
using Requires


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


function Base.setindex!(x::Adjoint_fermionfields{T},v,i)  where T <: Abstractfermion
    error("type $(typeof(x)) has no setindex method. This type is read only.")
end

function Base.getindex(x::Adjoint_fermionfields{T},i)  where T <: Abstractfermion 
    @inbounds return conj(x.parent[i])
end


function Base.size(x::Adjoint_fermionfields{T})  where T <: Abstractfermion 
    return size(x.parent)
end




abstract type Shifted_fermionfields{NC,Dim} <: Abstractfermion
end

const default_boundaryconditions = (nothing,[1,-1],nothing,[1,1,1,-1])

include("./AbstractFermions_4D.jl")
include("./AbstractFermions_5D.jl")
include("./AbstractFermions_2D.jl")
include("./AbstractFermions_3D.jl")

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin     
        include("./WilsonFermion/WilsonFermion_4D_wing_mpi.jl")   
        include("./WilsonFermion/WilsonFermion_4D_nowing_mpi.jl") 
        include("./DomainwallFermion/DomainwallFermion_5d_wing_mpi.jl")    
        include("./DomainwallFermion/DomainwallFermion_5d_mpi.jl")  
    end

end



function Initialize_pseudofermion_fields(u::AbstractGaugefields{NC,Dim},Dirac_operator::String;L5=2,nowing = false) where {NC,Dim}
    mpi = u.mpi
    if mpi
        if Dim == 4
            if Dirac_operator == "staggered"
                error("Dirac_operator  = $Dirac_operator  is not supported")
                #x = Initialize_StaggeredFermion(u)
            elseif Dirac_operator == "Wilson"
                if nowing
                    x = WilsonFermion_4D_nowing_mpi(u.NC,u.NX,u.NY,u.NZ,u.NT,u.PEs) 
                    #error("Dirac_operator  = $Dirac_operator with nowing = $nowing is not supported")
                else
                    x = WilsonFermion_4D_mpi(u.NC,u.NX,u.NY,u.NZ,u.NT,u.PEs) 
                end
                #x = Initialize_WilsonFermion(u)
            elseif Dirac_operator == "Domainwall"
                @warn "Domainwall fermion is not well tested!!"
                if nowing
                    x = DomainwallFermion_5D_mpi(L5,u.NC,u.NX,u.NY,u.NZ,u.NT,u.PEs,nowing=nowing) 
                else
                    x = DomainwallFermion_5D_wing_mpi(L5,u.NC,u.NX,u.NY,u.NZ,u.NT,u.PEs) 
                end
            else
                error("Dirac_operator = $Dirac_operator is not supported")
            end
        else
            error("Dim = $Dim is not supported")
        end

        #error("mpi = $mpi is not supported")
    else
        if Dim == 4
            if Dirac_operator == "staggered"
                x = Initialize_StaggeredFermion(u)
            elseif Dirac_operator == "Wilson"
                x = Initialize_WilsonFermion(u,nowing = nowing)
            elseif Dirac_operator == "Domainwall"
                @warn "Domainwall fermion is not well tested!!"
                x = Initialize_DomainwallFermion(u,L5,nowing=nowing)
            else
                error("Dirac_operator = $Dirac_operator is not supported")
            end
        elseif Dim == 2
            if Dirac_operator == "staggered"
                x = Initialize_StaggeredFermion(u)
            elseif Dirac_operator == "Wilson"
                x = Initialize_WilsonFermion(u)
            elseif Dirac_operator == "Domainwall"
                @warn "Domainwall fermion is not well tested!!"
                x = Initialize_DomainwallFermion(u,L5)
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
        elseif Dim == 2
                if Dirac_operator == "staggered"
                    x = Initialize_StaggeredFermion(u)
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

function apply_γ5!(F::T) where T <: AbstractFermionfields
    error("apply_γ5! is not implemented in type $(typeof(F)) ")
end

function Base.similar(F::T) where T <: AbstractFermionfields
    error("Base.similar is not implemented in type $(typeof(F)) ")
end

function Base.length(F::T) where T <: AbstractFermionfields
    error("Base.length(F) is not implemented in type $(typeof(F)) ")
end

function substitute_fermion!(a::T1,b::T2) where {T1 <: AbstractFermionfields, T2  <: AbstractFermionfields}
    error("substitute_fermion!(a,b) is not implemented in type $(typeof(a)) and type $(typeof(b)) ")
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
    error("Dx!(temp,U,x,temps) is not implemented in type $(typeof(temp)) ")
end


function LinearAlgebra.axpby!(a::Number, X::T1, b::Number, Y::T2) where {T1 <: Abstractfermion, T2 <: Abstractfermion}
    error("LinearAlgebra.axpby(! (Y <- a*X + b*Y ) is not implemented in type X:$T1,Y:$T2")
end

function LinearAlgebra.dot(a::T1,b::T2) where {T1 <: Abstractfermion, T2 <: Abstractfermion}
    error("LinearAlgebra.dot is not implemented in type X:$T1,Y:$T2")
end

function get_origin(a::T1) where T1 <: AbstractFermionfields
    error("get_origin is not implemented in type $(typeof(a)) ")
end

function initialize_Adjoint_fermion(x::T1) where T1 <: AbstractFermionfields
    error("initialize_Adjoint_fermion is not implemented in type $(typeof(x)) ")
end