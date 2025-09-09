
using Requires


abstract type Abstractfermion# <: AbstractVector{ComplexF64}
end

function get_myrank(x::T) where {T<:Abstractfermion}
    return 0
end

function get_nprocs(x::T) where {T<:Abstractfermion}
    return 0
end

abstract type AbstractFermionfields{NC,Dim} <: Abstractfermion end


abstract type Adjoint_fermion{T} <: Abstractfermion end

struct Adjoint_fermionfields{T} <: Adjoint_fermion{T}
    parent::T
end

function Base.adjoint(F::T) where {T<:Abstractfermion}
    Adjoint_fermionfields{T}(F)
end


function Base.setindex!(x::Adjoint_fermionfields{T}, v, i) where {T<:Abstractfermion}
    error("type $(typeof(x)) has no setindex method. This type is read only.")
end

function Base.getindex(x::Adjoint_fermionfields{T}, i) where {T<:Abstractfermion}
    @inbounds return conj(x.parent[i])
end


function Base.size(x::Adjoint_fermionfields{T}) where {T<:Abstractfermion}
    return size(x.parent)
end


struct σμν{μ,ν}
    σ::NTuple{4,ComplexF64}
    indices::NTuple{4,Int64}

    function σμν(μ, ν)
        if μ < ν
            facμν = 1
            μ0 = μ
            ν0 = ν
        else
            facμν = -1
            μ0 = ν
            ν0 = μ
        end

        if μ0 == 1 && ν0 == 2
            ϵ = facμν
            σ = (-ϵ, ϵ, -ϵ, ϵ)
            indices = (1, 2, 3, 4)
        elseif μ0 == 1 && ν0 == 3
            ϵ = facμν
            σ = (-im * ϵ, im * ϵ, -im * ϵ, im * ϵ)
            indices = (2, 1, 4, 3)
        elseif μ0 == 1 && ν0 == 4
            ϵ = facμν
            σ = (-ϵ, -ϵ, ϵ, ϵ)
            indices = (2, 1, 4, 3)
        elseif μ0 == 2 && ν0 == 3
            ϵ = facμν
            σ = (-ϵ, -ϵ, -ϵ, -ϵ)
            indices = (2, 1, 4, 3)
        elseif μ0 == 2 && ν0 == 4
            ϵ = facμν
            σ = (im * ϵ, -im * ϵ, -im * ϵ, im * ϵ)
            indices = (2, 1, 4, 3)
        elseif μ0 == 3 && ν0 == 4
            ϵ = facμν
            σ = (-ϵ, ϵ, ϵ, -ϵ)
            indices = (1, 2, 3, 4)
        else
            error("""something is wrong in σμν
                μ,ν : $μ $ν
                """)
        end
        return new{μ,ν}(σ, indices)
    end
    #=



    function σμν(μ,ν)        
        if μ == 2 && ν == 3
            i = 1
            ϵ = 1
            σ = (ϵ,ϵ,ϵ,ϵ)
            indices = (2,1,4,3)
        elseif μ == 1 && ν == 3
            i = 2
            ϵ = -1
            σ = (-im*ϵ,im*ϵ,-im*ϵ,im*ϵ)
            indices = (2,1,4,3)
        elseif μ == 1 && ν == 2
            i = 3
            ϵ = 1 #3,1,2
            σ = (ϵ,-ϵ,ϵ,-ϵ)
            indices = (1,2,3,4)
        elseif μ == 3 && ν == 2
            i = 1
            ϵ = -1 #132
            σ = (ϵ,ϵ,ϵ,ϵ)
            indices = (2,1,4,3)
        elseif μ == 3 && ν == 1
            i = 2
            ϵ = 1 #231
            σ = (-im*ϵ,im*ϵ,-im*ϵ,im*ϵ)
            indices = (2,1,4,3)
        elseif μ == 2 && ν == 1
            i = 3
            ϵ = -1 #321
            σ = (ϵ,-ϵ,ϵ,-ϵ)
            indices = (1,2,3,4)
        end
        return new{μ,ν}(σ,indices)
    end
    =#
end

#=
const σ12 = σμν(1,2)
const σ21 = σμν(2,1)
const σ13 = σμν(1,3)
const σ31 = σμν(3,1)
const σ23 = σμν(2,3)
const σ32 = σμν(3,2)
=#




abstract type Shifted_fermionfields{NC,Dim} <: Abstractfermion end

const default_boundaryconditions = (nothing, [1, -1], nothing, [1, 1, 1, -1])

include("./AbstractFermions_4D.jl")
include("./AbstractFermions_5D.jl")
include("./AbstractFermions_2D.jl")
include("./AbstractFermions_3D.jl")

function __init__()
    #@require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin



    #end

    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("./WilsonFermion/kernelfunctions/Wilson_cuda.jl")
        include("./WilsonFermion/kernelfunctions/linearalgebra_mul_cuda.jl")
    end

    #@require JACC = "0979c8fe-16a4-4796-9b82-89a9f10403ea" begin


    #end

end



function Initialize_pseudofermion_fields(
    u::AbstractGaugefields{NC,Dim},
    Dirac_operator::String;
    L5=2,
    nowing=true, kwargs...
) where {NC,Dim}
    mpi = u.mpi
    if mpi
        if Dim == 4
            if Dirac_operator == "staggered"
                if nowing
                    x = StaggeredFermion_4D_nowing_mpi(u.NC, u.NX, u.NY, u.NZ, u.NT, u.PEs)
                else
                    error(
                        "Dirac_operator  = $Dirac_operator witn nowing = $nowing is not supported",
                    )
                end
                #x = Initialize_StaggeredFermion(u)
            elseif Dirac_operator == "Wilson"
                if nowing
                    x = WilsonFermion_4D_nowing_mpi(u.NC, u.NX, u.NY, u.NZ, u.NT, u.PEs)
                    #error("Dirac_operator  = $Dirac_operator with nowing = $nowing is not supported")
                else
                    x = WilsonFermion_4D_mpi(u.NC, u.NX, u.NY, u.NZ, u.NT, u.PEs)
                end
                #x = Initialize_WilsonFermion(u)
            elseif Dirac_operator == "Domainwall"
                #@warn "Domainwall fermion is not well tested!!"
                if nowing
                    x = DomainwallFermion_5D_mpi(
                        L5,
                        u.NC,
                        u.NX,
                        u.NY,
                        u.NZ,
                        u.NT,
                        u.PEs,
                        nowing=nowing,
                    )
                else
                    x = DomainwallFermion_5D_wing_mpi(
                        L5,
                        u.NC,
                        u.NX,
                        u.NY,
                        u.NZ,
                        u.NT,
                        u.PEs,
                    )
                end
            elseif Dirac_operator == "MobiusDomainwall"
                #@warn "MobiusDomainwall fermion is not well tested!!"
                if nowing
                    x = MobiusDomainwallFermion_5D_mpi(
                        L5,
                        u.NC,
                        u.NX,
                        u.NY,
                        u.NZ,
                        u.NT,
                        u.PEs,
                        nowing=nowing,
                    )
                else
                    x = MobiusDomainwallFermion_5D_wing_mpi(
                        L5,
                        u.NC,
                        u.NX,
                        u.NY,
                        u.NZ,
                        u.NT,
                        u.PEs,
                    )
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
                x = Initialize_StaggeredFermion(u, nowing=nowing)
            elseif Dirac_operator == "Wilson"
                x = Initialize_WilsonFermion(u, nowing=nowing)
            elseif Dirac_operator == "Domainwall"
                #@warn "Domainwall fermion is not well tested!!"
                x = Initialize_DomainwallFermion(u, L5, nowing=nowing)
            elseif Dirac_operator == "MobiusDomainwall"
                #@warn "MobiusDomainwall fermion is not well tested!!"
                x = Initialize_MobiusDomainwallFermion(u, L5, nowing=nowing)
            elseif Dirac_operator == "GeneralizedDomainwall"
                #@warn "GeneralizedDomainwall fermion is not well tested!!"
                x = Initialize_GeneralizedDomainwallFermion(u, L5, nowing=nowing)

            else
                error("Dirac_operator = $Dirac_operator is not supported")
            end
        elseif Dim == 2
            if Dirac_operator == "staggered"
                x = Initialize_StaggeredFermion(u)
            elseif Dirac_operator == "Wilson"
                x = Initialize_WilsonFermion(u)
            elseif Dirac_operator == "Domainwall"
                #@warn "Domainwall fermion is not well tested!!"
                x = Initialize_DomainwallFermion(u, L5)
            elseif Dirac_operator == "MobiusDomainwall"
                @warn "MobiusDomainwall fermion is not well tested!!"
                x = Initialize_MobiusDomainwallFermion(u, L5)
            elseif Dirac_operator == "GeneralizedDomainwall"
                @warn "GeneralizedDomainwall fermion is not tested!!"
                x = Initialize_GeneralizedDomainwallFermion(u, L5, nowing=nowing)
            else
                error("Dirac_operator = $Dirac_operator is not supported")
            end
        else
            error("Dim = $Dim is not supported")
        end
    end

    return x
end

function Initialize_pseudofermion_fields(
    u::AbstractGaugefields{NC,Dim},
    parameters,
) where {NC,Dim}
    mpi = u.mpi
    if mpi
        error("mpi = $mpi is not supported")
    else
        if Dim == 4
            if Dirac_operator == "staggered"
                x = Initialize_StaggeredFermion(u)
            elseif parameters["Dirac_operator"] == "Wilson"
                x = WilsonFermion_4D_wing(parameters, u.NC, u.NX, u.NY, u.NZ, u.NT)
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

function convert_to_normalvector(F::T) where {T<:AbstractFermionfields}
    error("convert_to_normalvector is not implemented in type $(typeof(F)) ")
end

function clear_fermion!(F::T) where {T<:AbstractFermionfields}
    error("clear_fermion! is not implemented in type $(typeof(F)) ")
end

function apply_γ5!(F::T) where {T<:AbstractFermionfields}
    error("apply_γ5! is not implemented in type $(typeof(F)) ")
end

function Base.similar(F::T) where {T<:AbstractFermionfields}
    error("Base.similar is not implemented in type $(typeof(F)) ")
end

function Base.length(F::T) where {T<:AbstractFermionfields}
    error("Base.length(F) is not implemented in type $(typeof(F)) ")
end

function substitute_fermion!(
    a::T1,
    b::T2,
) where {T1<:AbstractFermionfields,T2<:AbstractFermionfields}
    error(
        "substitute_fermion!(a,b) is not implemented in type $(typeof(a)) and type $(typeof(b)) ",
    )
end

function gauss_distribution_fermion!(F::T) where {T<:AbstractFermionfields}
    error("gauss_distribution_fermi! is not implemented in type $(typeof(F)) ")
end

function set_wing_fermion!(
    F::Vector{<:AbstractFermionfields{NC,Dim}},
    boundarycondition,
) where {NC,Dim}
    for μ = 1:Dim
        set_wing_fermion!(F[μ], boundarycondition)
    end
end

function set_wing_fermion!(F::T, boundarycondition) where {T<:AbstractFermionfields}
    error("set_wing_fermion!(F,boundarycondition) is not implemented in type $(typeof(F)) ")
end

function set_wing_fermion!(F::AbstractFermionfields{NC,Dim}) where {NC,Dim}
    set_wing_fermion!(F, default_boundaryconditions[Dim])
end

function Dx!(temp::T, U, x, temps) where {T<:AbstractFermionfields}
    error("Dx!(temp,U,x,temps) is not implemented in type $(typeof(temp)) ")
end


function LinearAlgebra.axpby!(
    a::Number,
    X::T1,
    b::Number,
    Y::T2,
) where {T1<:Abstractfermion,T2<:Abstractfermion}
    error("LinearAlgebra.axpby(! (Y <- a*X + b*Y ) is not implemented in type X:$T1,Y:$T2")
end

function LinearAlgebra.dot(a::T1, b::T2) where {T1<:Abstractfermion,T2<:Abstractfermion}
    error("LinearAlgebra.dot is not implemented in type X:$T1,Y:$T2")
end

function get_origin(a::T1) where {T1<:AbstractFermionfields}
    error("get_origin is not implemented in type $(typeof(a)) ")
end

function initialize_Adjoint_fermion(x::T1) where {T1<:AbstractFermionfields}
    error("initialize_Adjoint_fermion is not implemented in type $(typeof(x)) ")
end

#=
function apply_σμν!(a::T1,μ,ν,b::T2) where {T1<:Abstractfermion,T2<:Abstractfermion}
    error("apply_σμν! is not implemented in type a:$(typeof(a)),b:$(typeof(b))")
end
=#

function apply_σ!(a::T1, σ::σμν{μ,ν}, b::T2; factor=1) where {μ,ν,T1<:Abstractfermion,T2<:Abstractfermion}
    error("apply_σ! is not implemented in type a:$(typeof(a)),b:$(typeof(b))")
end

function mul_x1plusγμ!(y::Abstractfermion, x, μ)
    if μ == 1
        mul_1minusγ1x!(y, x)
    elseif μ == 2
        mul_1plusγ2x!(y, x)
    elseif μ == 3
        mul_1minusγ3x!(y, x)
    elseif μ == 4
        mul_1plusγ4x!(y, x)
    end
end

function mul_x1minusγμ!(y::Abstractfermion, x, μ)
    if μ == 1
        mul_1plusγ1x!(y, x)
    elseif μ == 2
        mul_1minusγ2x!(y, x)
    elseif μ == 3
        mul_1plusγ3x!(y, x)
    elseif μ == 4
        mul_1minusγ4x!(y, x)
    end
end
