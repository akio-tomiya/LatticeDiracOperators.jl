struct Staggered_Dirac_operator{Dim,T,fermion} <:
       Staggered_Dirac_operators{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    boundarycondition::Vector{Int8}
    mass::Float64
    _temporary_fermi::Temporalfields{fermion}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}
end

function Staggered_Dirac_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    xtype = typeof(x)
    temporary_fermi = Temporalfields(x; num=6)

    @assert haskey(parameters, "mass") "parameters should have the keyword mass"
    mass = parameters["mass"]
    if Dim == 4
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, 1, 1, -1])
    elseif Dim == 2
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, -1])
    else
        error("Dim should be 2 or 4!")
    end
    eps_CG = check_parameters(parameters, "eps", default_eps_CG)
    MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)
    verbose_level = check_parameters(parameters, "verbose_level", 2)
    method_CG = check_parameters(parameters, "method_CG", "bicg")

    temporary_fermion_forCG = Temporalfields(x; num=8 * 4)
    verbose_print = Verbose_print(verbose_level, myid=get_myrank(x))

    return Staggered_Dirac_operator{Dim,eltype(U),xtype}(
        U,
        boundarycondition,
        mass,
        temporary_fermi,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        verbose_print,
        temporary_fermion_forCG,
    )
end

get_Dim(::Staggered_Dirac_operator{Dim}) where {Dim} = Dim
get_fermiontype(
    ::Staggered_Dirac_operator{Dim,T,fermion},
) where {Dim,T,fermion} = fermion

function (D::Staggered_Dirac_operator{Dim,T,fermion})(U) where {Dim,T,fermion}
    return Staggered_Dirac_operator{Dim,eltype(U),fermion}(
        U,
        D.boundarycondition,
        D.mass,
        D._temporary_fermi,
        D.eps_CG,
        D.MaxCGstep,
        D.verbose_level,
        D.method_CG,
        D.verbose_print,
        D._temporary_fermion_forCG,
    )
end

struct Adjoint_Staggered_operator{T} <: Adjoint_Dirac_operator
    parent::T
end

Base.adjoint(A::Staggered_Dirac_operator) =
    Adjoint_Staggered_operator(A)
Base.adjoint(A::Adjoint_Staggered_operator) = A.parent

function Initialize_StaggeredFermion(
    u::AbstractGaugefields{NC,Dim};
    nowing=false,
) where {NC,Dim}
    _, _, NN... = size(u)
    return Initialize_StaggeredFermion(NC, NN...; nowing)
end

function Initialize_StaggeredFermion(NC, NN...; nowing=false)
    Dim = length(NN)
    if Dim == 4
        return nowing ?
            StaggeredFermion_4D_nowing(NC, NN...) :
            StaggeredFermion_4D_wing(NC, NN...)
    elseif Dim == 2
        return nowing ?
            StaggeredFermion_2D_nowing(NC, NN...) :
            StaggeredFermion_2D_wing(NC, NN...)
    end
    error("Dimension $Dim is not supported")
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields,
    A::Staggered_Dirac_operator,
    x::AbstractFermionfields,
)
    temps = A._temporary_fermi
    temp, temp_token = get_temp(temps)
    tempvec, tempvec_token = get_temp(temps, 3)
    @assert typeof(temp) == typeof(x) "staggered temporary field type must match the source"

    Dx!(temp, A.U, x, tempvec, A.boundarycondition)
    clear_fermion!(y)
    add_fermion!(y, A.mass, x, 1, temp)
    set_wing_fermion!(y, A.boundarycondition)

    unused!(temps, temp_token)
    unused!(temps, tempvec_token)
    return y
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields,
    A::Adjoint_Staggered_operator,
    x::AbstractFermionfields,
)
    temps = A.parent._temporary_fermi
    temp, temp_token = get_temp(temps)
    tempvec, tempvec_token = get_temp(temps, 3)

    Dx!(temp, A.parent.U, x, tempvec, A.parent.boundarycondition)
    clear_fermion!(y)
    add_fermion!(y, A.parent.mass, x, -1, temp)
    set_wing_fermion!(y, A.parent.boundarycondition)

    unused!(temps, tempvec_token)
    unused!(temps, temp_token)
    return y
end
