import Gaugefields.Temporalfields_module: Temporalfields, unused!, get_temp


struct Staggered_Dirac_operator{Dim,T,fermion} <:
       Dirac_operator{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    boundarycondition::Vector{Int8}
    mass::Float64
    _temporary_fermi::Temporalfields{fermion}#Vector{fermion}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}#Vector{fermion}
    #verbose::Union{Verbose_1,Verbose_2,Verbose_3}
end

include("./StaggeredFermion_4D_wing.jl")
include("./StaggeredFermion_4D_nowing.jl")
include("./StaggeredFermion_2D_wing.jl")
include("./StaggeredFermion_2D_nowing.jl")
include("./StaggeredFermion_4D_nowing_mpi.jl")

function Staggered_Dirac_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    xtype = typeof(x)
    num = 6
    _temporary_fermi = Temporalfields(x; num)
    #_temporary_fermi = Array{xtype,1}(undef, num)

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


    #for i = 1:num
    #    _temporary_fermi[i] = similar(x)
    #end

    numcg = 8 * 4
    _temporary_fermion_forCG = Temporalfields(x; num=numcg)
    #_temporary_fermion_forCG = Array{xtype,1}(undef, numcg)
    #for i = 1:numcg
    #    _temporary_fermion_forCG[i] = similar(x)
    #end

    #verbose_print = Verbose_print(verbose_level)
    verbose_print = Verbose_print(verbose_level, myid=get_myrank(x))

    return Staggered_Dirac_operator{Dim,eltype(U),xtype}(
        U,
        boundarycondition,
        mass,
        _temporary_fermi,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        verbose_print,
        _temporary_fermion_forCG,
    )
end

function get_Dim(::Staggered_Dirac_operator{Dim,T,fermion}) where {Dim,T,fermion}
    return Dim
end

function get_fermiontype(::Staggered_Dirac_operator{Dim,T,fermion}) where {Dim,T,fermion}
    return fermion
end

function (D::Staggered_Dirac_operator{Dim,T,fermion})(U) where {Dim,T,fermion}
    return Staggered_Dirac_operator{Dim,T,fermion}(
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

struct DdagD_Staggered_operator{Dim,T,fermion} <: DdagD_operator
    dirac::Staggered_Dirac_operator{Dim,T,fermion}

    function DdagD_Staggered_operator(
        U::Array{T,1},
        x,
        parameters,
    ) where {T<:AbstractGaugefields}
        D = Staggered_Dirac_operator(U, x, parameters)
        Dim = get_Dim(D)
        fermion = get_fermiontype(D)
        return new{Dim,T,fermion}(D)
    end

    function DdagD_Staggered_operator(
        D::Staggered_Dirac_operator{Dim,T,fermion},
    ) where {Dim,T,fermion}
        return new{Dim,T,fermion}(D)
    end

end



struct Adjoint_Staggered_operator{T} <: Adjoint_Dirac_operator
    parent::T
end


function Base.adjoint(A::T) where {T<:Staggered_Dirac_operator}
    Adjoint_Staggered_operator{typeof(A)}(A)
end


function Initialize_StaggeredFermion(
    u::AbstractGaugefields{NC,Dim};
    nowing=false,
) where {NC,Dim}
    _, _, NN... = size(u)
    return Initialize_StaggeredFermion(NC, NN..., nowing=nowing)
end

function Initialize_StaggeredFermion(NC, NN...; nowing=false)
    Dim = length(NN)
    if Dim == 4
        if nowing
            fermion = StaggeredFermion_4D_nowing(NC, NN...)
        else
            fermion = StaggeredFermion_4D_wing(NC, NN...)
        end
    elseif Dim == 2
        if nowing
            fermion = StaggeredFermion_2D_nowing(NC, NN...)
        else
            fermion = StaggeredFermion_2D_wing(NC, NN...)
        end
    else
        error("Dimension $Dim is not supported")
    end
    return fermion
end


function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:Staggered_Dirac_operator,T3<:AbstractFermionfields}

    #@assert typeof(A._temporary_fermi[1]) == typeof(x) "type of A._temporary_fermi[1] $(typeof(A._temporary_fermi[1])) should be type of x: $(typeof(x))"

    temps = A._temporary_fermi
    temp, it_temp = get_temp(temps)
    tempvec, its_tempvec = get_temp(temps, 3)
    @assert typeof(temp) == typeof(x) "type of A._temporary_fermi[1] $(typeof(temp)) should be type of x: $(typeof(x))"

    #temp = temps[4]
    #Dx!(temp, A.U, x, [tempvec[1], tempvec[2], tempvec[3]], A.boundarycondition)
    Dx!(temp, A.U, x, tempvec, A.boundarycondition)
    clear_fermion!(y)
    add_fermion!(y, A.mass, x, 1, temp)
    set_wing_fermion!(y, A.boundarycondition)

    unused!(temps, it_temp)
    unused!(temps, its_tempvec)

    #error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")
end

function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:Adjoint_Staggered_operator,T3<:AbstractFermionfields}
    #error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")

    temps = A.parent._temporary_fermi
    temp, it_temp = get_temp(temps)
    tempvec, its_temp = get_temp(temps, 3)
    #temp = temps[4]
    #Dx!(temp, A.parent.U, x, [temps[1], temps[2], temps[3]], A.parent.boundarycondition)
    Dx!(temp, A.parent.U, x, tempvec, A.parent.boundarycondition)
    clear_fermion!(y)
    add_fermion!(y, A.parent.mass, x, -1, temp)
    set_wing_fermion!(y, A.parent.boundarycondition)
    unused!(temps, its_temp)
    unused!(temps, it_temp)
    #println(y[1,1,1,1,1,1])
    return
end

function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:DdagD_Staggered_operator,T3<:AbstractFermionfields}
    #@assert typeof(A.dirac._temporary_fermi[1]) == typeof(x) "type of A._temporary_fermi[1] $(typeof(A.dirac._temporary_fermi[1])) should be type of x: $(typeof(x))"
    temps = A.dirac._temporary_fermi
    temp, it_temp = get_temp(temps)
    temp2, it_temp2 = get_temp(temps)
    tempvec, its_tempvec = get_temp(temps, 3)
    @assert typeof(temp) == typeof(x) "type of A._temporary_fermi[1] $(typeof(temmp)) should be type of x: $(typeof(x))"

    #temp = temps[5]
    #temp2 = temps[6]
    #Dx!(temp, A.dirac.U, x, [temps[1], temps[2], temps[3]], A.dirac.boundarycondition)
    #Dx!(temp2, A.dirac.U, temp, [temps[1], temps[2], temps[3]], A.dirac.boundarycondition)
    Dx!(temp, A.dirac.U, x, tempvec, A.dirac.boundarycondition)
    Dx!(temp2, A.dirac.U, temp, tempvec, A.dirac.boundarycondition)

    clear_fermion!(y)
    add_fermion!(y, A.dirac.mass^2, x, -1, temp2)
    set_wing_fermion!(y, A.dirac.boundarycondition)

    unused!(temps, it_temp)
    unused!(temps, it_temp2)
    unused!(temps, its_tempvec)

    #error("xout")
    return
end

#include("./StaggeredFermion_4D_accelerator.jl")
#include("./kernelfunctions/Staggered_jacc.jl")
#include("./kernelfunctions/linearalgebra_mul_jacc.jl")
