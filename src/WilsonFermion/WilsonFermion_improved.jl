import LatticeMatrices: WilsonDiracOperator4D
struct Wilson_Dirac_operator_improved{Dim,T,fermion} <:
       Wilson_Dirac_operators{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    D::WilsonDiracOperator4D{T}
    κ::Float64 #Hopping parameter
    _temporary_fermi::Temporalfields{fermion}#Vector{fermion}
    factor::Float64
    boundarycondition::Vector{Int8}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}#Vector{fermion}
end

function Wilson_Dirac_operator_improved(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}


    T = eltype(U)
    xtype = typeof(x)
    @assert haskey(parameters, "κ") "parameters should have the keyword κ"

    κ = parameters["κ"]
    #@assert κ == 1 "In this improved Dirac operator, κ should be 1. Now κ = $κ"
    @assert Dim == 4 "This improved Wilson Dirac operator is implemented only for 4D lattice. Now Dim = $Dim"

    if Dim == 4
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, 1, 1, -1])
    elseif Dim == 2
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, -1])
    else
        error("Dim should be 2 or 4!")
    end

    num = 5
    _temporary_fermi = Temporalfields(x; num)

    factor = check_parameters(parameters, "factor", 1)#1/2κ)

    numcg = check_parameters(parameters, "numtempvec_CG", 12)
    _temporary_fermion_forCG = Temporalfields(x; num=numcg)#Array{xtype,1}(undef, numcg)

    eps_CG = check_parameters(parameters, "eps_CG", default_eps_CG)
    #println("eps_CG = ",eps_CG)
    MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)

    verbose_level = check_parameters(parameters, "verbose_level", 2)
    #verbose_print = Verbose_print(verbose_level)

    verbose_print = Verbose_print(verbose_level, myid=get_myrank(x))

    method_CG = check_parameters(parameters, "method_CG", "bicg")

    r = check_parameters(parameters, "r", 1.0)
    @assert r == 1 "In fast Wilson mode, r should be 1. Now r = $r"

    D = WilsonDiracOperator4D(U, κ)


    return Wilson_Dirac_operator_faster{Dim,T,xtype}(
        U,
        D,
        κ,
        _temporary_fermi,
        factor,
        boundarycondition,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        verbose_print,
        _temporary_fermion_forCG
    )


end

function (D::Wilson_Dirac_operator_improved{Dim,T,fermion})(U) where {Dim,T,fermion}
    return Wilson_Dirac_operator_improved{Dim,T,fermion}(
        U,
        WilsonDiracOperator4D(U, D.κ),
        D.κ,
        D._temporary_fermi,
        D.factor,
        D.boundarycondition,
        D.eps_CG,
        D.MaxCGstep,
        D.verbose_level,
        D.method_CG,
        D.verbose_print,
        D._temporary_fermion_forCG
    )
end

struct Adjoint_Wilson_operator_improved{T} <: Adjoint_Dirac_operator
    parent::T
end

function Base.adjoint(A::T) where {T<:Wilson_Dirac_operator_improved}
    Adjoint_Wilson_operator_improved{typeof(A)}(A)
end

"""
ψ_n - κ sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(
    y::T1,
    A::Wilson_Dirac_operator_improved{Dim,T,fermion},
    x::T3,
) where {T1<:AbstractFermionfields,T,Dim,fermion,T3<:AbstractFermionfields}


    clear_fermion!(y)
    temp1, it_temp1 = get_temp(A._temporary_fermi)

    mul!(temp1.f, A.D, x.f)

    add_fermion!(y, A.factor, temp1)
    unused!(A._temporary_fermi, it_temp1)
    set_wing_fermion!(y)
end
