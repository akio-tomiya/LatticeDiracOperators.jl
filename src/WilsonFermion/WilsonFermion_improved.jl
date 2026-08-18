import LatticeMatrices: WilsonDiracOperator4D, WilsonDiracOperator4D_Donly
struct Wilson_Dirac_operator_improved{Dim,T,fermion,TD} <:
       Wilson_Dirac_operators{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    D::TD
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

    γ::Array{ComplexF64,3}
    rplusγ::Array{ComplexF64,3}
    rminusγ::Array{ComplexF64,3}
    r::Float64 #Wilson term
    hopp::Array{ComplexF64,1}
    hopm::Array{ComplexF64,1}

end

@inline function _wilson_lm_links(U)
    length(U) == 4 || throw(ArgumentError(
        "the LatticeMatrices Wilson backend requires four gauge links"))
    return [link.U for link in U]
end

function _build_Wilson_Dirac_operator_LM(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
    lm_operator,
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

    if Dim == 4
        γ, rplusγ, rminusγ = mk_gamma(r)
        hopp = zeros(ComplexF64, 4)
        hopm = zeros(ComplexF64, 4)
        hopp .= κ
        hopm .= κ
    elseif Dim == 2
        γ, rplusγ, rminusγ = mk_sigma(r)
        hopp = zeros(ComplexF64, 2)
        hopm = zeros(ComplexF64, 2)
        hopp .= κ
        hopm .= κ
    end



    return Wilson_Dirac_operator_improved{Dim,T,xtype,typeof(lm_operator)}(
        U,
        lm_operator,
        κ,
        _temporary_fermi,
        factor,
        boundarycondition,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        verbose_print,
        _temporary_fermion_forCG,
        γ,#::Array{ComplexF64,3}
        rplusγ,#::Array{ComplexF64,3}
        rminusγ,#::Array{ComplexF64,3}
        r,#::Float64 #Wilson term
        hopp,#::Array{ComplexF64,1}
        hopm,#::Array{ComplexF64,1})
    )


end

function Wilson_Dirac_operator_improved(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    @assert haskey(parameters, "κ") "parameters should have the keyword κ"
    κ = parameters["κ"]
    links = _wilson_lm_links(U)
    lm_operator = if check_parameters(parameters, "Donly", false)
        WilsonDiracOperator4D_Donly(links)
    else
        WilsonDiracOperator4D(links, κ)
    end
    return _build_Wilson_Dirac_operator_LM(U, x, parameters, lm_operator)
end

function _replace_Wilson_LM_operator(
    D::Wilson_Dirac_operator_improved{Dim,T,fermion},
    U::Array{Tnew,1},
    lm_operator,
) where {Dim,T,fermion,Tnew}
    return Wilson_Dirac_operator_improved{Dim,Tnew,fermion,typeof(lm_operator)}(
        U,
        lm_operator,
        D.κ,
        D._temporary_fermi,
        D.factor,
        D.boundarycondition,
        D.eps_CG,
        D.MaxCGstep,
        D.verbose_level,
        D.method_CG,
        D.verbose_print,
        D._temporary_fermion_forCG,
        D.γ,#::Array{ComplexF64,3}
        D.rplusγ,#::Array{ComplexF64,3}
        D.rminusγ,#::Array{ComplexF64,3}
        D.r,#::Float64 #Wilson term
        D.hopp,#::Array{ComplexF64,1}
        D.hopm,#::Array{ComplexF64,1})
    )
end

function new_UinDonly(D::Wilson_Dirac_operator_improved, U)
    lm_operator = WilsonDiracOperator4D_Donly(_wilson_lm_links(U))
    return _replace_Wilson_LM_operator(D, U, lm_operator)
end

function _rebuild_Wilson_LM_operator(
    ::WilsonDiracOperator4D, U, κ,
)
    return WilsonDiracOperator4D(_wilson_lm_links(U), κ)
end

function _rebuild_Wilson_LM_operator(
    ::WilsonDiracOperator4D_Donly, U, κ,
)
    return WilsonDiracOperator4D_Donly(_wilson_lm_links(U))
end


function (D::Wilson_Dirac_operator_improved)(U)
    lm_operator = _rebuild_Wilson_LM_operator(D.D, U, D.κ)
    return _replace_Wilson_LM_operator(D, U, lm_operator)
end

has_cloverterm(::Wilson_Dirac_operator_improved) = false
_is_lm_clover(::Any) = false
_is_lm_clover(::Wilson_Dirac_operator_improved) = false

@inline function _mul_Wilson_LM!(result, lm_operator, U, source)
    return mul!(result, lm_operator, source)
end

@inline function _mul_Wilson_LM_adjoint!(result, lm_operator, U, source)
    return mul!(result, adjoint(lm_operator), source)
end

struct Adjoint_Wilson_Dirac_operator_improved{T} <: Adjoint_Dirac_operator
    parent::T
end

function Base.adjoint(A::T) where {T<:Wilson_Dirac_operator_improved}
    Adjoint_Wilson_Dirac_operator_improved{typeof(A)}(A)
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

    _mul_Wilson_LM!(temp1.f, A.D, A.U, x.f)

    add_fermion!(y, A.factor, temp1)
    unused!(A._temporary_fermi, it_temp1)
    set_wing_fermion!(y)
end

function LinearAlgebra.mul!(
    y::T1,
    A::Adjoint_Wilson_Dirac_operator_improved{T},
    x::T3,
) where {T1<:AbstractFermionfields,T,T3<:AbstractFermionfields}


    clear_fermion!(y)
    temp1, it_temp1 = get_temp(A.parent._temporary_fermi)

    _mul_Wilson_LM_adjoint!(temp1.f, A.parent.D, A.parent.U, x.f)

    add_fermion!(y, A.parent.factor, temp1)
    unused!(A.parent._temporary_fermi, it_temp1)
    set_wing_fermion!(y)
end


function D4x!(xout::T1, U::Array{G,1}, x::T2, A::TA, Dim) where {T1,T2,G<:AbstractGaugefields,
    TA<:Wilson_Dirac_operator_improved}

    Dwilson = new_UinDonly(A, U)
    mul!(xout.f, Dwilson.D, x.f)
    set_wing_fermion!(xout)
end

function D4dagx!(
    xout::T1,
    U::Array{G,1},
    x::T2,
    A::TA,
    Dim,
) where {T1,T2,G<:AbstractGaugefields,TA<:Wilson_Dirac_operator_improved}
    Dwilson = new_UinDonly(A, U)
    mul!(xout.f, Dwilson.D', x.f)
    set_wing_fermion!(xout)
end
