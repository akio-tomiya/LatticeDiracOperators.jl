"""
LatticeMatrices-backed generalized domain-wall wrapper.

The fermion storage is the common `DomainwallFermion_5D_MPILattice`; only the
fifth-direction coefficients differ from the Shamir/Möbius wrapper.  The
legacy `w[s]` implementation remains available through the old constructors.
"""
struct D5DW_GeneralizedDomainwall_operator_MPILattice{
    Dim,TU,fermion,TD,R,VC,
} <: AbstractD5DWGeneralizedDomainwallOperator{Dim}
    U::Array{TU,1}
    D::TD
    mass::R
    _temporary_fermi::Temporalfields{fermion}
    L5::Int64
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}
    boundarycondition::Vector{<:Number}
    M::R
    as::VC
    bs::VC
    cs::VC
end

function D5DW_GeneralizedDomainwall_operator_MPILattice(
    U::Array{<:Gaugefields_4D_MPILattice,1},
    x::DomainwallFermion_5D_MPILattice,
    parameters,
    mass,
    as,
    bs,
    cs,
)
    Dim = 4
    L5 = parameters["L5"]
    L5 == x.L5 || throw(DimensionMismatch(
        "operator L5=$L5 does not match field L5=$(x.L5)"))
    for (name, coefficients) in (("as", as), ("bs", bs), ("cs", cs))
        length(coefficients) == L5 || throw(DimensionMismatch(
            "$name must have length L5=$L5"))
    end

    R = promote_type(
        typeof(float(mass)), typeof(float(check_parameters(parameters, "M", -1))),
        eltype(as), eltype(bs), eltype(cs))
    mass_R = R(mass)
    M = R(check_parameters(parameters, "M", -1))
    as_R, bs_R, cs_R = R.(collect(as)), R.(collect(bs)), R.(collect(cs))
    links = [U[mu].U for mu in 1:4]
    operator = D5DW_GeneralizedDomainwallOperator5D(
        links, L5, mass_R, M, as_R, bs_R, cs_R)

    temporary_fermi = Temporalfields(x; num=4)
    temporary_fermion_forCG = Temporalfields(x; num=7)
    eps_CG = check_parameters(parameters, "eps_CG", default_eps_CG)
    MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)
    verbose_level = check_parameters(parameters, "verbose_level", 2)
    verbose_print = Verbose_print(verbose_level)
    method_CG = check_parameters(parameters, "method_CG", "bicg")
    boundarycondition = collect(x.f.phases)
    if haskey(parameters, "boundarycondition")
        requested_boundary = collect(parameters["boundarycondition"])
        length(requested_boundary) == 4 && push!(requested_boundary, 1)
        requested_boundary ≈ boundarycondition || throw(ArgumentError(
            "domain-wall boundarycondition must match the five-dimensional field phases"))
    end

    TU = eltype(U)
    TD = typeof(operator)
    VC = typeof(as_R)
    return D5DW_GeneralizedDomainwall_operator_MPILattice{
        Dim,TU,typeof(x),TD,R,VC,
    }(
        U, operator, mass_R, temporary_fermi, L5,
        eps_CG, MaxCGstep, verbose_level, method_CG, verbose_print,
        temporary_fermion_forCG, boundarycondition, M, as_R, bs_R, cs_R)
end

function (D::D5DW_GeneralizedDomainwall_operator_MPILattice{
    Dim,TU,fermion,TD,R,VC,
})(U) where {Dim,TU,fermion,TD,R,VC}
    links = [U[mu].U for mu in 1:4]
    operator = D5DW_GeneralizedDomainwallOperator5D(
        links, D.L5, D.mass, D.M, D.as, D.bs, D.cs)
    return D5DW_GeneralizedDomainwall_operator_MPILattice{
        Dim,eltype(U),fermion,typeof(operator),R,VC,
    }(
        U, operator, D.mass, D._temporary_fermi, D.L5,
        D.eps_CG, D.MaxCGstep, D.verbose_level, D.method_CG,
        D.verbose_print, D._temporary_fermion_forCG, D.boundarycondition,
        D.M, D.as, D.bs, D.cs)
end

struct Adjoint_D5DW_GeneralizedDomainwall_operator_MPILattice{T} <:
    Adjoint_Dirac_operator
    parent::T
end

Base.adjoint(D::D5DW_GeneralizedDomainwall_operator_MPILattice) =
    Adjoint_D5DW_GeneralizedDomainwall_operator_MPILattice(D)
Base.adjoint(D::Adjoint_D5DW_GeneralizedDomainwall_operator_MPILattice) =
    D.parent

function LinearAlgebra.mul!(
    result::DomainwallFermion_5D_MPILattice,
    D::D5DW_GeneralizedDomainwall_operator_MPILattice,
    source::DomainwallFermion_5D_MPILattice,
)
    mul!(result.f, D.D, source.f)
    set_halo!(result.f)
    return result
end

function LinearAlgebra.mul!(
    result::DomainwallFermion_5D_MPILattice,
    Ddag::Adjoint_D5DW_GeneralizedDomainwall_operator_MPILattice,
    source::DomainwallFermion_5D_MPILattice,
)
    mul!(result.f, adjoint(Ddag.parent.D), source.f)
    set_halo!(result.f)
    return result
end

function Initialize_GeneralizedDomainwallFermion(
    u::Gaugefields_4D_MPILattice,
    L5;
    nowing=true,
    kwargs...,
)
    return DomainwallFermion_5D_MPILattice(
        u, L5; operator_name="GeneralizedDomainwall", kwargs...)
end

function GeneralizedDomainwall_Dirac_operator(
    U::Array{<:Gaugefields_4D_MPILattice,1},
    x::DomainwallFermion_5D_MPILattice,
    parameters,
)
    Dim = 4
    mass = parameters["mass"]
    L5 = parameters["L5"]
    as = check_parameters(parameters, "as", ones(L5))
    bs = check_parameters(parameters, "bs", fill(1.5, L5))
    cs = check_parameters(parameters, "cs", fill(0.5, L5))
    D5DW = D5DW_GeneralizedDomainwall_operator_MPILattice(
        U, x, parameters, mass, as, bs, cs)
    D5DW_PV = D5DW_GeneralizedDomainwall_operator_MPILattice(
        U, x, parameters, one(mass), as, bs, cs)
    return GeneralizedDomainwall_Dirac_operator{
        Dim,eltype(U),typeof(x),Nothing,
    }(
        U, D5DW, D5DW_PV, mass,
        D5DW.eps_CG, D5DW.MaxCGstep, D5DW.verbose_level,
        D5DW.method_CG, D5DW.verbose_print, D5DW.boundarycondition,
        collect(bs), collect(cs))
end

export D5DW_GeneralizedDomainwall_operator_MPILattice
