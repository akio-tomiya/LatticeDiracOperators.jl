import LatticeMatrices: StaggeredDiracOperator4D

struct Staggered_Dirac_operator_MPILattice{
    Dim,T,fermion,TD,BC
} <: Staggered_Dirac_operators{Dim}
    U::Array{T,1}
    D::TD
    boundarycondition::BC
    mass::Float64
    _temporary_fermi::Temporalfields{fermion}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}
end

@inline function _staggered_lm_links(U)
    length(U) == 4 || throw(ArgumentError(
        "the LatticeMatrices staggered backend requires four gauge links"))
    return [link.U for link in U]
end

function Staggered_Dirac_operator(
    U::Array{T,1},
    x::StaggeredFermion_4D_MPILattice,
    parameters,
) where {T<:Gaugefields_4D_MPILattice}
    haskey(parameters, "mass") || throw(ArgumentError(
        "parameters must contain the keyword mass"))
    mass = Float64(parameters["mass"])
    boundarycondition = collect(check_parameters(
        parameters, "boundarycondition", [1, 1, 1, -1]))
    _staggered_boundary_phases_match(boundarycondition, x.f.phases) || throw(ArgumentError(
        "Dirac boundarycondition $(Tuple(boundarycondition)) does not match " *
        "the fermion phases $(x.f.phases); pass the boundary condition when " *
        "initializing the fermion field"))

    temporary_fermi = Temporalfields(x; num=6)
    numcg = check_parameters(parameters, "numtempvec_CG", 8 * 4)
    temporary_fermion_forCG = Temporalfields(x; num=numcg)
    eps_CG = Float64(check_parameters(
        parameters,
        "eps_CG",
        check_parameters(parameters, "eps", default_eps_CG),
    ))
    MaxCGstep = Int64(check_parameters(parameters, "MaxCGstep", default_MaxCGstep))
    verbose_level = Int8(check_parameters(parameters, "verbose_level", 2))
    method_CG = String(check_parameters(parameters, "method_CG", "bicg"))
    verbose_print = Verbose_print(verbose_level, myid=get_myrank(x))
    lm_operator = StaggeredDiracOperator4D(_staggered_lm_links(U), mass)

    return Staggered_Dirac_operator_MPILattice{
        4,T,typeof(x),typeof(lm_operator),typeof(boundarycondition)
    }(
        U,
        lm_operator,
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

get_Dim(::Staggered_Dirac_operator_MPILattice{Dim}) where {Dim} = Dim
get_fermiontype(
    ::Staggered_Dirac_operator_MPILattice{Dim,T,fermion},
) where {Dim,T,fermion} = fermion

function _replace_Staggered_LM_operator(
    D::Staggered_Dirac_operator_MPILattice{Dim,T,fermion},
    U::Array{Tnew,1},
    lm_operator,
) where {Dim,T,fermion,Tnew}
    return Staggered_Dirac_operator_MPILattice{
        Dim,Tnew,fermion,typeof(lm_operator),typeof(D.boundarycondition)
    }(
        U,
        lm_operator,
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

function (D::Staggered_Dirac_operator_MPILattice)(U)
    lm_operator = StaggeredDiracOperator4D(_staggered_lm_links(U), D.mass)
    return _replace_Staggered_LM_operator(D, U, lm_operator)
end

struct Adjoint_Staggered_operator_MPILattice{T} <: Adjoint_Dirac_operator
    parent::T
end

Base.adjoint(D::Staggered_Dirac_operator_MPILattice) =
    Adjoint_Staggered_operator_MPILattice(D)
Base.adjoint(D::Adjoint_Staggered_operator_MPILattice) = D.parent

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_MPILattice,
    D::Staggered_Dirac_operator_MPILattice,
    x::StaggeredFermion_4D_MPILattice,
)
    mul!(y.f, D.D, x.f)
    set_wing_fermion!(y, D.boundarycondition)
    return y
end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_MPILattice,
    Ddag::Adjoint_Staggered_operator_MPILattice,
    x::StaggeredFermion_4D_MPILattice,
)
    mul!(y.f, Ddag.parent.D', x.f)
    set_wing_fermion!(y, Ddag.parent.boundarycondition)
    return y
end

function DdagD_Staggered_operator(
    D::Staggered_Dirac_operator_MPILattice{Dim,T,fermion},
) where {Dim,T,fermion}
    return DdagD_Staggered_operator{
        Dim,T,fermion,typeof(D)
    }(D)
end
