import LatticeMatrices:
    HISQDiracCache4D, mul_cached_hisq!, mul_cached_hisq_adjoint!

"""
    HISQ_Dirac_operator_MPILattice

LDO wrapper for the LatticeMatrices HISQ operator.  The fermion field remains
`StaggeredFermion_4D_MPILattice`; `cache` owns the derived Fat7,
reunitarized, corrected-fat, and Naik links built from the thin gauge links.
"""
struct HISQ_Dirac_operator_MPILattice{
    Dim,T,fermion,TC,BC
} <: Staggered_Dirac_operators{Dim}
    U::Array{T,1}
    cache::TC
    boundarycondition::BC
    mass::Float64
    naik_epsilon::Float64
    _temporary_fermi::Temporalfields{fermion}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}
end

function _validate_hisq_wrapper_geometry(U, x::StaggeredFermion_4D_MPILattice)
    links = _staggered_lm_links(U)
    reference = links[1]
    x.NC == reference.NC1 && x.NC == reference.NC2 ||
        throw(ArgumentError(
            "the HISQ fermion field and thin gauge links must have the same number of colors"))
    x.f.gsize == reference.gsize && x.f.PN == reference.PN &&
        x.f.dims == reference.dims && x.f.nw == reference.nw ||
        throw(ArgumentError(
            "the HISQ fermion field and thin gauge links must use the same lattice geometry"))
    iszero(reference.nw) || reference.nw >= 3 || throw(ArgumentError(
        "HISQ requires halo width 0 or at least 3; use halo=3 for HMC force calculations"))
    return links
end

function _validate_hisq_replacement_geometry(
    D::HISQ_Dirac_operator_MPILattice,
    U,
)
    links = _staggered_lm_links(U)
    current = D.U[1].U
    replacement = links[1]
    current.gsize == replacement.gsize && current.PN == replacement.PN &&
        current.dims == replacement.dims && current.nw == replacement.nw ||
        throw(ArgumentError(
            "replacement HISQ thin links must preserve the operator lattice geometry"))
    replacement.NC1 == current.NC1 && replacement.NC2 == current.NC2 ||
        throw(ArgumentError(
            "replacement HISQ thin links must preserve the operator color dimension"))
    return links
end

function HISQ_Dirac_operator(
    U::Array{T,1},
    x::StaggeredFermion_4D_MPILattice,
    parameters,
) where {T<:Gaugefields_4D_MPILattice}
    haskey(parameters, "mass") || throw(ArgumentError(
        "parameters must contain the keyword mass"))
    mass = Float64(parameters["mass"])
    naik_epsilon = Float64(check_parameters(parameters, "naik_epsilon", 0.0))
    boundarycondition = collect(check_parameters(
        parameters, "boundarycondition", [1, 1, 1, -1]))
    _staggered_boundary_phases_match(boundarycondition, x.f.phases) ||
        throw(ArgumentError(
            "Dirac boundarycondition $(Tuple(boundarycondition)) does not match " *
            "the fermion phases $(x.f.phases); pass the boundary condition when " *
            "initializing the fermion field"))

    links = _validate_hisq_wrapper_geometry(U, x)
    cache = HISQDiracCache4D(links, mass; naik_epsilon)
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

    return HISQ_Dirac_operator_MPILattice{
        4,T,typeof(x),typeof(cache),typeof(boundarycondition)
    }(
        U,
        cache,
        boundarycondition,
        mass,
        naik_epsilon,
        temporary_fermi,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        verbose_print,
        temporary_fermion_forCG,
    )
end

get_Dim(::HISQ_Dirac_operator_MPILattice{Dim}) where {Dim} = Dim
get_fermiontype(
    ::HISQ_Dirac_operator_MPILattice{Dim,T,fermion},
) where {Dim,T,fermion} = fermion

function _replace_HISQ_links(
    D::HISQ_Dirac_operator_MPILattice{Dim,T,fermion},
    U::Array{Tnew,1},
    cache,
) where {Dim,T,fermion,Tnew}
    return HISQ_Dirac_operator_MPILattice{
        Dim,Tnew,fermion,typeof(cache),typeof(D.boundarycondition)
    }(
        U,
        cache,
        D.boundarycondition,
        D.mass,
        D.naik_epsilon,
        D._temporary_fermi,
        D.eps_CG,
        D.MaxCGstep,
        D.verbose_level,
        D.method_CG,
        D.verbose_print,
        D._temporary_fermion_forCG,
    )
end

function (D::HISQ_Dirac_operator_MPILattice)(U::Array{Tnew,1}) where {Tnew}
    links = _validate_hisq_replacement_geometry(D, U)
    cache_link_type = typeof(D.cache.level1_links[1])
    cache = typeof(links[1]) === cache_link_type ? D.cache :
        HISQDiracCache4D(links, D.mass; naik_epsilon=D.naik_epsilon)
    return _replace_HISQ_links(D, U, cache)
end

struct Adjoint_HISQ_operator_MPILattice{T} <: Adjoint_Dirac_operator
    parent::T
end

Base.adjoint(D::HISQ_Dirac_operator_MPILattice) =
    Adjoint_HISQ_operator_MPILattice(D)
Base.adjoint(D::Adjoint_HISQ_operator_MPILattice) = D.parent

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_MPILattice,
    D::HISQ_Dirac_operator_MPILattice,
    x::StaggeredFermion_4D_MPILattice,
)
    links = _staggered_lm_links(D.U)
    mul_cached_hisq!(
        y.f, D.cache, links[1], links[2], links[3], links[4], x.f)
    set_wing_fermion!(y, D.boundarycondition)
    return y
end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_MPILattice,
    Ddag::Adjoint_HISQ_operator_MPILattice,
    x::StaggeredFermion_4D_MPILattice,
)
    D = Ddag.parent
    links = _staggered_lm_links(D.U)
    mul_cached_hisq_adjoint!(
        y.f, D.cache, links[1], links[2], links[3], links[4], x.f)
    set_wing_fermion!(y, D.boundarycondition)
    return y
end

function DdagD_Staggered_operator(
    D::HISQ_Dirac_operator_MPILattice{Dim,T,fermion},
) where {Dim,T,fermion}
    return DdagD_Staggered_operator{
        Dim,T,fermion,typeof(D)
    }(D)
end
