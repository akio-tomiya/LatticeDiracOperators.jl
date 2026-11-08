import LatticeMatrices: WilsonDiracCloverOperator4D,
    mul_cached_clover!, mul_cached_clover_adjoint!

function _clover_migration_error()
    return ArgumentError("WilsonClover uses the LatticeMatrices backend. " *
        "Create gauge links with Gaugefields.gauge_configuration or " *
        "Initialize_Gaugefields(...; isMPILattice=true), load the original " *
        "configuration with Gaugefields.load_configuration!, then recreate " *
        "fermion fields, actions, momenta and force buffers from those links. " *
        "CPU, GPU and MPI execution use the same initialization path; " *
        "Dirac_operator does not copy or convert existing fields.")
end

"""
    Wilson_Dirac_operator_clover(U, x, parameters)

Build the standard MPILattice Wilson--clover wrapper.  The Wilson stencil,
clover field strength, cache, and their adjoints are owned by
LatticeMatrices.jl.
"""
function Wilson_Dirac_operator_clover(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    Dim == 4 || throw(ArgumentError(
        "the LatticeMatrices Wilson--clover backend is implemented only in 4D"))
    all(link -> link isa Gaugefields_4D_MPILattice, U) &&
        x isa WilsonFermion_4D_MPILattice || throw(_clover_migration_error())
    check_parameters(parameters, "r", 1.0) == 1 || throw(ArgumentError(
        "WilsonClover currently implements the Wilson parameter r=1"))
    check_parameters(parameters, "method_CG", "bicg") == "preconditiond_bicgstab" &&
        throw(ArgumentError("the Wilson unit-diagonal even/odd solver is not a " *
            "Clover Schur solver; use method_CG=\"bicg\" or \"bicgstab\""))
    @assert haskey(parameters, "κ") "parameters should have the keyword κ"
    check_parameters(parameters, "Donly", false) && throw(ArgumentError(
        "Donly=true is not supported for the Wilson--clover operator"))

    κ = parameters["κ"]
    cSW = check_parameters(parameters, "cSW", 1.5612)
    lm_operator = WilsonDiracCloverOperator4D(_wilson_lm_links(U), κ, cSW)
    return _build_Wilson_Dirac_operator_LM(U, x, parameters, lm_operator)
end

has_cloverterm(
    ::Wilson_Dirac_operator_improved{Dim,T,fermion,TD},
) where {Dim,T,fermion,TD<:WilsonDiracCloverOperator4D} = true

_is_lm_clover(
    ::Wilson_Dirac_operator_improved{Dim,T,fermion,TD},
) where {Dim,T,fermion,TD<:WilsonDiracCloverOperator4D} = true

function _rebuild_Wilson_LM_operator(
    lm_operator::WilsonDiracCloverOperator4D, U, κ,
)
    return WilsonDiracCloverOperator4D(
        _wilson_lm_links(U), κ, lm_operator.cSW)
end

function new_UinDonly(
    D::Wilson_Dirac_operator_improved{Dim,T,fermion,TD}, U,
) where {Dim,T,fermion,TD<:WilsonDiracCloverOperator4D}
    throw(ArgumentError(
        "the hopping-only D4x! interface is not defined for Wilson--clover"))
end

@inline function _mul_Wilson_LM!(
    result, lm_operator::WilsonDiracCloverOperator4D, U, source,
)
    return mul_cached_clover!(
        result, lm_operator,
        U[1].U, U[2].U, U[3].U, U[4].U, source,
    )
end

@inline function _mul_Wilson_LM_adjoint!(
    result, lm_operator::WilsonDiracCloverOperator4D, U, source,
)
    return mul_cached_clover_adjoint!(
        result, lm_operator,
        U[1].U, U[2].U, U[3].U, U[4].U, source,
    )
end
