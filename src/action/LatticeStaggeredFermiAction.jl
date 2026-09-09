struct _LatticeOneLinkStaggeredSpec{M}
    mass::M
end

struct _LatticeHISQStaggeredSpec{C}
    cache::C
end

struct _LatticeStaggeredActionData{D,S}
    DdagD::D
    specification::S
end

struct _ApplyLatticeOneLinkStaggered{adjoint_operator,M}
    mass::M
end

function (apply::_ApplyLatticeOneLinkStaggered{adjoint_operator})(
    result, U1, U2, U3, U4, input, _fermion_temporaries, _gauge_temporaries,
) where adjoint_operator
    links = [U1.U, U2.U, U3.U, U4.U]
    operator = LatticeMatrices.StaggeredDiracOperator4D(links, apply.mass)
    applied_operator = adjoint_operator ? adjoint(operator) : operator
    mul!(result.field, applied_operator, input.field)
    return result
end

struct _ApplyLatticeHISQStaggered{adjoint_operator,C}
    cache::C
end

function (apply::_ApplyLatticeHISQStaggered{adjoint_operator})(
    result, U1, U2, U3, U4, input, _fermion_temporaries, _gauge_temporaries,
) where adjoint_operator
    if adjoint_operator
        LatticeMatrices.mul_cached_hisq_adjoint!(
            result.field, apply.cache,
            U1.U, U2.U, U3.U, U4.U, input.field)
    else
        LatticeMatrices.mul_cached_hisq!(
            result.field, apply.cache,
            U1.U, U2.U, U3.U, U4.U, input.field)
    end
    return result
end

@inline function _lattice_staggered_links(U)
    length(U) == 4 || throw(ArgumentError(
        "a lattice staggered action requires four gauge links"))
    links = getproperty.(U, :U)
    all(link -> link isa LatticeMatrices.LatticeMatrix{4}, links) ||
        throw(ArgumentError(
            "a lattice staggered action requires LatticeMatrices-backed links"))
    return links
end

function _lattice_staggered_rhmc(Nf::Int)
    Nf in (1, 2, 4, 8) || throw(ArgumentError(
        "Nf must be one of 1, 2, 4, or 8"))
    if Nf == 4 || Nf == 8
        return nothing, nothing, 5
    end
    action_rhmc = RHMC(Nf // 16, n=15)
    md_rhmc = RHMC(Nf // 8, n=10)
    number_of_temporaries = max(
        get_order(action_rhmc), get_order(md_rhmc)) + 5
    return action_rhmc, md_rhmc, number_of_temporaries
end

"""
    StaggeredFermiAction(U, fermion; kwargs...)

Construct a LatticeMatrices-backed staggered action with an analytic link
force. Set `discretization=:staggered` for the one-link operator or
`discretization=:hisq` for the cached HISQ operator. Both paths run on every
JACC backend and do not require Enzyme.

The supported flavor counts are `Nf=1`, `2`, `4`, and `8`; the first two use
the existing RHMC approximation. HISQ requires gauge and fermion halo width
`nw >= 3`, while the one-link action requires `nw >= 1`.
"""
function StaggeredFermiAction(
    U::Vector{TG},
    fermion::GeneralFermion;
    mass::Real,
    Nf::Integer=4,
    discretization::Symbol=:staggered,
    naik_epsilon::Real=0,
    eps_CG::Real=1e-12,
    maxsteps::Integer=10_000,
    verbose_level::Integer=0,
) where {TG<:Fields_4D_MPILattice}
    links = _lattice_staggered_links(U)
    fermion.field.NC2 == 1 || throw(ArgumentError(
        "a staggered action requires a one-column fermion field"))
    all(link -> link.NC1 == fermion.field.NC1, links) || throw(ArgumentError(
        "gauge links and the staggered fermion must have equal color size"))
    all(link -> link.gsize == fermion.field.gsize &&
                link.PN == fermion.field.PN &&
                link.dims == fermion.field.dims &&
                link.nw == fermion.field.nw, links) || throw(ArgumentError(
        "gauge links and the staggered fermion must share a lattice layout"))
    fermion.field.nw >= 1 || throw(ArgumentError(
        "a lattice staggered action requires halo width nw >= 1"))

    real_type = typeof(real(zero(eltype(fermion.field.A))))
    typed_mass = convert(real_type, mass)
    typed_epsilon = convert(real_type, naik_epsilon)
    isfinite(typed_mass) || throw(ArgumentError("mass must be finite"))
    isfinite(typed_epsilon) || throw(ArgumentError(
        "naik_epsilon must be finite"))

    specification, apply_D, apply_Ddag = if discretization == :staggered
        spec = _LatticeOneLinkStaggeredSpec(typed_mass)
        forward = _ApplyLatticeOneLinkStaggered{false,typeof(typed_mass)}(
            typed_mass)
        backward = _ApplyLatticeOneLinkStaggered{true,typeof(typed_mass)}(
            typed_mass)
        spec, forward, backward
    elseif discretization == :hisq
        fermion.field.nw >= 3 || throw(ArgumentError(
            "a HISQ action requires halo width nw >= 3"))
        cache = LatticeMatrices.HISQDiracCache4D(
            links, typed_mass; naik_epsilon=typed_epsilon)
        spec = _LatticeHISQStaggeredSpec(cache)
        forward = _ApplyLatticeHISQStaggered{false,typeof(cache)}(cache)
        backward = _ApplyLatticeHISQStaggered{true,typeof(cache)}(cache)
        spec, forward, backward
    else
        throw(ArgumentError(
            "discretization must be :staggered or :hisq"))
    end

    DdagD = DdagDgeneral(
        U, fermion, apply_D, apply_Ddag;
        numcg=4,
        num=3,
        numg=1,
        eps_CG=Float64(eps_CG),
        maxsteps=Int(maxsteps),
        verbose_level=Int(verbose_level),
        numtemp=1,
    )
    action_data = _LatticeStaggeredActionData(DdagD, specification)
    flavors = Int(Nf)
    action_rhmc, md_rhmc, number_of_temporaries =
        _lattice_staggered_rhmc(flavors)
    return StaggeredFermiAction(
        Val(:lattice_matrices), action_data, fermion, U[1], flavors,
        action_rhmc, md_rhmc, number_of_temporaries)
end

@inline function _lattice_staggered_dagdag(action, U)
    return action.diracoperator.DdagD(U)
end

@inline function _lattice_staggered_apply_D!(result, action, U, input)
    DdagD = action.diracoperator.DdagD
    return DdagD.apply_D(
        result, U[1], U[2], U[3], U[4], input, nothing, nothing)
end

@inline function _lattice_staggered_apply_Ddag!(result, action, U, input)
    DdagD = action.diracoperator.DdagD
    return DdagD.apply_Ddag(
        result, U[1], U[2], U[3], U[4], input, nothing, nothing)
end

@inline function _kernel_clear_general_fermion_parity!(
    site, field, clear_even, ::Val{NC}, ::Val{nw}, indexer,
    mpi_coordinates, local_size,
) where {NC,nw}
    x = LatticeMatrices.delinearize(indexer, site, nw)
    global_sum = zero(Int)
    @inbounds for direction in 1:4
        global_sum += x[direction] - nw - 1 +
            mpi_coordinates[direction] * local_size[direction]
    end
    if iseven(global_sum) == clear_even
        @inbounds for color in 1:NC
            field[color, 1, x...] = zero(eltype(field))
        end
    end
    return nothing
end

function clear_fermion!(fermion::GeneralFermion, clear_even::Bool)
    field = fermion.field
    JACC.parallel_for(
        prod(field.PN), _kernel_clear_general_fermion_parity!,
        field.A, clear_even, Val(field.NC1), Val(field.nw), field.indexer,
        field.coords, field.PN)
    _mark_halo_dirty!(field)
    return fermion
end

function evaluate_FermiAction(
    action::StaggeredFermiAction{4,D,F,G,Nf},
    U,
    pseudofermion::GeneralFermion,
) where {D<:_LatticeStaggeredActionData,F,G,Nf}
    DdagD = _lattice_staggered_dagdag(action, U)
    if Nf == 4 || Nf == 8
        solution, solution_index = get_temp(action._temporary_fermionfields)
        clear_fermion!(solution)
        solve_DinvX!(solution, DdagD, pseudofermion)
        potential = real(dot(pseudofermion, solution))
        unused!(action._temporary_fermionfields, solution_index)
        return potential
    end

    rhmc = action.rhmc_info_for_action
    order = get_order(rhmc)
    solution, solution_index = get_temp(action._temporary_fermionfields)
    shifted, shifted_indices = get_temp(
        action._temporary_fermionfields, order)
    clear_fermion!(solution)
    clear_fermion!.(shifted)
    shiftedcg(
        shifted,
        get_β_inverse(rhmc),
        solution,
        DdagD,
        pseudofermion;
        eps=DdagD.eps_CG,
        maxsteps=DdagD.MaxCGstep,
        verbose=DdagD.verbose_print,
    )
    clear_fermion!(solution)
    add_fermion!(solution, get_α0_inverse(rhmc), pseudofermion)
    for index in eachindex(shifted)
        add_fermion!(solution, get_α_inverse(rhmc)[index], shifted[index])
    end
    potential = real(dot(solution, solution))
    unused!(action._temporary_fermionfields, solution_index)
    unused!(action._temporary_fermionfields, shifted_indices)
    return potential
end

function gauss_sampling_in_action!(
    gaussian::GeneralFermion,
    _U,
    ::StaggeredFermiAction{4,D,F,G,Nf},
) where {D<:_LatticeStaggeredActionData,F,G,Nf}
    gauss_distribution_fermion!(gaussian)
    return gaussian
end

function sample_pseudofermions!(
    pseudofermion::GeneralFermion,
    U,
    action::StaggeredFermiAction{4,D,F,G,Nf},
    gaussian::GeneralFermion,
) where {D<:_LatticeStaggeredActionData,F,G,Nf}
    if Nf == 4 || Nf == 8
        _lattice_staggered_apply_Ddag!(
            pseudofermion, action, U, gaussian)
        Nf == 4 && clear_fermion!(pseudofermion, false)
        set_wing_fermion!(pseudofermion)
        return pseudofermion
    end

    DdagD = _lattice_staggered_dagdag(action, U)
    rhmc = action.rhmc_info_for_action
    order = get_order(rhmc)
    shifted, shifted_indices = get_temp(
        action._temporary_fermionfields, order)
    clear_fermion!.(shifted)
    shiftedcg(
        shifted,
        get_β(rhmc),
        pseudofermion,
        DdagD,
        gaussian;
        eps=DdagD.eps_CG,
        maxsteps=DdagD.MaxCGstep,
        verbose=DdagD.verbose_print,
    )
    clear_fermion!(pseudofermion)
    add_fermion!(pseudofermion, get_α0(rhmc), gaussian)
    for index in eachindex(shifted)
        add_fermion!(pseudofermion, get_α(rhmc)[index], shifted[index])
    end
    set_wing_fermion!(pseudofermion)
    unused!(action._temporary_fermionfields, shifted_indices)
    return pseudofermion
end

function _lattice_staggered_link_gradient!(
    gradients,
    action::StaggeredFermiAction{4,D,F,G,Nf},
    U,
    result_cotangent::GeneralFermion,
    input::GeneralFermion,
    coefficient,
) where {D<:_LatticeStaggeredActionData,F,G,Nf}
    clear_U!(gradients)
    gradient_links = getproperty.(gradients, :U)
    thin_links = _lattice_staggered_links(U)
    specification = action.diracoperator.specification
    if specification isa _LatticeOneLinkStaggeredSpec
        # LDO stores -1/2 times the ordinary link gradient. The physical
        # action variation contributes -2 Re <D X, (delta D) X>, so the two
        # factors cancel before converting the cotangent to U*dS/dU.
        LatticeMatrices.staggered_link_pullback!(
            gradient_links, thin_links,
            result_cotangent.field, input.field;
            coefficient=coefficient)
    else
        LatticeMatrices.hisq_link_pullback!(
            gradient_links, specification.cache, thin_links,
            result_cotangent.field, input.field;
            coefficient=coefficient)
    end
    return gradients
end

function _lattice_staggered_force_from_solution!(
    UdSfdU,
    action::StaggeredFermiAction{4,D,F,G,Nf},
    U,
    solution::GeneralFermion;
    coefficient=1,
) where {D<:_LatticeStaggeredActionData,F,G,Nf}
    applied, applied_index = get_temp(action._temporary_fermionfields)
    gradients, gradient_indices = get_temp(
        action._temporary_gaugefields, 4)
    work, work_index = get_temp(action._temporary_gaugefields)
    _lattice_staggered_apply_D!(applied, action, U, solution)
    set_wing_fermion!(applied)
    _lattice_staggered_link_gradient!(
        gradients, action, U, applied, solution, coefficient)
    for direction in eachindex(U)
        mul!(work, U[direction], gradients[direction]')
        add_U!(UdSfdU[direction], 1, work)
    end
    unused!(action._temporary_fermionfields, applied_index)
    unused!(action._temporary_gaugefields, gradient_indices)
    unused!(action._temporary_gaugefields, work_index)
    return UdSfdU
end

function calc_UdSfdU!(
    UdSfdU::Vector{<:AbstractGaugefields},
    action::StaggeredFermiAction{4,D,F,G,Nf},
    U::Vector{<:AbstractGaugefields},
    pseudofermion::GeneralFermion,
) where {D<:_LatticeStaggeredActionData,F,G,Nf}
    clear_U!(UdSfdU)
    DdagD = _lattice_staggered_dagdag(action, U)
    if Nf == 4 || Nf == 8
        solution, solution_index = get_temp(action._temporary_fermionfields)
        clear_fermion!(solution)
        solve_DinvX!(solution, DdagD, pseudofermion)
        _lattice_staggered_force_from_solution!(
            UdSfdU, action, U, solution)
        unused!(action._temporary_fermionfields, solution_index)
        set_wing_U!(UdSfdU)
        return UdSfdU
    end

    rhmc = action.rhmc_info_for_MD
    order = get_order(rhmc)
    solution, solution_index = get_temp(action._temporary_fermionfields)
    shifted, shifted_indices = get_temp(
        action._temporary_fermionfields, order)
    clear_fermion!(solution)
    clear_fermion!.(shifted)
    shiftedcg(
        shifted,
        get_β_inverse(rhmc),
        solution,
        DdagD,
        pseudofermion;
        eps=DdagD.eps_CG,
        maxsteps=DdagD.MaxCGstep,
        verbose=DdagD.verbose_print,
    )
    coefficients = get_α_inverse(rhmc)
    for index in eachindex(shifted)
        set_wing_fermion!(shifted[index])
        _lattice_staggered_force_from_solution!(
            UdSfdU, action, U, shifted[index];
            coefficient=coefficients[index])
    end
    unused!(action._temporary_fermionfields, solution_index)
    unused!(action._temporary_fermionfields, shifted_indices)
    set_wing_U!(UdSfdU)
    return UdSfdU
end
