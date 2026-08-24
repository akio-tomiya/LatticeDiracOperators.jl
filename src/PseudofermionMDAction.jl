import Gaugefields:
    back_prop!,
    calc_smearedU,
    clear_U!,
    Traceless_antihermitian_add!,
    md_action_workspace,
    md_force!,
    md_potential
using LinearAlgebra: mul!

"""
    PseudofermionMDAction(action, pseudofermion[, smearing])

Adapt an LDO pseudofermion `action` with a fixed `pseudofermion` field to the
Gaugefields molecular-dynamics action-provider interface.  The
pseudofermion field is held fixed during an MD trajectory; refresh it between
trajectories with [`refresh_pseudofermion!`](@ref).

When `action` was constructed with `FermiAction(...; covneuralnet=smearing)`,
the two-argument constructor automatically uses that smearing.  An explicit
third argument overrides this choice.  Refresh, potential evaluation, and
force evaluation then all use the smeared links, and the force is pulled back
to the thin links before it is returned to the MD driver.

The provider can be passed directly to `Gaugefields.md_driver`, or combined
with gauge and other pseudofermion actions using `Gaugefields.MDActionSet`.
"""
struct PseudofermionMDAction{A,P,S}
    action::A
    pseudofermion::P
    smearing::S
end

@inline function _action_smearing(action)
    return hasproperty(action, :covneuralnet) ?
           getproperty(action, :covneuralnet) : nothing
end

PseudofermionMDAction(action, pseudofermion) = PseudofermionMDAction(
    action,
    pseudofermion,
    _action_smearing(action),
)

struct PseudofermionMDWorkspace{D}
    derivative::D
end

struct SmearedPseudofermionMDWorkspace{R,S,T,C}
    raw_derivative::R
    smeared_derivative::S
    thin_derivative::T
    conversion::C
end

"""
    refresh_pseudofermion!(provider, U, noise; kwargs...)

Draw the action-specific Gaussian `noise` and use it to refresh the
pseudofermion field owned by `provider`. Standard LatticeMatrices fields
accept `seed`, `sweep`, `direction`, `color`, `subgroup`, and `rng_algorithm`
keywords. Explicit stream identifiers give backend- and MPI-decomposition-
independent noise. Random-number policy remains outside the deterministic
Gaugefields MD driver.
"""
function refresh_pseudofermion!(
    provider::PseudofermionMDAction{A,P,Nothing},
    U,
    noise;
    kwargs...,
) where {A,P}
    gauss_sampling_in_action!(noise, U, provider.action; kwargs...)
    sample_pseudofermions!(
        provider.pseudofermion,
        U,
        provider.action,
        noise,
    )
    return provider
end

function refresh_pseudofermion!(
    provider::PseudofermionMDAction{A,P,S},
    U,
    noise;
    kwargs...,
) where {A,P,S}
    smeared_links, _, _ = calc_smearedU(U, provider.smearing)
    gauss_sampling_in_action!(
        noise,
        smeared_links,
        provider.action;
        kwargs...,
    )
    sample_pseudofermions!(
        provider.pseudofermion,
        smeared_links,
        provider.action,
        noise,
    )
    return provider
end

function _check_pseudofermion_directions(U)
    isempty(U) && throw(ArgumentError(
        "a pseudofermion MD action requires at least one gauge direction",
    ))
    return nothing
end

function md_action_workspace(
    provider::PseudofermionMDAction{A,P,Nothing},
    U,
) where {A,P}
    _check_pseudofermion_directions(U)
    return PseudofermionMDWorkspace(map(similar, U))
end

function md_action_workspace(provider::PseudofermionMDAction, U)
    _check_pseudofermion_directions(U)
    return SmearedPseudofermionMDWorkspace(
        map(similar, U),
        map(similar, U),
        map(similar, U),
        similar(first(U)),
    )
end

function md_potential(
    provider::PseudofermionMDAction{A,P,Nothing},
    U,
    workspace::PseudofermionMDWorkspace,
) where {A,P}
    return real(evaluate_FermiAction(
        provider.action,
        U,
        provider.pseudofermion,
    ))
end

function md_potential(
    provider::PseudofermionMDAction,
    U,
    workspace::SmearedPseudofermionMDWorkspace,
)
    smeared_links, _, _ = calc_smearedU(U, provider.smearing)
    return real(evaluate_FermiAction(
        provider.action,
        smeared_links,
        provider.pseudofermion,
    ))
end

function md_force!(
    force,
    provider::PseudofermionMDAction{A,P,Nothing},
    U,
    workspace::PseudofermionMDWorkspace,
) where {A,P}
    length(force) == length(U) || throw(ArgumentError(
        "force and U must have the same number of directions",
    ))
    length(workspace.derivative) == length(U) || throw(ArgumentError(
        "the pseudofermion MD workspace has the wrong number of directions",
    ))

    calc_UdSfdU!(
        workspace.derivative,
        provider.action,
        U,
        provider.pseudofermion,
    )
    for direction in eachindex(U)
        clear_U!(force[direction])
        Traceless_antihermitian_add!(
            force[direction],
            -1,
            workspace.derivative[direction],
        )
    end
    return nothing
end

function md_force!(
    force,
    provider::PseudofermionMDAction,
    U,
    workspace::SmearedPseudofermionMDWorkspace,
)
    length(force) == length(U) || throw(ArgumentError(
        "force and U must have the same number of directions",
    ))
    length(workspace.raw_derivative) == length(U) || throw(ArgumentError(
        "the pseudofermion MD workspace has the wrong number of directions",
    ))

    smeared_links, link_history, _ = calc_smearedU(U, provider.smearing)
    calc_UdSfdU!(
        workspace.raw_derivative,
        provider.action,
        smeared_links,
        provider.pseudofermion,
    )
    for direction in eachindex(U)
        mul!(
            workspace.smeared_derivative[direction],
            smeared_links[direction]',
            workspace.raw_derivative[direction],
        )
    end
    back_prop!(
        workspace.thin_derivative,
        workspace.smeared_derivative,
        provider.smearing,
        link_history,
        U,
    )

    for direction in eachindex(U)
        clear_U!(force[direction])
        mul!(
            workspace.conversion,
            U[direction],
            workspace.thin_derivative[direction],
        )
        Traceless_antihermitian_add!(
            force[direction],
            -1,
            workspace.conversion,
        )
    end
    return nothing
end
