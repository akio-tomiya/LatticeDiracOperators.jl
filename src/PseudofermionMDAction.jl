import Gaugefields:
    Traceless_antihermitian_add!,
    md_action_workspace,
    md_force!,
    md_potential

"""
    PseudofermionMDAction(action, pseudofermion)

Adapt an LDO pseudofermion `action` with a fixed `pseudofermion` field to the
Gaugefields molecular-dynamics action-provider interface.  The
pseudofermion field is held fixed during an MD trajectory; refresh it between
trajectories with [`refresh_pseudofermion!`](@ref).

The provider can be passed directly to `Gaugefields.md_driver`, or combined
with gauge and other pseudofermion actions using `Gaugefields.MDActionSet`.
"""
struct PseudofermionMDAction{A,P}
    action::A
    pseudofermion::P
end

struct PseudofermionMDWorkspace{D}
    derivative::D
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
    provider::PseudofermionMDAction,
    U,
    noise;
    kwargs...,
)
    gauss_sampling_in_action!(noise, U, provider.action; kwargs...)
    sample_pseudofermions!(
        provider.pseudofermion,
        U,
        provider.action,
        noise,
    )
    return provider
end

function md_action_workspace(provider::PseudofermionMDAction, U)
    isempty(U) && throw(ArgumentError(
        "a pseudofermion MD action requires at least one gauge direction",
    ))
    return PseudofermionMDWorkspace(map(similar, U))
end

function md_potential(
    provider::PseudofermionMDAction,
    U,
    workspace::PseudofermionMDWorkspace,
)
    return real(evaluate_FermiAction(
        provider.action,
        U,
        provider.pseudofermion,
    ))
end

function md_force!(
    force,
    provider::PseudofermionMDAction,
    U,
    workspace::PseudofermionMDWorkspace,
)
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
