using LinearAlgebra: mul!

import Gaugefields:
    MDActionSet,
    MDDriver,
    NHYPSmearing,
    NHYPSmearingCache,
    Traceless_antihermitian_add!,
    clear_U!,
    md_action_workspace,
    md_force!,
    md_potential,
    nhyp_pullback!,
    nhyp_smear!,
    set_wing_U!

"""
    NHYPSmearedFermiAction(fermi_action, pseudofermion, smearing)
    NHYPSmearedFermiAction(fermi_action, pseudofermion;
                          alpha_outer=0.5, alpha_middle=0.5,
                          alpha_inner=0.4)

Wrap an LDO `FermiAction` as a Gaugefields molecular-dynamics action provider.
The fermion potential is evaluated on nHYP-smeared links and its force is
pulled back analytically to the thin links. Pass the result directly to
`md_driver`, either alone or as a member of an `MDActionSet`.

The pseudofermion is held by reference and must be refreshed before each HMC
trajectory with [`refresh_nhyp_pseudofermions!`](@ref). The current nHYP
implementation requires a four-dimensional LatticeMatrices-backed gauge
configuration with a nonzero halo.
"""
struct NHYPSmearedFermiAction{
    F<:FermiAction,
    P,
    S<:NHYPSmearing,
}
    fermi_action::F
    pseudofermion::P
    smearing::S
end

function NHYPSmearedFermiAction(
    fermi_action::FermiAction,
    pseudofermion;
    alpha_outer::Real=0.5,
    alpha_middle::Real=0.5,
    alpha_inner::Real=0.4,
)
    return NHYPSmearedFermiAction(
        fermi_action,
        pseudofermion,
        NHYPSmearing(; alpha_outer, alpha_middle, alpha_inner),
    )
end

struct _NHYPSmearedFermiActionMDWorkspace{V,C,T}
    smeared::V
    smearing_cache::C
    smeared_link_force::V
    smeared_cotangent::V
    thin_cotangent::V
    force_work::T
end

function md_action_workspace(action::NHYPSmearedFermiAction, U)
    return _NHYPSmearedFermiActionMDWorkspace(
        similar(U),
        NHYPSmearingCache(U, action.smearing),
        similar(U),
        similar(U),
        similar(U),
        similar(U[1]),
    )
end

function md_potential(
    action::NHYPSmearedFermiAction,
    U,
    workspace::_NHYPSmearedFermiActionMDWorkspace,
)
    nhyp_smear!(workspace.smeared, U, workspace.smearing_cache)
    return evaluate_FermiAction(
        action.fermi_action,
        workspace.smeared,
        action.pseudofermion,
    )
end

function md_force!(
    force,
    action::NHYPSmearedFermiAction,
    U,
    workspace::_NHYPSmearedFermiActionMDWorkspace,
)
    length(force) == length(U) || throw(ArgumentError(
        "force and U must have the same number of directions",
    ))
    nhyp_smear!(workspace.smeared, U, workspace.smearing_cache)
    calc_UdSfdU!(
        workspace.smeared_link_force,
        action.fermi_action,
        workspace.smeared,
        action.pseudofermion,
    )

    # LDO stores -1/2 of the usual link-gradient representation. Therefore
    # F' * V is the same -1/2-scaled Frobenius cotangent. The nHYP pullback
    # is linear, so retaining that scale produces the required LDO force on
    # the thin links.
    for direction in eachindex(U)
        mul!(
            workspace.smeared_cotangent[direction],
            workspace.smeared_link_force[direction]',
            workspace.smeared[direction],
        )
    end
    set_wing_U!(workspace.smeared_cotangent)
    nhyp_pullback!(
        workspace.thin_cotangent,
        workspace.smeared_cotangent,
        U,
        workspace.smearing_cache,
    )

    for direction in eachindex(U)
        mul!(
            workspace.force_work,
            U[direction],
            workspace.thin_cotangent[direction]',
        )
        clear_U!(force[direction])
        Traceless_antihermitian_add!(
            force[direction],
            -1,
            workspace.force_work,
        )
    end
    return nothing
end

"""
    refresh_nhyp_pseudofermions!(gaussian, action, U, workspace)

Refresh `gaussian` and the pseudofermion owned by `action` using nHYP-smeared
links. `workspace` is the action workspace returned by `md_action_workspace`.
The method returns the refreshed pseudofermion.
"""
function refresh_nhyp_pseudofermions!(
    gaussian,
    action::NHYPSmearedFermiAction,
    U,
    workspace::_NHYPSmearedFermiActionMDWorkspace,
)
    nhyp_smear!(workspace.smeared, U, workspace.smearing_cache)
    gauss_sampling_in_action!(gaussian, workspace.smeared, action.fermi_action)
    sample_pseudofermions!(
        action.pseudofermion,
        workspace.smeared,
        action.fermi_action,
        gaussian,
    )
    return action.pseudofermion
end

"""
    refresh_nhyp_pseudofermions!(gaussian, U, driver)
    refresh_nhyp_pseudofermions!(gaussian, U, driver, name)

Refresh the nHYP pseudofermion using the workspace already owned by an
`MDDriver`. Supply `name` when the provider is a named member of an
`MDActionSet`.
"""
function refresh_nhyp_pseudofermions!(
    gaussian,
    U,
    driver::MDDriver,
)
    action = driver.action
    action isa NHYPSmearedFermiAction || throw(ArgumentError(
        "the MDDriver action must be an NHYPSmearedFermiAction",
    ))
    return refresh_nhyp_pseudofermions!(
        gaussian,
        action,
        U,
        driver.action_workspace,
    )
end

function refresh_nhyp_pseudofermions!(
    gaussian,
    U,
    driver::MDDriver,
    name::Symbol,
)
    driver.action isa MDActionSet || throw(ArgumentError(
        "a named pseudofermion refresh requires an MDActionSet",
    ))
    hasproperty(driver.action.terms, name) || throw(ArgumentError(
        "the MDActionSet has no action named $name",
    ))
    action = getproperty(driver.action.terms, name)
    action isa NHYPSmearedFermiAction || throw(ArgumentError(
        "the action named $name must be an NHYPSmearedFermiAction",
    ))
    workspace = getproperty(driver.action_workspace.terms, name)
    return refresh_nhyp_pseudofermions!(gaussian, action, U, workspace)
end
