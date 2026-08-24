"""
    FermiAction(D, parameters_action; covneuralnet=nothing)

Construct the standard pseudofermion action associated with Dirac operator
`D`. Staggered and HISQ actions require `parameters_action["Nf"]`; Wilson and
domain-wall actions accept an empty dictionary. Use
[`evaluate_FermiAction`](@ref) for the action and [`calc_UdSfdU`](@ref) or
[`calc_UdSfdU!`](@ref) for its gauge force.
"""
abstract type FermiAction{Dim,Dirac,fermion,gauge} end

include("./StaggeredFermiAction.jl")
include("./HISQFermiForce.jl")
include("./WilsonFermiAction.jl")
include("./DomainwallFermiAction.jl")
include("./MobiusDomainwallFermiAction.jl")
include("./GeneralizedDomainwallFermiAction.jl")
include("./WilsontypeFermiAction.jl")
include("./GeneralFermionAction.jl")

function FermiAction(
    D::Dirac_operator{Dim},
    parameters_action;
    covneuralnet=nothing,
) where {Dim}
    diractype = typeof(D)
    if covneuralnet == nothing
        hascovnet = false
    else
        hascovnet = true
    end


    if diractype <: Staggered_Dirac_operators
        return StaggeredFermiAction(D, hascovnet, covneuralnet, parameters_action)
    elseif diractype <: Wilson_Dirac_operators
        return WilsonFermiAction(D, hascovnet, covneuralnet, parameters_action)
    elseif diractype <: Domainwall_Dirac_operator
        return DomainwallFermiAction(D, hascovnet, covneuralnet)
    elseif diractype <: MobiusDomainwall_Dirac_operator
        return MobiusDomainwallFermiAction(D, hascovnet, covneuralnet)
    elseif diractype <: GeneralizedDomainwall_Dirac_operator
        return GeneralizedDomainwallFermiAction(D, hascovnet, covneuralnet)
    elseif diractype <: Wilson_GeneralDirac_operator
        return Wilson_GeneralDirac_FermiAction(
            D,
            hascovnet,
            covneuralnet,
            parameters_action,
        )
    else
        error("Action type $diractype is not supported")
    end

end

"""
    evaluate_FermiAction(action, U, phi)

Evaluate the real pseudofermion action for gauge links `U` and pseudofermion
field `phi`. The action rebuilds its gauge-dependent operator from `U` before
solving.
"""
function evaluate_FermiAction(fermi_action::FermiAction, U, ϕ::AbstractFermionfields)
    error(
        "evaluate_FermiAction(fermi_action,U,ϕ) is not implemented in type fermi_action:$(typeof(fermi_action)), U:$(typeof(U)), and ϕ:$(typeof(ϕ)),  ",
    )
end

"""
    gauss_sampling_in_action!(noise, U, action; kwargs...)

Fill `noise` with the Gaussian field required by `action`. Standard
LatticeMatrices fields accept global-site RNG keywords such as `seed`,
`sweep`, and `subgroup`.
"""
function gauss_sampling_in_action!(
    η::AbstractFermionfields,
    U,
    fermi_action::FermiAction;
    kwargs...,
)
    error(
        "gauss_sampling_in_action!(η,fermi_action) is not implemented in type η:$(typeof(η)), fermi_action:$(typeof(fermi_action))",
    )
end

#=
 Conventional case: 
det(D)^Nf = det(D^+ D)^{Nf/2}
 = int dphi dphi^* exp[- phi^* (D^+ D)^{-1} phi] 
 = int dphi dphi^* exp[- phi^* D^{-1} (D^+)^{-1} phi] 

 RHMC case: 
det(D)^Nf = 
 = int dphi dphi^* exp[- phi^* D^{-Nf} phi]
 = int dphi dphi^* exp[- phi^* D^{-Nf/2} D^{-Nf/2} phi]
=#

"""
    sample_pseudofermions!(phi, U, action, noise)

Construct pseudofermion field `phi` from Gaussian `noise` for `action` and
gauge links `U`.
"""
function sample_pseudofermions!(ϕ::AbstractFermionfields, U, fermi_action::FermiAction, ξ)
    error(
        "sample_pseudofermions!(ϕ,fermi_action,ξ) is not implemented in type ϕ:$(typeof(ϕ)), fermi_action:$(typeof(fermi_action)), ξ:$(typeof(ξ))",
    )
end

"""
    calc_UdSfdU(action, U, phi)

Allocate and return the fermion force in LDO's existing
`Uμ (∂Sf/∂Uμ)†` convention. The standard Wilson, staggered, HISQ, and
domain-wall paths are analytic. Wilson--clover and `GeneralFermionAction`
callback forces require the optional Enzyme extension.
"""
function calc_UdSfdU(
    fermi_action::FermiAction{Dim,Dirac,fermion,gauge},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    x = U[1]
    UdSfdU = Array{typeof(x),1}(undef, Dim)
    for μ = 1:Dim
        UdSfdU[μ] = similar(x)
    end
    calc_UdSfdU!(UdSfdU, fermi_action, U, ϕ)
    return UdSfdU
end

"""
    calc_UdSfdU!(force, action, U, phi)

Write the fermion force into the preallocated vector `force`. See
[`calc_UdSfdU`](@ref) for the force convention and differentiation backends.
"""
function calc_UdSfdU!(
    UdSfdU::Vector{<:AbstractGaugefields},
    fermi_action::FermiAction,
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
)
    error(
        "cald_UdSfdU!(UdSfdU,fermi_action,U) is not implemented in type UdSfdU:$(typeof(UdSfdU)), fermi_action:$(typeof(fermi_action)), U:$(typeof(U)), ϕ:$(typeof(ϕ))",
    )
end
