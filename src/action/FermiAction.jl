abstract type FermiAction{Dim,Dirac,fermion,gauge} end


include("./StaggeredFermiAction.jl")
include("./WilsonFermiAction.jl")
include("./DomainwallFermiAction.jl")
include("./WilsontypeFermiAction.jl")

function FermiAction(D::Dirac_operator{Dim},parameters_action;covneuralnet = nothing) where {NC,Dim}
    diractype = typeof(D)
    if covneuralnet ==  nothing
        hascovnet = false
    else
        hascovnet = true
    end

    if diractype <: Staggered_Dirac_operator
        return StaggeredFermiAction(D,hascovnet,covneuralnet,parameters_action)
    elseif diractype <: Wilson_Dirac_operator
        return WilsonFermiAction(D,hascovnet,covneuralnet,parameters_action)
    elseif diractype <: Domainwall_Dirac_operator
        return DomainwallFermiAction(D,hascovnet,covneuralnet) 
    elseif diractype <: Wilson_GeneralDirac_operator
        return Wilson_GeneralDirac_FermiAction(D,hascovnet,covneuralnet,parameters_action)
    else
        error("Action type $diractype is not supported")
    end
    
end

function evaluate_FermiAction(fermi_action::FermiAction,U,ϕ::AbstractFermionfields)
    error("evaluate_FermiAction(fermi_action,U,ϕ) is not implemented in type fermi_action:$(typeof(fermi_action)), U:$(typeof(U)), and ϕ:$(typeof(ϕ)),  ")
end

function gauss_sampling_in_action!(η::AbstractFermionfields,U,fermi_action::FermiAction)
    error("gauss_sampling_in_action!(η,fermi_action) is not implemented in type η:$(typeof(η)), fermi_action:$(typeof(fermi_action))")
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

function sample_pseudofermions!(ϕ::AbstractFermionfields,U,fermi_action::FermiAction,ξ) 
    error("sample_pseudofermions!(ϕ,fermi_action,ξ) is not implemented in type ϕ:$(typeof(ϕ)), fermi_action:$(typeof(fermi_action)), ξ:$(typeof(ξ))")
end

function calc_UdSfdU(fermi_action::FermiAction{Dim,Dirac,fermion,gauge},U::Vector{<: AbstractGaugefields},ϕ::AbstractFermionfields) where {Dim,Dirac,fermion,gauge}
    x = U[1]
    UdSfdU = Array{typeof(x),1}(undef,Dim)
    for μ=1:Dim
        UdSfdU[μ] = similar(x)
    end
    calc_UdSfdU!(UdSfdU,fermi_action,U,ϕ)
end

function calc_UdSfdU!(UdSfdU::Vector{<: AbstractGaugefields},fermi_action::FermiAction,U::Vector{<: AbstractGaugefields},ϕ::AbstractFermionfields)
    error("cald_UdSfdU!(UdSfdU,fermi_action,U) is not implemented in type UdSfdU:$(typeof(UdSfdU)), fermi_action:$(typeof(fermi_action)), U:$(typeof(U)), ϕ:$(typeof(ϕ))")
end

