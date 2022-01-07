abstract type FermiAction{Dim,Dirac,fermion} end


include("./StaggeredFermiAction.jl")

function FermiAction(D::Dirac_operator{Dim},parameters_action;covneuralnet = nothing) where {NC,Dim}
    diractype = typeof(D)
    if covneuralnet ==  nothing
        hascovnet = false
    else
        hascovnet = true
    end

    if diractype <: Staggered_Dirac_operator
        return StaggeredFermiAction(D,hascovnet,covneuralnet,parameters_action)
    else
        error("Action type $diractype is not supported")
    end
    
end

function gauss_sampling_in_action!(η::AbstractFermionfields,fermi_action::FermiAction)
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

function sample_pseudofermions!(ϕ::AbstractFermionfields,fermi_action::FermiAction,ξ) 
    error("sample_pseudofermions!(ϕ,fermi_action,ξ) is not implemented in type ϕ:$(typeof(ϕ)), fermi_action:$(typeof(fermi_action)), ξ:$(typeof(ξ))")
end

