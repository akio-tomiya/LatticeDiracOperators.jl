import .Rhmc:RHMC,get_order,get_α,get_β,get_α0

struct StaggeredFermiAction{Dim,Dirac,fermion,Nf} <: FermiAction{Dim,Dirac,fermion}
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
    Nf::Int64
    rhmc_info_for_action::Union{Nothing,RHMC}
    rhmc_info_for_MD::Union{Nothing,RHMC}
    _temporary_fermionfields::Vector{fermion}


    function StaggeredFermiAction(D::Dirac_operator{Dim},hascovnet,covneuralnet,parameters_action) where Dim
        @assert haskey(parameters_action,"Nf") "parameters for action should have the keyword Nf"
        Nf = parameters_action["Nf"]





        if Nf == 4 || Nf == 8 # 8 flavors if phi (Mdag M)^{-1} phi
            rhmc_info_for_action = nothing
            rhmc_info_for_MD = nothing
        else
            #for action: r_action
            #Nf = 8 -> alpha = 1 -> power x^{1/2} 
            #Nf = 2 -> alpha = 1/4 -> power x^1/8 
            #Nf = 1 -> alpha = 1/8  -> power x^1/16 
            order = Nf //16

            rhmc_info_for_action = RHMC(order,n=15)

            #for MD: r_MD
            #Nf = 8 -> alpha = 1 -> power x^{1} 
            #Nf = 2 -> alpha = 1/4 -> power x^1/4 
            #Nf = 1 -> alpha = 1/8  -> power x^1/8 
            order = Nf // 8
            #rhmcorder = 8 ÷ Nf
            rhmc_info_for_MD = RHMC(order,n=10)

            N_action = get_order(rhmc_info_for_action)
            N_MD = get_order(rhmc_info_for_MD)
        end

        if Nf == 4 
            num = 1
        elseif Nf == 8
            num = 1
        else
            num = maximum((N_action,N_MD))+1
        end

        x = D._temporary_fermi[1]
        xtype = typeof(x)
        _temporary_fermionfields = Array{xtype,1}(undef,num)
        for i=1:num
            _temporary_fermionfields[i] = similar(x)
        end

        return new{Dim,typeof(D),xtype,Nf}(hascovnet,covneuralnet,D,Nf,rhmc_info_for_action,rhmc_info_for_MD,_temporary_fermionfields)
    end
end

function gauss_sampling_in_action!(η::AbstractFermionfields,fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,Nf}) where {Dim,Dirac,fermion,Nf}
    gauss_distribution_fermion!(η)
end

function gauss_sampling_in_action!(η::AbstractFermionfields,fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,4}) where {Dim,Dirac,fermion}
    evensite = false
    W = fermi_action.diracoperator
    temp = fermi_action._temporary_fermionfields[1]
    gauss_distribution_fermion!(η)
    mul!(temp,W',η)
    clear_fermion!(temp,evensite)
    solve_DinvX!(η,W',temp)
end

function sample_pseudofermions!(ϕ::AbstractFermionfields,fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,8},ξ::AbstractFermionfields) where {Dim,Dirac,fermion}
    W = fermi_action.diracoperator
    solve_DinvX!(ϕ,W',ξ)
end

function sample_pseudofermions!(ϕ::AbstractFermionfields,fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,4},ξ::AbstractFermionfields) where {Dim,Dirac,fermion}
    W = fermi_action.diracoperator
    solve_DinvX!(ϕ,W',ξ)
end


function sample_pseudofermions!(ϕ::AbstractFermionfields,fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,Nf},ξ::AbstractFermionfields) where {Dim,Dirac,Nf,fermion}
    W = fermi_action.diracoperator
    WdagW = DdagD_Staggered_operator(W)

    rhmc = fermi_action.rhmc_info_for_action

    N = get_order(rhmc)
    
    x = fermi_action._temporary_fermionfields[1]
    vec_x = fermi_action._temporary_fermionfields[2:N+1]
    for j=1:N
        clear_fermion!(vec_x[j])
    end

    vec_β = get_β(rhmc)
    vec_α = get_α(rhmc)
    α0 = get_α0(rhmc)

    shiftedcg(vec_x,vec_β,x,WdagW,ξ,eps = W.eps_CG,maxsteps= W.MaxCGstep, verbose = W.verbose)
    clear_fermion!(ϕ)
    add_fermion!(ϕ,α0,ξ)
    for j=1:N
        αk = vec_α[j]
        add_fermion!(ϕ,αk,vec_x[j])
    end
    set_wing_fermion!(ϕ)

    #error("sample_pseudofermions!(ϕ,fermi_action) is not implemented in type ϕ:$(typeof(ϕ)), fermi_action:$(typeof(fermi_action))")
end


