include("../src/LatticeDiracOperators.jl")

using Gaugefields
using LinearAlgebra
using InteractiveUtils
using Random
using .LatticeDiracOperators
import Gaugefields:Initialize_4DGaugefields

function MDtest!(gauge_action,U,Dim,fermi_action,nn,fermi_action_eff,nn_eff,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 20
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    Random.seed!(123)

    numtrj = 400
    for itrj = 1:numtrj
       #@code_warntype MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        #error("cc")
        @time accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,nn,dSdU,fermi_action_eff,nn_eff,η,ξ)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,nn,dSdU,fermi_action_eff,nn_eff,η,ξ)
    Δτ = 1/MDsteps
    NC,_,NN... = size(U[1])
    for μ=1:Dim
        pwork = gauss_distribution(prod(NN)*(NC^2-1))
        substitute_U!(p[μ],pwork)
    end

    #println(p[1][1,1,1,1,1,1])

    #gauss_distribution!(p)
    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    
    substitute_U!(Uold,U)
    gauss_sampling_in_action!(ξ,Uout,fermi_action)
    sample_pseudofermions!(η,Uout,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println("Sfold = $Sfold")

    #@code_warntype calc_action(gauge_action,U,p) 

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)



        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        #        println(" p = ", p[1][1,1,1,1,1])
        #P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,fermi_action,η)
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn_eff,fermi_action_eff,η)
        #error("dd")

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end

    Uout_eff,Uout_multi_eff,_ = calc_smearedU(U,nn_eff)
    dSdp = get_parameter_derivatives(dSdU,nn_eff,Uout_multi_eff,U)

    Uout,Uout_multi,_ = calc_smearedU(U,nn)

    Sfnew = evaluate_FermiAction(fermi_action,Uout,η)
    Sfnew_eff = evaluate_FermiAction(fermi_action_eff,Uout_eff,η)
    println("Sfnew = $Sfnew")
    println("Sfnew_eff = $Sfnew_eff")
    println("Sfnew -Sfnew_eff = ",Sfnew- Sfnew_eff)
    eta = -1e-4
    for i=1:length(dSdp)
        println(i,"-th layer")
        println("dSdp[i] ",dSdp[i])
        println("dSdp[i]*(Sfnew- Sfnew_eff) ", dSdp[i]*(Sfnew- Sfnew_eff))
        diff = dSdp[i]*(Sfnew- Sfnew_eff)
        newρ = nn_eff[i].ρs .- eta*diff
        println("new: ", newρ)
        #set_parameters(nn_eff,i,newρ)
    end
    Snew = calc_action(gauge_action,U,p) + Sfnew
    
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")

    accept = exp(Sold - Snew) >= rand()

    #ratio = min(1,exp(Snew-Sold))
    if accept != true #rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end

function P_update_fermion!(U,p,ϵ,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    Uout,Uout_multi,_ = calc_smearedU(U,nn)

    #calc_p_UdSfdU!(p,fermi_action,U,η,factor)

    for μ=1:Dim
        calc_UdSfdU!(UdSfdUμ,fermi_action,Uout,η)
        mul!(dSdU[μ],Uout[μ]',UdSfdUμ[μ])
    end

    dSdUbare = back_prop(dSdU,nn,Uout_multi,U) 

    #calc_UdSfdU!(UdSfdUμ,fermi_action,U,η)

    for μ=1:Dim
        mul!(temps[1],U[μ],dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
        #Traceless_antihermitian_add!(p[μ],factor,UdSfdUμ[μ])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end
    
    
end

function test1()
    NX = 3
    NY = 3
    NZ = 3
    NT = 3
    Nwing = 1
    Dim = 4
    NC = 3

    U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    #U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    L5 = 4
    x = Initialize_pseudofermion_fields(U[1],"Domainwall",L5 = L5)


    params = Dict()
    params["Dirac_operator"] = "Domainwall"
    params["mass"] = 0.05
    params["L5"] = L5
    params["eps_CG"] = 1.0e-19
    params["verbose_level"] = 2
    #params["method_CG"] = "preconditiond_bicgstab"
    #params["method_CG"] = "bicgstab"
    params["method_CG"] = "bicg"
    D = Dirac_operator(U,x,params)

    nn = CovNeuralnet()
    ρ = [0.1]
    layername = ["plaquette"]
    L = [NX,NY,NZ,NT]
    st = STOUT_Layer(layername,ρ,L)
    push!(nn,st)

    nn_eff = deepcopy(nn)

    #=
    nn_eff = CovNeuralnet()
    #set_parameters(nn_eff,1,[0.1])
    st = STOUT_Layer(["plaquette","rectangular"],[0.1,0.001],L)
    push!(nn_eff,st)
    st2 = STOUT_Layer(["plaquette","rectangular"],[0.1,0.001],L)
    push!(nn_eff,st2)
    =#
    show(nn_eff)



    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)

    params_eff = deepcopy(params)
    params_eff["mass"] = 0.2
    D_eff = Dirac_operator(U,x,params_eff)
    
    fermi_action_eff = FermiAction(D_eff,parameters_action)
    #gauss_sampling_in_action!(x,U,fermi_action)
    #println("Sfold = ", dot(x,x))
    y = similar(x)

    Random.seed!(123)
    
    MDtest!(gauge_action,U,Dim,fermi_action,nn,fermi_action_eff,nn_eff,x,y)

end


function gauss_distribution(nv) 
    variance = 1
    nvh = div(nv,2)
    granf = zeros(Float64,nv)
    for i=1:nvh
        rho = sqrt(-2*log(rand())*variance)
        theta = 2pi*rand()
        granf[i] = rho*cos(theta)
        granf[i+nvh] = rho*sin(theta)
    end
    if 2*nvh == nv
        return granf
    end

    granf[nv] = sqrt(-2*log(rand())*variance) * cos(2pi*rand())
    return granf
end



test1()