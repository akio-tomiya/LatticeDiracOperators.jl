using Gaugefields
using LinearAlgebra
using LatticeDiracOperators

function MDtest!(gauge_action,U,Dim,nn,fermi_action,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)
    
    substitute_U!(Uold,U)
    MDsteps = 10
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ)
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


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ)
    

    Δτ = 1/MDsteps
    gauss_distribution!(p)


    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    #Sold = calc_action(gauge_action,Uout,p)

    substitute_U!(Uold,U)

    gauss_sampling_in_action!(ξ,Uout,fermi_action)
    sample_pseudofermions!(η,Uout,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println("Sfold = $Sfold")

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)


    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end

    
    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    #Snew = calc_action(gauge_action,Uout,p)

    Sfnew = evaluate_FermiAction(fermi_action,Uout,η)
    println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(Snew-Sold))
    if rand() > ratio
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

    #for μ=1:4
    #    println("μ = ",μ)
    #println(Uout[μ])
    #end


    for μ=1:Dim
        calc_UdSfdU!(UdSfdUμ,fermi_action,Uout,η)
        mul!(dSdU[μ],Uout[μ]',UdSfdUμ[μ])
    end

    dSdUbare = back_prop(dSdU,nn,Uout_multi,U) 
    

    for μ=1:Dim
        mul!(temps[1],U[μ],dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end
end

function test1()
    NX = 2
    NY = 2
    NZ = 2
    NT = 2
    Nwing = 1
    Dim = 4
    NC = 3

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(gauge_action,β,plaqloop)

    show(gauge_action)

    L = [NX,NY,NZ,NT]
    nn = CovNeuralnet()
    ρ = [0.1,0.1]
    layername = ["plaquette","rectangular"]
    st = STOUT_Layer(layername,ρ,L)
    push!(nn,st)
    #push!(nn,st)

    x = Initialize_pseudofermion_fields(U[1],"Wilson")


    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.141139
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)


    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)

    y = similar(x)
    

    MDtest!(gauge_action,U,Dim,nn,fermi_action,x,y)

end


test1()