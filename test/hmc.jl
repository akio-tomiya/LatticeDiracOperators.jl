using Gaugefields
using LinearAlgebra

function MDtest!(gauge_action,U,Dim,fermi_action,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 30
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        @time accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println_verbose_level1(U[1],"$itrj plaq_t = $plaq_t")
        println_verbose_level1(U[1],"acceptance ratio ",numaccepted/itrj)
    end
    @test numaccepted/numtrj > 0.7
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    
    substitute_U!(Uold,U)
    gauss_sampling_in_action!(ξ,U,fermi_action)
    sample_pseudofermions!(η,U,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println_verbose_level2(U[1],"Sfold = $Sfold")
    #println("Sfold = $Sfold")

    Sold = calc_action(gauge_action,U,p) + Sfold

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,fermi_action,η)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Sfnew = evaluate_FermiAction(fermi_action,U,η)
    println_verbose_level2(U[1],"Sfnew = $Sfnew")
    #println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    
    println_verbose_level2(U[1],"Sold = $Sold, Snew = $Snew")
    #println("Sold = $Sold, Snew = $Snew")
    println_verbose_level2(U[1],"Snew - Sold = $(Snew-Sold)")
    #println("Snew - Sold = $(Snew-Sold)")
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

function P_update_fermion!(U,p,ϵ,Δτ,Dim,gauge_action,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    calc_UdSfdU!(UdSfdUμ,fermi_action,U,η)

    for μ=1:Dim
        Traceless_antihermitian_add!(p[μ],factor,UdSfdUμ[μ])
    end
end

function test1_4D()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    x = Initialize_pseudofermion_fields(U[1],"staggered")
    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = 0.1
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)

    Nf = 2
    for Nf  in [8,4,2]
        println("Nf = $Nf")
        parameters_action = Dict()
        parameters_action["Nf"] = Nf
        fermi_action = FermiAction(D,parameters_action)
        gauss_sampling_in_action!(x,U,fermi_action)
        println("Sfold = ", dot(x,x))
        y = similar(x)

        MDtest!(gauge_action,U,Dim,fermi_action,x,y)
    end

end

function test1_2D()
    NX = 4
    NT = 4
    Nwing = 1
    Dim = 2
    NC = 3

    U  =Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette",Dim=Dim)
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    x = Initialize_pseudofermion_fields(U[1],"staggered")
    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = 0.1
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)

    Nf = 2
    for Nf  in [4,2,1]#[8,4,2]
        println("Nf = $Nf")
        parameters_action = Dict()
        parameters_action["Nf"] = Nf
        fermi_action = FermiAction(D,parameters_action)
        gauss_sampling_in_action!(x,U,fermi_action)
        println("Sfold = ", dot(x,x))
        y = similar(x)

        MDtest!(gauge_action,U,Dim,fermi_action,x,y)
    end

end

function test1_2D_NC(NC)
    NX = 4
    NT = 4
    Nwing = 1
    Dim = 2
    #NC = 2

    U  =Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette",Dim=Dim)
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    x = Initialize_pseudofermion_fields(U[1],"staggered")
    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = 0.1
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)

    Nf = 2
    for Nf  in [4,2,1]#[8,4,2]
        println("Nf = $Nf")
        parameters_action = Dict()
        parameters_action["Nf"] = Nf
        fermi_action = FermiAction(D,parameters_action)
        gauss_sampling_in_action!(x,U,fermi_action)
        println("Sfold = ", dot(x,x))
        y = similar(x)

        MDtest!(gauge_action,U,Dim,fermi_action,x,y)
    end

end

println("2D HMC ")
println("NC = 1")
NC = 1
test1_2D_NC(NC)
println("NC = 2")
NC = 2
test1_2D_NC(NC)
println("NC = 3")
NC = 3
test1_2D_NC(NC)

println("4D HMC ")
test1_4D()