import JACC
JACC.@init_backend
using Gaugefields
using LinearAlgebra
using InteractiveUtils
using Random
using LatticeDiracOperators
import Gaugefields: Initialize_4DGaugefields
using Test
import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!


function MDtest!(gauge_action, U, Dim, fermi_action, η, ξ, ξcpu, Ucpu,
    fermi_action_cpu, ηcpu, gauge_action_cpu)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold, U)
    MDsteps = 10 * 4
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 2
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0
    Random.seed!(123)

    numtrj = 10
    for itrj = 1:numtrj
        #@code_warntype MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        #error("cc")
        @time accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, fermi_action, η, ξ,
            ξcpu, Ucpu, fermi_action_cpu, ηcpu, gauge_action_cpu)
        numaccepted += ifelse(accepted, 1, 0)

        plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        println_verbose_level1(U[1], "$itrj plaq_t = $plaq_t")
        println_verbose_level1(U[1], "acceptance ratio ", numaccepted / itrj)
        #println("$itrj plaq_t = $plaq_t")
        #println("acceptance ratio ",numaccepted/itrj)
    end
    @test numaccepted / numtrj > 0.8
end

function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, fermi_action, η, ξ,
    ξcpu, Ucpu, fermi_action_cpu, ηcpu, gauge_action_cpu)
    Δτ = 1 / MDsteps
    NC, _, NN... = size(U[1])


    for μ = 1:Dim
        gauss_distribution!(p[μ])
        #pwork = gauss_distribution(prod(NN) * (NC^2 - 1))
        #substitute_U!(p[μ], pwork)
    end

    #println(p[1][1,1,1,1,1,1])

    #gauss_distribution!(p)

    substitute_U!(Uold, U)
    gauss_sampling_in_action!(ξ, U, fermi_action)


    gauss_sampling_in_action!(ξcpu, Ucpu, fermi_action_cpu)
    substitute_fermion!(ξ, ξcpu)
    #set_wing_fermion!(ξ)
    #set_wing_fermion!(ξcpu)
    #error("d")


    sample_pseudofermions!(η, U, fermi_action, ξ)
    println(dot(η, η))

    sample_pseudofermions!(ηcpu, Ucpu, fermi_action_cpu, ξcpu)
    println(dot(ηcpu, ηcpu))

    Sfold = real(dot(ξ, ξ))
    println_verbose_level2(U[1], "Sfold = $Sfold")

    #println("Sfold = $Sfold")

    #@code_warntype calc_action(gauge_action,U,p) 

    Sold = calc_action(gauge_action, U, p) + Sfold
    println_verbose_level2(U[1], "Sold = ", Sold)
    #println("Sold = ",Sold)
    #error("debug")

    #=
    W = fermi_action.diracoperator(U)
    WdagW = LatticeDiracOperators.Dirac_operators.DdagD_Wilson_operator(W)
    X, it_X = get_temp(fermi_action._temporary_fermionfields)
    Y, it_Y = get_temp(fermi_action._temporary_fermionfields)
    println("---------------------")
    println("---------------------")
    println("---------------------")
    println("---------------------mul")
    mul!(X, WdagW, η)
    println("---------------------")
    println("---------------------")
    println("---------------------")
    println("---------------------")
    set_wing_fermion!(X)
    println("---------------------")
    println("---------------------")
    println("---------------------")
    println("---------------------")
    Wcpu = fermi_action_cpu.diracoperator(Ucpu)
    WdagW = LatticeDiracOperators.Dirac_operators.DdagD_Wilson_operator(Wcpu)
    X, it_X = get_temp(fermi_action_cpu._temporary_fermionfields)
    Y, it_Y = get_temp(fermi_action_cpu._temporary_fermionfields)
    println("---------------------")
    println("---------------------")
    println("---------------------")
    println("---------------------mul")
    mul!(X, WdagW, ηcpu)
    println("---------------------")
    println("---------------------")
    println("---------------------")
    set_wing_fermion!(X)
    println("---------------------")
    println("---------------------")
    println("---------------------")
    error("dd")
    =#



    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)


        P_update!(U, p, 1.0, Δτ, Dim, gauge_action)
        #println(" U1 = ", U[1][1,1,1,1,1,1])
        #        println(" p = ", p[1][1,1,1,1,1])
        P_update_fermion!(U, p, 1.0, Δτ, Dim, gauge_action, fermi_action, η,
            ηcpu, Ucpu, fermi_action_cpu, gauge_action_cpu)
        #error("dd")

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
    end
    Sfnew = evaluate_FermiAction(fermi_action, U, η)
    println_verbose_level2(U[1], "Sfnew = $Sfnew")
    #println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action, U, p) + Sfnew

    println_verbose_level2(U[1], "Sold = $Sold, Snew = $Snew")
    #println("Sold = $Sold, Snew = $Snew")
    println_verbose_level2(U[1], "Snew - Sold = $(Snew-Sold)")
    #println("Snew - Sold = $(Snew-Sold)")

    accept = exp(Sold - Snew) >= rand()

    #ratio = min(1,exp(Snew-Sold))
    if accept != true #rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function U_update!(U, p, ϵ, Δτ, Dim, gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)

    end
end

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temps[1], U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temps[1])
    end
end

function P_update_fermion!(U, p, ϵ, Δτ, Dim, gauge_action, fermi_action, η,
    ηcpu, Ucpu, fermi_action_cpu, gauge_action_cpu)  # p -> p +factor*U*dSdUμ

    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor = -ϵ * Δτ

    #substitute_U!(Ucpu, U)
    #tempscpu = get_temporary_gaugefields(gauge_action_cpu)
    #UdSfdUμcpu = tempscpu[1:Dim]
    #calc_UdSfdU!(UdSfdUμcpu, fermi_action_cpu, Ucpu, ηcpu)

    #calc_p_UdSfdU!(p,fermi_action,U,η,factor)
    calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)

    #display(UdSfdUμcpu[1].U[:, :, 1, 1, 1, 1])
    #display(UdSfdUμ[1].U.A[:, :, 2, 2, 2, 2])
    #substitute_U!(UdSfdUμ, UdSfdUμcpu)
    #error("dd")
    for μ = 1:Dim
        Traceless_antihermitian_add!(p[μ], factor, UdSfdUμ[μ])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end

end

function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3
    singleprecision = false

    #U = Initialize_4DGaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold")
    Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold";
        isMPILattice=true, singleprecision)
    #U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    #U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = 5.5 / 2
    push!(gauge_action, β, plaqloop)

    show(gauge_action)

    gauge_action_cpu = GaugeAction(Ucpu)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = 5.5 / 2
    push!(gauge_action_cpu, β, plaqloop)

    x = Initialize_pseudofermion_fields(U[1], "Wilson")
    xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson")
    substitute_fermion!(x, xcpu)

    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.141139 / 2
    params["eps_CG"] = 1.0e-18
    #params["eps_CG"] = 1.0e-1
    #params["verbose_level"] = 3
    #params["method_CG"] = "preconditiond_bicgstab"
    #params["method_CG"] = "bicgstab"
    params["method_CG"] = "bicg"
    #params["faster version"] = true
    D = Dirac_operator(U, x, params)

    parameters_action = Dict()
    fermi_action = FermiAction(D, parameters_action)
    y = similar(x)
    ycpu = similar(xcpu)

    Dcpu = Dirac_operator(Ucpu, xcpu, params)
    parameters_actioncpu = Dict()
    fermi_action_cpu = FermiAction(Dcpu, parameters_actioncpu)

    MDtest!(gauge_action, U, Dim, fermi_action, x, y, ycpu, Ucpu, fermi_action_cpu, xcpu, gauge_action_cpu)

end


function gauss_distribution(nv)
    variance = 1
    nvh = div(nv, 2)
    granf = zeros(Float64, nv)
    for i = 1:nvh
        rho = sqrt(-2 * log(rand()) * variance)
        theta = 2pi * rand()
        granf[i] = rho * cos(theta)
        granf[i+nvh] = rho * sin(theta)
    end
    if 2 * nvh == nv
        return granf
    end

    granf[nv] = sqrt(-2 * log(rand()) * variance) * cos(2pi * urand())
    return granf
end


#println("2D HMC ")
#test1_2D()

println("4D HMC ")
test1()