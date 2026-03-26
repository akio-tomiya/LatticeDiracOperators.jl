using Pkg
Pkg.activate("develop")
ENV["CUDA_LAUNCH_BLOCKING"]="1"
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
using CUDA
using Optimisers
#ENV["JULIA_DEBUG"] = "CUDA" 
const stt = 0

function MDtest!(gauge_action, U, Dim, fermi_action, η, ξ, ξcpu, Ucpu, params,
    fermi_action_cpu, ηcpu, gauge_action_cpu, D)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold, U)
    pcpu = initialize_TA_Gaugefields(Ucpu)
    MDsteps = 16
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0
    Random.seed!(123)
    b = 2.0
    c = 1.0
    W = [b,c]

    eta = 1e-1

    Optimiser = Optimisers.AMSGrad(eta)
    state = Optimisers.setup(Optimiser,W)

    dirname="confs"
    run(`mkdir -p $dirname`)

    hist = open("gpuml1/thermalhistory_mobius", "a")
    # accp = open("gpuml1/acceptance_mobius","w")
    btras = open("gpuml1/b1.txt", "a")
    ctras = open("gpuml1/c1.txt", "a")
    loss = open("gpuml1/lossfunction_mobius.txt", "a")
    dgdbs = open("gpuml1/dMdb1.txt", "a")
    dgdcs = open("gpuml1/dMdc1.txt", "a")
    ΔL = 0.0
    dMdb = 0.0
    dMdc = 0.0

    numtrj = 10
    for itrj = 1:numtrj
        #@code_warntype MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        #error("cc")
        @time accepted, dL, dMdb, dMdc = MDstep!(gauge_action, U, p, pcpu, MDsteps, Dim, Uold, fermi_action, η, ξ, 
            ξcpu, Ucpu, fermi_action_cpu, ηcpu, gauge_action_cpu, W, state, params, itrj, D)
        numaccepted += ifelse(accepted, 1, 0)

        plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        println_verbose_level1(U[1], "$itrj plaq_t = $plaq_t")
        println_verbose_level1(U[1], "acceptance ratio ", numaccepted / itrj)

        jtrj = itrj + stt

        println(hist, "$jtrj $plaq_t")
        println(btras, "$jtrj $(W[1])")
        println(ctras, "$jtrj $(W[2])")
        println(loss, "$dL")
        println(dgdbs, "$dMdb")
        println(dgdcs, "$dMdc")
        #println("$itrj plaq_t = $plaq_t")
        #println("acceptance ratio ",numaccepted/itrj)
        substitute_U!(Ucpu, U)
        flush(hist)
        flush(btras)
        flush(ctras)
        flush(loss)
        flush(dgdbs)
        flush(dgdcs)
        flush(stdout)

        jtrj += 200000
        filename = "mobiusconf_$jtrj.txt"
        save_textdata(Ucpu,joinpath(dirname,filename))
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


function MDstep!(gauge_action, U, p, pcpu, MDsteps, Dim, Uold, fermi_action, η, ξ, 
    ξcpu, Ucpu, fermi_action_cpu, ηcpu, gauge_action_cpu, W, state, params, traj, D)
    Δτ = 1 / MDsteps
    NC, _, NN... = size(U[1])


    for μ = 1:Dim
        # d = Normal(0.0, 1.0)
        # pwork = rand(d, NumofBasis, 1, NX, NY, NZ, NT)
        gauss_distribution!(p[μ])
        # pwork = gauss_distribution(prod(NN) * (NC^2 - 1))
        # substitute_U!(p, pwork)
    end
    # gauss_distribution!(p)

    #println(p[1][1,1,1,1,1,1])

    # gauss_distribution!(pcpu)
    # # LatticeDiracOperators.Dirac_operators.Elements!(p, pcpu)
    # substitute_U!(p, pcpu)

    substitute_U!(Uold, U)
    # gauss_sampling_in_action!(ξ, U, fermi_action)


    gauss_sampling_in_action!(ξcpu, Ucpu, fermi_action_cpu)
    substitute_fermion!(ξ, ξcpu)
    #set_wing_fermion!(ξ)
    #set_wing_fermion!(ξcpu)
    #error("d")


    sample_pseudofermions!(η, U, fermi_action, ξ)
    println(dot(η, η))

    # sample_pseudofermions!(ηcpu, Ucpu, fermi_action_cpu, ξcpu)
    # println(dot(ηcpu, ηcpu))

    Sfold = real(dot(ξ, ξ))
    println_verbose_level2(U[1], "Sfold = $Sfold")

    #println("Sfold = $Sfold")

    #@code_warntype calc_action(gauge_action,U,p) 

    Sold = calc_action(gauge_action, U, p) + Sfold
    println_verbose_level2(U[1], "Sold = ", Sold)
    #println("Sold = ",Sold)
    #error("debug")

    # params["b"] = W[1]
    # params["c"] = W[2]

    # println("Old b=$(W[1])")
    # println("Old c=$(W[2])")

    # η = Initialize_pseudofermion_fields(U[1],"MobiusDomainwall",L5=params["L5"], nowing=true)
    # D = Dirac_operator(U,η,params)
    # D = D(W[1], W[2])
    parameters_action = Dict()
    fermi_action = FermiAction(LatticeDiracOperators.Dirac_operators.Renew(D, W[1], W[2]), parameters_action)
    # D = D(b,c)


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

    if accept != true #rand() > ratio
        substitute_U!(U,Uold)
    end

    ΔL = 0.0
    dMdb = 0.0
    dMdc = 0.0

    # if traj > 0
        mres, dMdb, dMdc = LatticeDiracOperators.Dirac_operators.calc_mres_and_derivative(fermi_action, U, 10)
    #     ΔL = mres^2

    #     # if ΔL > 1e-4
        # gradW = [dMdb, dMdc]
        # gradW .*= mres * η.L5
        # state, W = Optimisers.update!(state, W, gradW)
        println("mres = $mres")
    #     # end

    #     println("Loss function = $ΔL")
        
    #     println("New b=$(W[1])")
    #     println("New c=$(W[2])")
    # end

    return accept, ΔL, dMdb, dMdc
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
    L = [NX, NY, NZ, NT]
    L5 = 8
    Nwing = 1
    Dim = 4
    NC = 3
    singleprecision = false

    #U = Initialize_4DGaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold")
    Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    #U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold";
        isMPILattice=true, singleprecision)

    #U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold"; accelerator="cuda")     
    #U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold"; accelerator="jacc")    
    #U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold") 
    #U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    #U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = 6.0 / 2
    push!(gauge_action, β, plaqloop)

    show(gauge_action)

    gauge_action_cpu = GaugeAction(Ucpu)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = 6.0 / 2
    push!(gauge_action_cpu, β, plaqloop)

    #is5D = false
    is5D = true
    x = Initialize_pseudofermion_fields(U[1], "MobiusDomainwall"; L5, is5D)
    # x = Initialize_pseudofermion_fields(U[1], "MobiusDomainwall"; L5, is5D, PEs=(1,1,1,4))

    xcpu = Initialize_pseudofermion_fields(Ucpu[1], "MobiusDomainwall"; L5)
    substitute_fermion!(x, xcpu)

    params = Dict()
    params["Dirac_operator"] = "MobiusDomainwall"
    params["κ"] = 0.141139 / 2
    params["eps_CG"] = 1.0e-16
    params["improved gpu"] = true
    #params["improved gpu"] = false
    params["mass"] = 0.02
    params["L5"] = L5
    # params["b"] = 2.14
    # params["c"] = 2.39
    params["b"] = 2.0
    params["c"] = 1.0
    params["M"] = -1.9
    #params["improved gpu"] = true
    #params["eps_CG"] = 1.0e-1
    #params["verbose_level"] = 3
    #params["method_CG"] = "preconditiond_bicgstab"
    #params["method_CG"] = "bicgstab"
    params["method_CG"] = "bicg"
    params["faster version"] = true
    D = Dirac_operator(U, x, params)

    parameters_action = Dict()
    fermi_action = FermiAction(D, parameters_action)
    y = similar(x)


    ycpu = similar(xcpu)
    params["improved gpu"] = false
    Dcpu = Dirac_operator(Ucpu, xcpu, params)

    params["improved gpu"] = true
    

    # load_BridgeText!("confs/mobiusconf_$stt.txt", Ucpu, L, NC)
    # substitute_U!(U, Ucpu)



    parameters_actioncpu = Dict()
    fermi_action_cpu = FermiAction(Dcpu, parameters_actioncpu)

    #return

    #=
    for i = 1:10
        gauss_sampling_in_action!(xcpu, Ucpu, fermi_action_cpu)
        substitute_fermion!(x, xcpu)
        println("new")
        @time mul!(y, D, x)
        println(dot(y, y))
        println("cpu")
        @time mul!(ycpu, Dcpu, xcpu)
        println(dot(ycpu, ycpu))
    end
    return
    =#

    MDtest!(gauge_action, U, Dim, fermi_action, x, y, ycpu, Ucpu, params, fermi_action_cpu, xcpu, gauge_action_cpu, D)

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
run(`mkdir -p gpuml1`)
println("4D HMC ")
test1()