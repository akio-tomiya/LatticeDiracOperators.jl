include("../src/LatticeDiracOperators.jl")

using Gaugefields
using LinearAlgebra
using InteractiveUtils
using Random
using .LatticeDiracOperators
import Gaugefields:Initialize_4DGaugefields

function MDtest!(gauge_action,U,Dim,fermi_action,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 10
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    Random.seed!(123)

    numtrj = 10
    for itrj = 1:numtrj
       #@code_warntype MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        #error("cc")
        @time accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
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


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
    Δτ = 1/MDsteps
    NC,_,NN... = size(U[1])
    
    for μ=1:Dim
        pwork = gauss_distribution(prod(NN)*(NC^2-1))
        substitute_U!(p[μ],pwork)
    end
    println(p[1][1,1,1,1,1,1])

    #gauss_distribution!(p)
    
    substitute_U!(Uold,U)
    gauss_sampling_in_action!(ξ,U,fermi_action)
    sample_pseudofermions!(η,U,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println("Sfold = $Sfold")

    #@code_warntype calc_action(gauge_action,U,p) 

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        #println(" U1 = ", U[1][1,1,1,1,1,1])
        #        println(" p = ", p[1][1,1,1,1,1])
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,fermi_action,η)
        #error("dd")

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Sfnew = evaluate_FermiAction(fermi_action,U,η)
    println("Sfnew = $Sfnew")
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

function P_update_fermion!(U,p,ϵ,Δτ,Dim,gauge_action,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    calc_UdSfdU!(UdSfdUμ,fermi_action,U,η)

    for μ=1:Dim
        Traceless_antihermitian_add!(p[μ],factor,UdSfdUμ[μ])
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

    U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    #U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    #Initialize_4DWilsonFermion(U[1])
    #error("init")
    #@code_warntype Initialize_pseudofermion_fields(U[1],"Wilson")
    #error("init")
    
    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.141139
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    #x = Initialize_pseudofermion_fields(U[1],params)

    #x = Initialize_4DWilsonFermion(U[1])
    x = Initialize_pseudofermion_fields(U[1],"Wilson")

    #=
    xplus = shift_fermion(x,1)
    

    ix = 1
    iy = 1
    iz = 1
    it = 1
    ialpha  =1
    xplus[1,ix,iy,iz,it,ialpha]
    println("dd")

    @inbounds for ix= 1:2
         @code_llvm x.f[1,ix,iy,iz,it,ialpha]
    end

    vv = rand(ComplexF64,2,2,2,2,2,2)
    @inbounds for ix= 1:2
        @code_llvm vv[1,ix,iy,iz,it,ialpha]
    end

    #error("err")
    @code_warntype shift_fermion(x,1)
    #error("2")
    ix = 1
    iy = 1
    iz = 1
    it = 1
    ialpha  =1
    @time @inbounds x.f[1,ix,iy,iz,it,ialpha]
    @code_llvm   @inbounds x.f[1,ix,iy,iz,it,ialpha]
    println("dd")
    @code_lowered   Base.getindex(x,1,ix,iy,iz,it,ialpha)
    println("dt")
    println(x[1,ix,iy,iz,it,ialpha])
    @code_llvm   @inbounds  x[1,ix,iy,iz,it,ialpha]

    @code_llvm  x[1,ix,iy,iz,it,ialpha]

    @code_llvm xplus[1,ix,iy,iz,it,ialpha]
    println(xplus[1,ix,iy,iz,it,ialpha])
    @code_llvm xplus[1,ix,iy,iz,it,ialpha]

    @code_warntype xplus[1,ix,iy,iz,it,ialpha]

    @time xplus[1,ix,iy,iz,it,ialpha]
    for ix = 1:4
        @time xplus.parent[1,ix,iy,iz,it,ialpha]
       # @time xplus[1,ix,iy,iz,it,ialpha]
    end
    #error("dd")
    #return 

    =#
    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.141139
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)


    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)
    #gauss_sampling_in_action!(x,U,fermi_action)
    #println("Sfold = ", dot(x,x))
    y = similar(x)

    
    MDtest!(gauge_action,U,Dim,fermi_action,x,y)

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

    granf[nv] = sqrt(-2*log(rand())*variance) * cos(2pi*urand())
    return granf
end



test1()