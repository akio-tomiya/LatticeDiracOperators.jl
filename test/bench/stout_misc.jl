import JACC
using Enzyme
JACC.@init_backend
using MPI
using Gaugefields
using LinearAlgebra
using InteractiveUtils
using Random
using LatticeDiracOperators
import Gaugefields: Initialize_4DGaugefields
using Test
import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!


function TA(M, NC)
    A = (M - M') / 2
    A = A - (tr(A) / NC) * I
    return A
end

function make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
    clear_U!(E)
    for ν = μ:dim
        if ν == μ
            continue
        end
        shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
        _calc_action_step_addsum!(E, C, D, U[μ], U[ν], shift_μ, shift_ν)
        #S += realtrace(Uout)
    end
    # _hmc_assert_finite_u("make_mu_loop:E_before_exp(mu=$μ)", E)
    UTA = Traceless_AntiHermitian(E)
    exptU!(Uout, UTA, t)
    # _hmc_assert_finite_u("make_mu_loop:Uout(mu=$μ)", Uout)
end

function _calc_action_step_addsum!(Uout, C, D, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_U(Uμ, shift_ν)
    Uν_pμ = shift_U(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(C, D, Uν')
    add_U!(Uout, C)
    #add_matrix!(Uout, C)
    #S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_U!(Uout, C)
    #S += realtrace(E)

    #return S
end


function shiftedadd(y, Uμ, x, γμ, shift_p, shift_m, phi1, phi2, κ)
    #U_n[ν](1 - γν) * ψ_{n + ν}
    mul_AshiftB!(phi1, Uμ, x, shift_p)
    mul!(phi2, phi1, transpose(I(4) - γμ))
    add_fermion!(y, phi2, -κ)

    # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
    mul_shiftAshiftB!(phi1, Uμ', x, shift_m, shift_m)
    mul!(phi2, phi1, transpose(I(4) + γμ))
    add_fermion!(y, phi2, -κ)
end

function stoutsmearing_U!(Ufat,U,t,Uout,C, D, E,dim)
    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
        set_wing_U!(Ufat[μ])
        # _hmc_assert_finite_u("apply_wilson:Ufat(mu=$μ)", Ufat[μ])
    end
end


#ψ_n - κ sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
function apply_wilson!(y, U1, U2, U3, U4, x, params, phitemps, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    Uout = temp[3]
    Ufat1 = temp[4]
    Ufat2 = temp[5]
    Ufat3 = temp[6]
    Ufat4 = temp[7]
    E = temp[8]


    clear_fermion!(y)
    add_fermion!(y, 1, x)
    γs = (γ1, γ2, γ3, γ4)
    κ = params.κ

    clear_U!(E)
    Ufat = (Ufat1, Ufat2, Ufat3, Ufat4)
    t = params.t
    dim = 4

    
    stoutsmearing_U!(Ufat,U,t,Uout,C, D, E,dim)

    #=
    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
        set_wing_U!(Ufat[μ])
        # _hmc_assert_finite_u("apply_wilson:Ufat(mu=$μ)", Ufat[μ])
    end
    =#
    


    phi1 = phitemps[3]
    phi2 = phitemps[4]
    dim = 4
    for μ = 1:dim
        shift_p = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        shift_m = ntuple(i -> ifelse(i == μ, -1, 0), dim)
        #substitute_U!(Ufat[μ], U[μ])
        #shiftedadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
        shiftedadd(y, Ufat[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
    end
end

function shifteddagadd(y, Uμ, x, γμ, shift_p, shift_m, phi1, phi2, κ)
    #U_n[ν](1 - γν) * ψ_{n + ν}
    mul_AshiftB!(phi1, Uμ, x, shift_p)
    mul!(phi2, phi1, transpose(I(4) + γμ))
    add_fermion!(y, -κ, phi2)

    # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
    mul_shiftAshiftB!(phi1, Uμ', x, shift_m, shift_m)
    mul!(phi2, phi1, transpose(I(4) - γμ))
    add_fermion!(y, -κ, phi2)
end


#ψ_n - κ sum_ν U_n[ν](1 + γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 - γν)*ψ_{n-ν}
function apply_wilson_dag!(y, U1, U2, U3, U4, x, params, phitemps, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    Uout = temp[3]
    Ufat1 = temp[4]
    Ufat2 = temp[5]
    Ufat3 = temp[6]
    Ufat4 = temp[7]
    E = temp[8]
    clear_U!(E)
    t = params.t

    Ufat = (Ufat1, Ufat2, Ufat3, Ufat4)
    dim = 4
    
    stoutsmearing_U!(Ufat,U,t,Uout,C, D, E,dim)

    #=
    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
        set_wing_U!(Ufat[μ])
        # _hmc_assert_finite_u("apply_wilson_dag:Ufat(mu=$μ)", Ufat[μ])
    end
    =#

    

    clear_fermion!(y)
    add_fermion!(y, 1, x)
    γs = (γ1, γ2, γ3, γ4)
    κ = params.κ

    phi1 = phitemps[3]
    phi2 = phitemps[4]
    dim = 4
    for μ = 1:dim
        shift_p = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        shift_m = ntuple(i -> ifelse(i == μ, -1, 0), dim)
        #substitute_U!(Ufat[μ], U[μ])
        #shifteddagadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
        shifteddagadd(y, Ufat[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
    end
end


function MDtest!(gauge_action,U,Dim,nn,fermi_action,η,ξ,
    gauge_action_ML,UML,fermi_action_ML,η_ML,ξ_ML,nn_ML)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    pML = initialize_TA_Gaugefields(UML) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 


    Uold = similar(U)
    Uold_ML = similar(UML)
    dSdU = similar(U)
    dSdU_ML = similar(UML)

    substitute_U!(Uold,U)
    substitute_U!(Uold_ML,UML)

    if length(ARGS) < 1
        cpumode = true
    else
        cpumode = parse(Bool, ARGS[1])
    end
    if cpumode
        println("CPU mode")
    else
        println("AD mode")
    end


    MDsteps = 60
    if cpumode
        temp1 = similar(U[1])
        temp2 = similar(U[1])
    else
        temp1 = similar(UML[1])
        temp2 = similar(UML[1])
    end
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    
    

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep_both!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ,
            gauge_action_ML,UML,pML,Uold_ML,nn_ML,dSdU_ML,fermi_action_ML,η_ML,ξ_ML)
        error("stout")

        if cpumode
            accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,
                            nn,dSdU,fermi_action,η,ξ)
        else
            accepted = MDstep_ML!(gauge_action_ML,UML,pML,MDsteps,
                    Dim,Uold_ML,
                    nn_ML,dSdU_ML,fermi_action_ML,η_ML,ξ_ML)
        end

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

function MDstep_both!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ,
    gauge_action_ML,UML,pML,Uold_ML,nn_ML,dSdU_ML,fermi_action_ML,η_ML,ξ_ML)

    NC = U[1].NC
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    dim = length(U)

    Uout = similar(U)
    Uout_ML = similar(UML)
    C = similar(UML[1])
    D = similar(UML[1])
    E = similar(UML[1])
    temp = similar(UML[1])
    t = nn_ML.t
    UdSfdUμML= similar(UML)
   
    indices = (1, 2, 1, 3)
    indices_halo = Tuple(collect(indices) .+ 1)
    
    substitute_U!(UML,U)

    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    display(Uout[1].U[:,:,indices...])
    stoutsmearing_U!(Uout_ML,UML,t,temp,C, D, E,dim)
    display(Uout_ML[1].U.A[:,:,indices_halo...])
    ##Sold = calc_action(gauge_action,Uout,p)
    error("U")

    substitute_U!(Uold,U)
    substitute_U!(Uold_ML,UML)

    gauss_sampling_in_action!(ξ,Uout,fermi_action)
    sample_pseudofermions!(η,Uout,fermi_action,ξ)
    substitute_fermion!(η_ML,η)
    substitute_fermion!(ξ_ML,ξ)


    Sfold = real(dot(ξ,ξ))
    SfoldML = real(dot(ξ_ML,ξ_ML))
    println("Sfold = $Sfold, SfoldML = $SfoldML")
    

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)

    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    UdSfdUμ = similar(U)
    dSdU = similar(U)

    for μ=1:Dim
        calc_UdSfdU!(UdSfdUμ,fermi_action,Uout,η)
        mul!(dSdU[μ],Uout[μ]',UdSfdUμ[μ])
    end
    dSdUbare = back_prop(dSdU,nn,Uout_multi,U) 


    temp2 = similar(U[1])
    for μ=1:Dim
        mul!(temp2,U[μ],dSdUbare[μ]) # U*dSdUμ
        a = temp2.U[:,:,indices...]
        a = TA(a,NC)
        display(a)
    end



    
    calc_UdSfdU!(UdSfdUμML, fermi_action_ML, UML, η_ML)
    for μ=1:Dim
        mul!(temp2,U[μ],dSdUbare[μ]) # U*dSdUμ
        a = temp2.U[:,:,indices...]
        a = TA(a,NC)
        display(a)

        b = UdSfdUμML[μ].U.A[:,:,indices_halo...]
        b = TA(b,NC)
        display(b)
        println("Difference in UdSfdUμML[$μ] at indices $indices: $(norm(a - b))")
    end

    error("d")



    error("test")


    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        #P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)
        P_update_fermion_ML!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    
    stoutsmearing_U!(Uout,U,t,temp,C, D, E,dim)
    #Uout,Uout_multi,_ = calc_smearedU(U,nn)
    #Snew = calc_action(gauge_action,Uout,p)

    Sfnew = evaluate_FermiAction(fermi_action,Uout,η)
    println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
     error("d")

     
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function MDstep_ML!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    dim = length(U)

    Uout = similar(U)
    C = similar(U[1])
    D = similar(U[1])
    E = similar(U[1])
    temp = similar(U[1])
    t = nn.t

    stoutsmearing_U!(Uout,U,t,temp,C, D, E,dim)

     #Uout,Uout_multi,_ = calc_smearedU(U,nn)
    ##Sold = calc_action(gauge_action,Uout,p)

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
        #P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)
        P_update_fermion_ML!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end

    stoutsmearing_U!(Uout,U,t,temp,C, D, E,dim)
    #Uout,Uout_multi,_ = calc_smearedU(U,nn)
    #Snew = calc_action(gauge_action,Uout,p)

    Sfnew = evaluate_FermiAction(fermi_action,Uout,η)
    println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
     error("d")

     
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
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
    ratio = min(1,exp(-Snew+Sold))
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

function P_update_fermion_ML!(U,p,ϵ,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)

    for μ=1:Dim
        Traceless_antihermitian_add!(p[μ], factor, UdSfdUμ[μ])
        #mul!(temps[1],U[μ],dSdUbare[μ]) # U*dSdUμ
        #Traceless_antihermitian_add!(p[μ],factor,temps[1])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end
end

function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0
    Dim = 4
    NC = 3

    Random.seed!(123)

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    UML  =Initialize_Gaugefields(NC,1,NX,NY,NZ,NT,condition = "hot",isMPILattice=true)
    substitute_U!(UML,U)

    gauge_action = GaugeAction(U)
    gauge_action_ML = GaugeAction(UML)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(gauge_action,β,plaqloop)
    push!(gauge_action_ML,β,plaqloop)

    show(gauge_action)

    L = [NX,NY,NZ,NT]
    nn = CovNeuralnet(U)
    t = 0.1
    ρ = [t]
    layername = ["plaquette"]
    #st = STOUT_Layer(layername,ρ,L)
    st = STOUT_Layer(layername, ρ, U)
    push!(nn,st)
    #push!(nn,st)

    x = Initialize_pseudofermion_fields(U[1],"Wilson")

    gsize = (NX, NY, NZ, NT)
    PEs = (1, 1, 1, 1)
    nw = 1
    elementtype = ComplexF64
    NG = 4
    x_ML = GeneralFermion(NC, NG, gsize, PEs; nw, elementtype)
    substitute_fermion!(x_ML, x)

    κ   = 0.141139

    parameters = (κ=κ, t=t)
    nn_ML = parameters
    apply_D(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson!(y, U1, U2, U3, U4, x, parameters, phitemp, temp)
    apply_Ddag(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson_dag!(y, U1, U2, U3, U4, x, parameters, phitemp, temp)

    eps_CG = 1e-8
    fermi_action_ML = GeneralFermionAction(UML, x_ML, apply_D, apply_Ddag; numtemp=12,  eps_CG)


    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = κ 
    params["eps_CG"] =  eps_CG
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)


    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)

    y = similar(x)
    y_ML = similar(x_ML)
    
    MDtest!(gauge_action,U,Dim,nn,fermi_action,x,y,gauge_action_ML,UML,fermi_action_ML,x_ML,y_ML,nn_ML)

end


test1()