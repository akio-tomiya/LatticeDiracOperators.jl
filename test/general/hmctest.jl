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
    UTA = Traceless_AntiHermitian(E)
    exptU!(Uout, UTA, t)
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

    #=
    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
        set_wing_U!(Ufat[μ])
    end

    =#

    phi1 = phitemps[3]
    phi2 = phitemps[4]
    dim = 4
    for μ = 1:dim
        shift_p = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        shift_m = ntuple(i -> ifelse(i == μ, -1, 0), dim)
        #substitute_U!(Ufat[μ], U[μ])
        shiftedadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
        #shiftedadd(y, Ufat[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
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
    #=
        for μ = 1:dim
            shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
            make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
            mul!(Ufat[μ], Uout, U[μ])
            set_wing_U!(Ufat[μ])
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
        shifteddagadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
        #shifteddagadd(y, Ufat[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
    end
end


function MDtest!(gauge_action, U, Dim, η, ξ, fermi_action, ξcpu, Ucpu,
    fermi_action_cpu, ηcpu, gauge_action_cpu)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    pcpu = initialize_TA_Gaugefields(Ucpu) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 

    Uold = similar(U)
    substitute_U!(Uold, U)
    Ucpuold = similar(Ucpu)
    substitute_U!(Ucpuold, Ucpu)

    if length(ARGS) < 1
        cpumode = false
        println("AD is used")
    else
        cpumode = parse(Bool, ARGS[1])
    end


    MDsteps = 10
    if cpumode
        temp1 = similar(Ucpu[1])
        temp2 = similar(Ucpu[1])
    else
        temp1 = similar(U[1])
        temp2 = similar(U[1])
    end
    comb = 2
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0
    Random.seed!(123)


    numtrj = 10
    for itrj = 1:numtrj
        #@code_warntype MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        #error("cc")
        @time accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, η, ξ, fermi_action,
            ξcpu, Ucpu, fermi_action_cpu, ηcpu, gauge_action_cpu, pcpu, Ucpuold)

        if cpumode
            @time accepted = MDstep!(gauge_action_cpu, Ucpu, pcpu, MDsteps, Dim, Ucpuold, ηcpu, ξcpu, fermi_action_cpu)
        else
            @time accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, η, ξ, fermi_action)
        end
        numaccepted += ifelse(accepted, 1, 0)

        if cpumode
            plaq_t = calculate_Plaquette(Ucpu, temp1, temp2) * factor
        else
            plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        end
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

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, η, ξ, fermi_action)
    Δτ = 1 / MDsteps
    NC, _, NN... = size(U[1])

    for μ = 1:Dim
        gauss_distribution!(p[μ])

    end

    substitute_U!(Uold, U)
    gauss_sampling_in_action!(ξ, U, fermi_action)

    sample_pseudofermions!(η, U, fermi_action, ξ)
    println(dot(η, η))

    Sfold = real(dot(ξ, ξ))
    #println_verbose_level2(U[1], "Sfold = $Sfold")

    #println("Sfold = $Sfold")

    #@code_warntype calc_action(gauge_action,U,p) 

    Sgold = calc_action(gauge_action, U, p)
    #println_verbose_level2(U[1], "Sfold = $Sfold Sgnew = $Sgold ")

    Sold = Sgold + Sfold
    #println_verbose_level2(U[1], "Sold = ", Sold)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
        P_update!(U, p, 1.0, Δτ, Dim, gauge_action)

        P_update_fermion!(U, p, 1.0, Δτ, Dim, gauge_action, fermi_action, η)
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
    end
    Sfnew = evaluate_FermiAction(fermi_action, U, η)
    Sgnew = calc_action(gauge_action, U, p)

    #println_verbose_level2(U[1], "Sfnew = $Sfnew Sgnew = $Sgnew ")
    #println("Sfnew = $Sfnew")
    Snew = Sgnew + Sfnew
    #println_verbose_level2(U[1], "Snew = ", Snew)


    println_verbose_level2(U[1], "Sold = $Sold, Snew = $Snew")
    #println("Sold = $Sold, Snew = $Snew")
    println_verbose_level2(U[1], "Snew - Sold = $(Snew-Sold)")
    #println("Snew - Sold = $(Snew-Sold)")
    #error("MD")

    accept = exp(Sold - Snew) >= rand()

    #ratio = min(1,exp(Snew-Sold))
    if accept != true #rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end



function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, η, ξ, fermi_action,
    ξcpu, Ucpu, fermi_action_cpu, ηcpu, gauge_action_cpu, pcpu, Ucpuold)
    Δτ = 1 / MDsteps
    NC, _, NN... = size(U[1])

    if length(ARGS) < 1
        cpumode = false
    else
        cpumode = parse(Bool, ARGS[1])
    end


    for μ = 1:Dim
        gauss_distribution!(p[μ])
        gauss_distribution!(pcpu[μ])
        #pwork = gauss_distribution(prod(NN) * (NC^2 - 1))
        #substitute_U!(p[μ], pwork)
    end

    #println(p[1][1,1,1,1,1,1])

    #gauss_distribution!(p)

    #substitute_U!(Ucpu, U)

    substitute_U!(Uold, U)
    substitute_U!(Ucpuold, Ucpu)

    gauss_sampling_in_action!(ξcpu, Ucpu, fermi_action_cpu)
    gauss_sampling_in_action!(ξ, U, fermi_action)
    #substitute_fermion!(ξ, ξcpu)
    #set_wing_fermion!(ξ)
    #set_wing_fermion!(ξcpu)
    #error("d")


    #sample_pseudofermions!(η, U, fermi_action, ξ)
    #println(dot(η, η))

    sample_pseudofermions!(ηcpu, Ucpu, fermi_action_cpu, ξcpu)
    sample_pseudofermions!(η, U, fermi_action, ξ)
    println(dot(ηcpu, ηcpu))
    println(dot(η, η))
    #substitute_fermion!(η, ηcpu)

    #Stest = evaluate_FermiAction(fermi_action, U, η)
    #println(Stest)
    #Stest = evaluate_FermiAction(fermi_action_cpu, Ucpu, ηcpu)
    #println("cpu: ", Stest)
    #return

    Sfcpuold = real(dot(ξcpu, ξcpu))
    Sfold = real(dot(ξ, ξ))
    println_verbose_level2(U[1], "Sfold = $Sfold")
    println_verbose_level2(U[1], "Sfcpuold = $Sfcpuold")

    #println("Sfold = $Sfold")

    #@code_warntype calc_action(gauge_action,U,p) 

    Sgold = calc_action(gauge_action, U, p)
    Sgcpuold = calc_action(gauge_action_cpu, Ucpu, pcpu)
    println_verbose_level2(U[1], "Sfold = $Sfold Sgold = $Sgold ")
    println_verbose_level2(U[1], "Sfcpuold = $Sfcpuold Sgcpuold = $Sgcpuold ")

    Sold = Sgold + Sfold
    Scpuold = Sgcpuold + Sfcpuold
    println_verbose_level2(U[1], "Sold = ", Sold)
    println_verbose_level2(U[1], "Scpuold = ", Scpuold)
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
        U_update!(Ucpu, pcpu, 0.5, Δτ, Dim, gauge_action_cpu)


        P_update!(U, p, 1.0, Δτ, Dim, gauge_action)
        P_update!(Ucpu, pcpu, 1.0, Δτ, Dim, gauge_action_cpu)
        #println(" U1 = ", U[1][1,1,1,1,1,1])
        #        println(" p = ", p[1][1,1,1,1,1])
        P_update_fermion!(Ucpu, pcpu, 1.0, Δτ, Dim, gauge_action_cpu, fermi_action_cpu, ηcpu)
        P_update_fermion!(U, p, 1.0, Δτ, Dim, gauge_action, η,
            ηcpu, Ucpu, fermi_action_cpu, gauge_action_cpu, fermi_action)


        #error("dd")

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
        U_update!(Ucpu, pcpu, 0.5, Δτ, Dim, gauge_action_cpu)
    end
    Sfnew = evaluate_FermiAction(fermi_action, U, η)
    Sgnew = calc_action(gauge_action, U, p)
    Sfcpunew = evaluate_FermiAction(fermi_action_cpu, Ucpu, ηcpu)
    Sgcpunew = calc_action(gauge_action_cpu, Ucpu, pcpu)
    println_verbose_level2(U[1], "Sfnew = $Sfnew Sgnew = $Sgnew ")
    println_verbose_level2(U[1], "Sfcpunew = $Sfcpunew Sgcpunew = $Sgcpunew ")
    #println("Sfnew = $Sfnew")
    Snew = Sgnew + Sfnew
    println_verbose_level2(U[1], "Snew = ", Snew)
    Scpunew = Sgcpunew + Sfcpunew
    println_verbose_level2(U[1], "Scpunew = ", Scpunew)

    println_verbose_level2(U[1], "Sold = $Sold, Snew = $Snew")
    println_verbose_level2(U[1], "Scpuold = $Scpuold, Scpunew = $Scpunew")
    #println("Sold = $Sold, Snew = $Snew")
    println_verbose_level2(U[1], "Snew - Sold = $(Snew-Sold)")
    println_verbose_level2(U[1], "Scpunew - Scpuold = $(Scpunew-Scpuold)")
    #println("Snew - Sold = $(Snew-Sold)")
    #error("MD")

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

function P_update_fermion!(U, p, ϵ, Δτ, Dim, gauge_action, fermi_action, η)  # p -> p +factor*U*dSdUμ

    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor = -ϵ * Δτ

    #clear_U!(UdSfdUμ)
    calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)


    for μ = 1:Dim
        Traceless_antihermitian_add!(p[μ], factor, UdSfdUμ[μ])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end

end

function Liederivative(X, μ, indices_halo, U, ϵ, η, fermi_action)
    Ut = similar(U)
    substitute_U!(Ut, U)
    Ui = U[μ].U.A[:, :, indices_halo...]
    Upi = exp(ϵ * X) * Ui
    Ut[μ].U.A[:, :, indices_halo...] .= Upi
    set_wing_U!(Ut[μ])

    Sp = evaluate_FermiAction(fermi_action, Ut, η)

    substitute_U!(Ut, U)
    Ui = U[μ].U.A[:, :, indices_halo...]
    Umi = exp(-ϵ * X) * Ui
    Ut[μ].U.A[:, :, indices_halo...] .= Umi
    set_wing_U!(Ut[μ])

    Sm = evaluate_FermiAction(fermi_action, Ut, η)
    dSdU = (Sp - Sm) / (2ϵ)
    return dSdU

end

#TA(M, NC) = (M - M') / 2 - I * tr((M - M') / NC) / 2

function TA(M, NC)
    A = (M - M') / 2
    A = A - (tr(A) / NC) * I
    return A
end

function P_update_fermion!(U, p, ϵ, Δτ, Dim, gauge_action, η,
    ηcpu, Ucpu, fermi_action_cpu, gauge_action_cpu, fermi_action)  # p -> p +factor*U*dSdUμ

    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    #UdSfdUμ = temps[1:Dim]
    UdSfdUμ = temps[1:Dim]
    Gaugefields.clear_U!(UdSfdUμ)
    #factor = -ϵ * Δτ
    factor_ad = -ϵ * Δτ #* (-0.5)

    #substitute_U!(Ucpu, U)
    tempscpu = get_temporary_gaugefields(gauge_action_cpu)
    UdSfdUμcpu = tempscpu[1:Dim]
    substitute_fermion!(η, ηcpu)
    substitute_U!(Ucpu, U)

    set_wing_U!(U)
    set_wing_U!(Ucpu)
    set_wing_fermion!(ηcpu)
    set_wing_fermion!(η)


    calc_UdSfdU!(UdSfdUμcpu, fermi_action_cpu, Ucpu, ηcpu)

    NX = U[1].NX
    NY = U[1].NY
    NZ = U[1].NZ
    NT = U[1].NT

    calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)
    nr = 1000
    fp = open("fnorm.txt", "w")

    #for k = 1:nr
    for ix = 1:NX
        for iy = 1:NY
            for iz = 1:NZ
                for it = 1:NT
                    for μ = 1:4
                        #ix = rand(1:NX)
                        #iy = rand(1:NY)
                        #iz = rand(1:NZ)
                        #it = rand(1:NT)
                        indices = (ix, iy, iz, it)
                        #println(indices)

                        indices_halo = Tuple(collect(indices) .+ 1)

                        #μ = rand(1:4)
                        UdSfdUμi = UdSfdUμ[μ].U.A[:, :, indices_halo...]
                        F = TA(UdSfdUμi, NC)

                        UdSfdUμicpu = UdSfdUμcpu[μ].U.A[:, :, indices_halo...]
                        Fcpu = TA(UdSfdUμicpu, NC)
                        diff = F - Fcpu
                        #display(F)
                        #display(Fcpu)
                        rel = sqrt(tr(diff' * diff)) / sqrt(tr(Fcpu' * Fcpu))
                        #println(rel)
                        println(fp, real(rel))
                    end

                end
            end
        end
    end
    close(fp)


    #=
    indices = (1, 2, 1, 3)
    indices_halo = Tuple(collect(indices) .+ 1)

    dS = zeros(ComplexF64, NC, NC)
    #dScpu = zeros(ComplexF64, NC, NC)
    Stest = evaluate_FermiAction(fermi_action, U, η)
    #Stestcpu = evaluate_FermiAction(fermi_action_cpu, Ucpu, ηcpu)
    Ut = similar(U)
    #Utcpu = similar(Ucpu)

    for i = 1:NC
        for j = 1:NC
            substitute_U!(Ut, U)
            #substitute_U!(Utcpu, Ucpu)
            set_wing_U!(Ut)
            #set_wing_U!(Utcpu)
            eta = 1e-4
            Ut[1].U.A[i, j, indices_halo...] += eta
            set_wing_U!(Ut)
            #Utcpu[1].U[i, j, indices...] += eta
            #set_wing_U!(Utcpu)
            reStest = evaluate_FermiAction(fermi_action, Ut, η)
            #reStestcpu = evaluate_FermiAction(fermi_action_cpu, Utcpu, ηcpu)

            substitute_U!(Ut, U)
            set_wing_U!(Ut)
            #substitute_U!(Utcpu, Ucpu)
            #set_wing_U!(Utcpu)
            Ut[1].U.A[i, j, indices_halo...] += im * eta
            #Utcpu[1].U[i, j, indices...] += im * eta
            set_wing_U!(Ut)
            #set_wing_U!(Utcpu)
            imStest = evaluate_FermiAction(fermi_action, Ut, η)
            #imStestcpu = evaluate_FermiAction(fermi_action_cpu, Utcpu, ηcpu)
            #println((Stest, reStest, imStest))
            dS[i, j] = (reStest - Stest) / eta + im * (imStest - Stest) / eta
            #dScpu[i, j] = (reStestcpu - Stestcpu) / eta + im * (imStestcpu - Stestcpu) / eta
        end
    end

    println("dS")
    display(dS .* (-0.5))
    #println("dScpu")
    # display(dScpu)

    #calc_UdSfdU!(UdSfdUμcpu, fermi_action_cpu, Ucpu, ηcpu)
    #substitute_U!(UdSfdUμ, UdSfdUμcpu)
    calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)

    #display(Ucpu[1].U[:, :, 2, 2, 2, 2]' * UdSfdUμcpu[1].U[:, :, 2, 2, 2, 2])
    #println("cpu")


    #display(UdSfdUμcpu[1].U[:, :, indices...])
    display(UdSfdUμ[1].U.A[:, :, indices_halo...])
    println("U")
    display(U[1].U.A[:, :, indices_halo...])
    println("UdU")
    display(U[1].U.A[:, :, indices_halo...]' * UdSfdUμ[1].U.A[:, :, indices_halo...])
    #println("Ucpu")
    #display(Ucpu[1].U[:, :, indices...])
    println(tr(UdSfdUμ[1]))
    #println(tr(UdSfdUμcpu[1]))

    #udsfdu = UdSfdUμcpu[1].U[:, :, indices...]
    #u = Ucpu[1].U[:, :, indices...]
    #println("udag udsfdu")
    #display(u' * udsfdu)
    error("d")

    =#

    #=
    σ = sqrt(1 / 2)

    NX = U[1].NX
    NY = U[1].NY
    NZ = U[1].NZ
    NT = U[1].NT


    calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)
    nr = 10
    ϵ = 1e-4
    ϵs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    fp2 = open("liederiv.txt", "w")
    for ϵ in ϵs
        println("ϵ = ", ϵ)
        fp = open("liederiv_$ϵ.txt", "w")
        for k = 1:nr
            ix = rand(1:NX)
            iy = rand(1:NY)
            iz = rand(1:NZ)
            it = rand(1:NT)
            indices = (ix, iy, iz, it)
            println(indices)

            indices_halo = Tuple(collect(indices) .+ 1)

            X = σ * randn(NC, NC) + im * σ * randn(NC, NC)
            X = TA(X, NC)
            X = X ./ sqrt(tr(X' * X))
            μ = rand(1:4)

            UdSfdUμi = UdSfdUμ[μ].U.A[:, :, indices_halo...]
            F = TA(UdSfdUμi, NC)
            d = real(tr(F * X)) * (-2)
            println("Lie AD: ", d)


            dSdUn = Liederivative(X, μ, indices_halo, U, ϵ, η, fermi_action)
            println("Lie: ", dSdUn)
            println("Lie AD: ", d, " Numerical ", dSdUn)
            println(fp, d, "\t", dSdUn, "\t", abs(d - dSdUn) / abs(d))
            println(fp2, ϵ, "\t", d, "\t", dSdUn, "\t", abs(d - dSdUn) / abs(d))

        end
        close(fp)
    end
    close(fp2)

    =#



    error("dd")


    #display(U[1].U.A[:, :, 3, 3, 3, 3]' * UdSfdUμ[1].U.A[:, :, 2+1, 2+1, 2+1, 2+1])
    #error("u")
    #println(tr(UdSfdUμcpu[1]))
    #println(tr(UdSfdUμ[1]))

    #error("dd")
    #dSFdU!(U, dSfdU, apply_D, apply_Ddag, η)
    #dSFdU!(U, dfdU, apply_D, apply_Ddag, x)
    #calc_p_UdSfdU!(p,fermi_action,U,η,factor)

    #calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)

    #display(UdSfdUμcpu[1].U[:, :, 1, 1, 1, 1])
    #display(UdSfdUμ[1].U.A[:, :, 2, 2, 2, 2])
    #substitute_U!(UdSfdUμ, UdSfdUμcpu)
    #error("dd")
    for μ = 1:Dim
        #mul!(temp1, U[μ], dSfdU[μ]')
        #display(temp1.U.A[:, :, 2, 2, 2, 2])
        #display(UdSfdUμcpu.U.A[:, :, 2, 2, 2, 2])
        Traceless_antihermitian_add!(p[μ], factor_ad, UdSfdUμ[μ])
        #Traceless_antihermitian_add!(p[μ], factor, UdSfdUμ[μ])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end

end

function test1()
    MPI.Init()
    NX = 16
    NY = 16
    NZ = 16
    NT = 16
    Nwing = 1
    Dim = 4
    NC = 3
    singleprecision = false

    #U = Initialize_4DGaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold")
    #Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    Ucpu = Initialize_Gaugefields(NC, 1, NX, NY, NZ, NT, condition="cold", isMPILattice=true)
    #U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold";
    #    isMPILattice=true, singleprecision)

    #U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold"; accelerator="cuda")     
    #U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold"; accelerator="jacc")    
    U = Initialize_Gaugefields(NC, 1, NX, NY, NZ, NT, condition="cold", isMPILattice=true)
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


    gsize = (NX, NY, NZ, NT)
    PEs = (1, 1, 1, 1)
    nw = 1
    elementtype = ComplexF64
    NG = 4

    x = GeneralFermion(NC, NG, gsize, PEs; nw, elementtype)

    xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson")
    #substitute_fermion!(x, xcpu)

    κ = 0.141139
    t = 0.1#0.0001
    parameters = (κ=κ, t=t)
    apply_D(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson!(y, U1, U2, U3, U4, x, parameters, phitemp, temp)
    apply_Ddag(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson_dag!(y, U1, U2, U3, U4, x, parameters, phitemp, temp)

    fermi_action = GeneralFermionAction(U, x, apply_D, apply_Ddag; numtemp=8, eps_CG=1e-15)

    #DdagD = DdagDgeneral(U, x, apply_D, apply_Ddag)

    #D = Dirac_operator(U, x, params)

    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = κ
    params["eps_CG"] = 1.0e-18
    params["improved gpu"] = false
    #params["improved gpu"] = true
    #params["eps_CG"] = 1.0e-1
    #params["verbose_level"] = 3
    #params["method_CG"] = "preconditiond_bicgstab"
    #params["method_CG"] = "bicgstab"
    params["method_CG"] = "bicg"
    #params["faster version"] = true

    Dcpu = Dirac_operator(Ucpu, xcpu, params)

    y = similar(x)
    ycpu = similar(xcpu)
    x = GeneralFermion(NC, NG, gsize, PEs; nw, elementtype)
    xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson")
    gauss_distribution_fermion!(xcpu)
    gauss_distribution_fermion!(x)
    #substitute_fermion!(x, xcpu)

    mul!(ycpu, Dcpu, xcpu)
    println(dot(ycpu, ycpu))

    phitemp = typeof(x)[]
    dphitemp = typeof(x)[]
    for i = 1:5
        push!(phitemp, zero(x))
        push!(dphitemp, zero(x))
    end
    temp = typeof(U[1])[]
    dtemp = typeof(U[1])[]
    for i = 1:8
        push!(temp, similar(U[1]))
        push!(dtemp, similar(U[1]))
    end

    apply_D(y, U[1], U[2], U[3], U[4], x, phitemp, temp)
    println(dot(y, y))

    #return


    parameters_actioncpu = Dict()
    fermi_action_cpu = FermiAction(Dcpu, parameters_actioncpu)



    MDtest!(gauge_action, U, Dim, x, y, fermi_action, ycpu, Ucpu, fermi_action_cpu, xcpu, gauge_action_cpu)

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