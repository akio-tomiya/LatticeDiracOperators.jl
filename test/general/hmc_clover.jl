


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

    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
        set_wing_U!(Ufat[μ])
    end



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

    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
        set_wing_U!(Ufat[μ])
    end



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
