import JACC
using Random
using Enzyme
JACC.@init_backend
using MPI
using Gaugefields
using LatticeDiracOperators
using PreallocatedArrays
using LinearAlgebra


function _calc_action_matrixadd!(Uout, C, D, Uμ, Uν, shift_μ, shift_ν)
    #clear_U!(E)
    Uμ_pν = shift_U(Uμ, shift_ν)
    Uν_pμ = shift_U(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(C, D, Uν')
    add_U!(Uout, C)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_U!(Uout, C)
    #S += realtrace(E)
    return
end


function calc_action(U1, U2, U3, U4, β, NC, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    S = 0.0

    Uout = temp[3]
    clear_U!(Uout)

    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            _calc_action_matrixadd!(Uout, C, D, U[μ], U[ν], shift_μ, shift_ν)
        end
    end
    S = realtrace(Uout)
    return -S * β / NC
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

    clear_fermion!(y)
    add_fermion!(y, 1, x)
    γs = (γ1, γ2, γ3, γ4)
    κ = params.κ

    dim = 4

    phi1 = phitemps[3]
    phi2 = phitemps[4]
    dim = 4
    for μ = 1:dim
        shift_p = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        shift_m = ntuple(i -> ifelse(i == μ, -1, 0), dim)
        shiftedadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
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

    dim = 4

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
        shifteddagadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
    end
end


function MDstep!(U, p, MDsteps, Dim, Uold, tempvec, β, η, ξ, fermi_action)
    NC = U[1].NC
    Δτ = 1.0 / MDsteps
    temp1, it_temp1 = get_block(tempvec)
    temp, its_temp = get_block(tempvec, 9)
    dtemp, its_dtemp = get_block(tempvec, 9)


    gauss_distribution!(p)
    gauss_sampling_in_action!(ξ, U, fermi_action)
    sample_pseudofermions!(η, U, fermi_action, ξ)
    Sfold = real(dot(ξ, ξ))
    println("Sfold ", Sfold)


    Sold = calc_action(U..., β, NC, temp) + p * p / 2 + Sfold
    substitute_U!(Uold, U)




    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, tempvec)
        P_update!(U, p, 1.0, Δτ, Dim, temp1, temp, dtemp, tempvec, β)

        P_update_fermion!(U, p, 1.0, Δτ, Dim, tempvec, fermi_action, η)
        U_update!(U, p, 0.5, Δτ, Dim, tempvec)
    end
    Sfnew = evaluate_FermiAction(fermi_action, U, η)

    Snew = calc_action(U..., β, NC, temp) + p * p / 2 + Sfnew
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))

    unused!(tempvec, it_temp1)
    unused!(tempvec, its_temp)
    unused!(tempvec, its_dtemp)


    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function U_update!(U, p, ϵ, Δτ, Dim, tempvec)
    temp1, it_temp1 = get_block(tempvec)
    temp2, it_temp2 = get_block(tempvec)
    expU, it_expU = get_block(tempvec)
    W, it_W = get_block(tempvec)


    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)
    end
    unused!(tempvec, it_temp1)
    unused!(tempvec, it_temp2)
    unused!(tempvec, it_expU)
    unused!(tempvec, it_W)

end

function P_update_fermion!(U, p, ϵ, Δτ, Dim, temps, fermi_action, η)  # p -> p +factor*U*dSdUμ

    UdSfdUμ, it_UdSfdUμ = get_block(temps, Dim)
    factor = -ϵ * Δτ

    calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)

    for μ = 1:Dim
        Traceless_antihermitian_add!(p[μ], factor, UdSfdUμ[μ])
    end

    unused!(temps, it_UdSfdUμ)
end


function P_update!(U, p, ϵ, Δτ, Dim, temp1, temp, dtemp, temps, β) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    factor_ad = ϵ * Δτ / 2

    dSdU, it_dSdU = get_block(temps, 4)#temps[end]
    Gaugefields.clear_U!(dSdU)

    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]
    set_wing_U!(U)
    Enzyme_derivative!(
        calc_action,
        U1, U2, U3, U4,
        dSdU[1], dSdU[2], dSdU[3], dSdU[4], nodiff(β), nodiff(NC);
        temp,
        dtemp
    )

    for μ = 1:Dim
        mul!(temp1, U[μ], dSdU[μ]')
        Traceless_antihermitian_add!(p[μ], factor_ad, temp1)
    end

    unused!(temps, it_dSdU)
end

function HMC_test_4D(NX, NY, NZ, NT, NC, β)
    Dim = 4
    Nwing = 1
    Random.seed!(123)
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="cold";
        isMPILattice=true)


    comb = 6 #4*3/2
    factor = 1 / (comb * U[1].NV * U[1].NC)
    tempvec = PreallocatedArray(U[1]; num=30, haslabel=false)
    temp1, ittemp1 = get_block(tempvec)
    temp2, ittemp2 = get_block(tempvec)

    gsize = (NX, NY, NZ, NT)
    PEs = (1, 1, 1, 1)
    nw = 1
    elementtype = ComplexF64
    NG = 4
    x = GeneralFermion(NC, NG, gsize, PEs; nw, elementtype)

    κ = 0.141139
    parameters = (κ=κ,)
    apply_D(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson!(y, U1, U2, U3, U4, x, parameters, phitemp, temp)
    apply_Ddag(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson_dag!(y, U1, U2, U3, U4, x, parameters, phitemp, temp)
    eps_CG = 1e-16
    fermi_action = GeneralFermionAction(U, x, apply_D, apply_Ddag; numtemp=8, eps_CG)

    η = x
    ξ = similar(x)


    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    β = β / 2
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold, U)
    MDsteps = 10
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        @time accepted = MDstep!(U, p, MDsteps, Dim, Uold, tempvec, β, η, ξ, fermi_action)

        numaccepted += ifelse(accepted, 1, 0)

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temp1, temp2)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ", numaccepted / itrj)
        end
    end
    unused!(tempvec, ittemp1)
    unused!(tempvec, ittemp2)

    return plaq_t, numaccepted / numtrj
end

function main()
    β = 6
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    HMC_test_4D(NX, NY, NZ, NT, NC, β)
end
main()