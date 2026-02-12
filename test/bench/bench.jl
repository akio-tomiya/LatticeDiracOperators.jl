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
        shifteddagadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
        #shifteddagadd(y, Ufat[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
    end
end


function main()
    MPI.Init()
    NX = 24
    NY = 24
    NZ = 24
    NT = 24
    Nwing = 1
    Dim = 4
    NC = 3
    singleprecision = false

    Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="hot")
    U = Initialize_Gaugefields(NC, 1, NX, NY, NZ, NT, condition="hot", isMPILattice=true)
    substitute_U!(U,Ucpu)

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
    xwilson = Initialize_pseudofermion_fields(U[1], "Wilson")
    xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson")


    κ = 0.141139
    t = 0.1#0.0001
    parameters = (κ=κ, t=t)
    apply_D(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson!(y, U1, U2, U3, U4, x, parameters, phitemp, temp)
    apply_Ddag(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson_dag!(y, U1, U2, U3, U4, x, parameters, phitemp, temp)

    fermi_action = GeneralFermionAction(U, x, apply_D, apply_Ddag; numtemp=8, eps_CG=1e-15)


    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = κ
    params["eps_CG"] = 1.0e-18
    params["improved gpu"] = false
    params["method_CG"] = "bicg"

    Dcpu = Dirac_operator(Ucpu, xcpu, params)
    Dwilson = Dirac_operator(U, xwilson, params)

    parameters_actioncpu = Dict()
    fermi_action_cpu = FermiAction(Dcpu, parameters_actioncpu)
    fermi_action_wilson = FermiAction(Dwilson, parameters_actioncpu)

    nv = 10
    for k=1:nv
        Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="hot")
        substitute_U!(U,Ucpu)

        gauss_distribution_fermion!(xcpu)
        substitute_fermion!(x, xcpu)
        substitute_fermion!(xwilson, xcpu)
        set_wing_fermion!(xwilson)
        set_wing_fermion!(x)
        set_wing_fermion!(xcpu)
        set_wing_U!(U)
        set_wing_U!(Ucpu)

        println(dot(x,x))
        println(dot(xwilson,xwilson))
        println(dot(xcpu,xcpu))

        UdSfdUμcpu = similar(Ucpu)
        UdSfdUμwilson = similar(U)
        UdSfdUμAD = similar(U)
        clear_U!(UdSfdUμcpu)
        clear_U!(UdSfdUμwilson)
        clear_U!(UdSfdUμAD)

        ix = rand(1:NX)
        iy = rand(1:NY)
        iz = rand(1:NZ)
        it = rand(1:NT)
        indices = (ix, iy, iz, it)
        indices_halo = Tuple(collect(indices) .+ 1)


        println("UdSfdUμcpu")
        @time calc_UdSfdU!(UdSfdUμcpu, fermi_action_cpu, Ucpu, xcpu)
        println("UdSfdUμwilson")
        @time calc_UdSfdU!(UdSfdUμwilson, fermi_action_wilson, U, xwilson)


        for mu in 1:4
            a = UdSfdUμcpu[mu].U[:,:,indices...]
            b = Array(UdSfdUμwilson[mu].U.A[:,:,indices_halo...])
            A = TA(a, NC)
            B = TA(b, NC)

            #display(UdSfdUμcpu[mu].U[:,:,indices...])
            #display(UdSfdUμwilson[mu].U.A[:,:,indices_halo...])
            #display( UdSfdUμAD[mu].U.A[:,:,indices_halo...])
            diff1 = A-B

            display(A)
            display(B)
            println("Difference in UdSfdUwilson[$mu] at indices $indices: $(norm(diff1))")
        end 
        

        println("UdSfdUμAD")
        @time calc_UdSfdU!(UdSfdUμAD, fermi_action, U, x)


        for mu in 1:4
            a = UdSfdUμcpu[mu].U[:,:,indices...]
            c = Array(UdSfdUμAD[mu].U.A[:,:,indices_halo...])
            A = TA(a, NC)
            C = TA(c, NC)

            diff2 = A-C

            display(A)
            display(C)
            println("Difference in UdSfdUμ[$mu] at indices $indices: $(norm(diff2))")
        end 
        
    end



end
main()
