using LatticeMatrices
using MPI
using LinearAlgebra
using Enzyme
import JACC
using LatticeDiracOperators
JACC.@init_backend
using Gaugefields

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


function g(χ, U1, U2, U3, U4, η, p, apply, phitemp, temp)
    phitemp1 = phitemp[end]
    apply(phitemp1, U1, U2, U3, U4, η, p, phitemp, temp)
    #Dmul!(phitemp1, U1, U2, U3, U4, D, η)
    s = -2 * real(dot(χ, phitemp1))
    return s
end


function main()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    NG = 4
    MPI.Init()

    gsize = (NX, NY, NZ, NT)
    dim = length(gsize)
    PEs = (1, 1, 1, 1)
    nw = 1
    elementtype = ComplexF64

    U = Initialize_Gaugefields(NC, nw, NX, NY, NZ, NT,
        condition="hot";
        isMPILattice=true)

    x = GeneralFermion(NC, NG, gsize, PEs; nw, elementtype)
    y = zero(x)
    z = similar(x)
    substitute_fermion!(y, x)

    gauss_distribution_fermion!(x)
    gauss_distribution_fermion!(y)
    s = dot(x, y)

    mul!(z, U[1], x)
    println(s)
    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]

    κ = 0.141139
    params = (κ=κ,)

    phitemp = typeof(x)[]
    dphitemp = typeof(x)[]
    for i = 1:5
        push!(phitemp, zero(x))
        push!(dphitemp, zero(x))
    end
    temp = typeof(U[1])[]
    dtemp = typeof(U[1])[]
    for i = 1:5
        push!(temp, similar(U[1]))
        push!(dtemp, similar(U[1]))
    end

    apply_wilson!(y, U1, U2, U3, U4, x, params, phitemp, temp)

    χ = similar(x)
    η = similar(x)
    gauss_distribution_fermion!(χ)
    gauss_distribution_fermion!(η)


    apply_D(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson!(y, U1, U2, U3, U4, x, params, phitemp, temp)
    apply_Ddag(y, U1, U2, U3, U4, x, phitemp, temp) = apply_wilson_dag!(y, U1, U2, U3, U4, x, params, phitemp, temp)


    fermi_action = GeneralFermionAction(U, x, apply_D, apply_Ddag)

    gauss_distribution_fermion!(x)
    set_wing_fermion!(x)

    UdSfdU = similar(U)

    println(dot(x, x))

    Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT,
        condition="hot")
    substitute_U!(U, Ucpu)
    xw = Initialize_pseudofermion_fields(U[1], "Wilson")
    gauss_distribution_fermion!(xw)
    println(dot(xw, xw))


    yw = similar(xw)
    xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson")
    gauss_distribution_fermion!(xcpu)
    println(dot(xcpu, xcpu))
    set_wing_fermion!(xw)
    substitute_fermion!(xw, xcpu)
    #substitute_fermion!(x, xcpu)
    #substitute_fermion!(x, xw)


    calc_UdSfdU!(
        UdSfdU,
        fermi_action,
        U,
        x,
    )
    return


    paramsw = Dict()
    paramsw["Dirac_operator"] = "Wilson"
    paramsw["κ"] = κ
    paramsw["eps_CG"] = 1.0e-18
    paramsw["faster version"] = true
    Dw = Dirac_operator(U, xw, paramsw)

    substitute_fermion!(x, xcpu)


    apply_D(y, U1, U2, U3, U4, x, phitemp, temp)
    println(dot(y, y))
    mul!(yw, Dw, xw)
    println(dot(yw, yw))

    apply_Ddag(y, U1, U2, U3, U4, x, phitemp, temp)
    println(dot(y, y))
    mul!(yw, Dw', xw)
    println(dot(yw, yw))
    #return

    DdagD = DdagDgeneral(U, x, apply_D, apply_Ddag)
    mul!(χ, DdagD, η)
    dfdU = similar(U)
    dfdU1 = dfdU[1]
    dfdU2 = dfdU[2]
    dfdU3 = dfdU[3]
    dfdU4 = dfdU[4]

    set_wing_fermion!(x)
    dSFdU!(U, dfdU, apply_D, apply_Ddag, x)


    return

    func(U1, U2, U3, U4, χ, η, apply, phitemp, temp) = g(χ, U1, U2, U3, U4, η, params, apply, phitemp, temp)


    indices_mid = (3, 3, 3, 3)
    indices_halo = (2, 3, 3, 3)
    indices_mid_o = Tuple(collect(indices_mid) .+ nw)
    println(indices_mid_o)
    indices_halo_o = Tuple(collect(indices_halo) .+ nw)
    println(indices_halo_o)

    f(U1, U2, U3, U4) = func(U1, U2, U3, U4, χ, η, apply_wilson!, phitemp, temp)

    dSFdUn = Numerical_derivative_Enzyme(f, indices_mid, U1, U2, U3, U4)
    for μ = 1:4
        display(dSFdUn[μ])
    end

    dSFdUn_halo = Numerical_derivative_Enzyme(f, indices_halo, U1, U2, U3, U4)
    for μ = 1:4
        display(dSFdUn_halo[μ])
    end

    Enzyme_derivative!(
        func,
        U1,
        U2,
        U3,
        U4,
        dfdU1,
        dfdU2,
        dfdU3,
        dfdU4,
        nodiff(χ), nodiff(η), nodiff(apply_wilson!); temp=temp, dtemp=dtemp, phitemp=phitemp, dphitemp=dphitemp)



    println("halo ", indices_halo)
    for μ = 1:4
        println("mu = $μ ")
        display(dfdU[μ].U.A[:, :, indices_halo_o...])
        println("AD ")
        display(dSFdUn_halo[μ])
    end
    println("mid ", indices_mid)
    for μ = 1:4
        println("mu = $μ ")
        display(dfdU[μ].U.A[:, :, indices_mid_o...])
        println("AD ")
        display(dSFdUn[μ])
    end


end
main()