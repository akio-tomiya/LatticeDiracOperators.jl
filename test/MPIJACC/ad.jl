using Random
using Gaugefields
using LinearAlgebra
using PreallocatedArrays
using Enzyme
using LatticeDiracOperators
import JACC
JACC.@init_backend

# Relax Enzyme type analysis before any autodiff to avoid EnzymeNoTypeError.
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypedepth!(20)
Enzyme.API.maxtypeoffset!(2048)

function shiftedadd(y, Uμ, x, γμ, shift_p, shift_m, phi1, phi2, κ)
    #U_n[ν](1 - γν) * ψ_{n + ν}
    mul_AshiftB!(phi1, Uμ, x, shift_p)
    #println("phi1 ", dot(phi1, phi1))
    mul!(phi2, phi1, transpose(I(4) - γμ))
    #println("phi2 ", dot(phi2, phi2))
    add_fermion!(y, -κ, phi2)

    # U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
    mul_shiftAshiftB!(phi1, Uμ', x, shift_m, shift_m)
    mul!(phi2, phi1, transpose(I(4) + γμ))
    add_fermion!(y, -κ, phi2)
end


#D.apply(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)
#ψ_n - κ sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
function apply_wilson!(y, U1, U2, U3, U4, x, params, phitemps, temp)
    U = (U1, U2, U3, U4)

    clear_fermion!(y)
    #println("x ", dot(x, x))
    #println(tr(U1))
    add_fermion!(y, 1, x)
    γs = (γ1, γ2, γ3, γ4)
    κ = params.κ

    phi1 = phitemps[3]
    phi2 = phitemps[4]
    dim = 4
    #println("y 1 ", dot(y, y))
    for μ = 1:dim
        #println("$μ y ", dot(y, y))

        shift_p = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        shift_m = ntuple(i -> ifelse(i == μ, -1, 0), dim)
        #clear_fermion!(y)
        shiftedadd(y, U[μ], x, γs[μ], shift_p, shift_m, phi1, phi2, κ)
        #println("μ 1 ", dot(y, y) / (-κ)^2)

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

function loss_wilson(U1, U2, U3, U4, phi, phitemp, params, temp)
    y = phitemp[3]
    x = phi
    #params = nothing
    apply_wilson!(y, U1, U2, U3, U4, x, params, phitemp, temp)
    return real(dot(phi, y))
end

function g(χ, U1, U2, U3, U4, η, p, apply, phitemp, temp)
    phitemp1 = phitemp[end]
    apply(phitemp1, U1, U2, U3, U4, η, p, phitemp, temp)
    #Dmul!(phitemp1, U1, U2, U3, U4, D, η)
    #s = -2 * real(dot(_lm_primal(χ), _lm_primal(phitemp1)))
    #s = -2 * real(dot(χ.f, phitemp1.f))
    s = -2 * real(dot(χ, phitemp1))


    return s
end



function main()
    β = 6
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    Dim = 4
    Nwing = 1

    Random.seed!(123)

    #U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="hot";
        isMPILattice=true)
    Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT,
        condition="hot")
    substitute_U!(U, Ucpu)

    x = Initialize_pseudofermion_fields(U[1], "Wilson")
    xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson")
    x2 = Initialize_pseudofermion_fields(U[1], "Wilson")
    phi = Initialize_pseudofermion_fields(U[1], "Wilson")

    phitemp = typeof(x)[]
    temp = typeof(U[1])[]
    # Allocate enough temporaries to avoid dynamic growth warnings.
    for i = 1:16
        push!(phitemp, similar(x))
        push!(temp, similar(U[1]))
    end
    κ = 0.141139
    params = (κ=κ,)

    gauss_distribution_fermion!(phi)
    #s = loss_wilson(U[1], U[2], U[3], U[4], phi, phitemp, params, temp)
    set_wing_U!(U)


    params = Dict()
    params["Dirac_operator"] = "GeneralDirac"
    params["apply_D"] = apply_wilson!
    params["apply_Ddag"] = apply_wilson_dag!
    params["eps_CG"] = 1.0e-18
    params["parameters"] = (κ=κ,)
    D = Dirac_operator(U, x, params)


    paramsw = Dict()
    paramsw["Dirac_operator"] = "Wilson"
    paramsw["κ"] = 0.141139
    paramsw["eps_CG"] = 1.0e-18
    paramsw["faster version"] = true
    Dw = Dirac_operator(U, x, paramsw)

    gauss_distribution_fermion!(xcpu)
    set_wing_fermion!(x)
    substitute_fermion!(x, xcpu)
    substitute_fermion!(x2, xcpu)
    set_wing_fermion!(x2)

    mul!(phitemp[1], D, x)
    println(dot(phitemp[1], phitemp[1]))




    mul!(phitemp[2], Dw, x2)

    set_wing_fermion!(phitemp[1])
    set_wing_fermion!(phitemp[2])


    println(dot(phitemp[2], phitemp[2]))


    println("norm phitemp[1] = ", dot(phitemp[1], phitemp[1]))
    println("norm phitemp[2] = ", dot(phitemp[2], phitemp[2]))


    DdagD = DdagD_operator(U, x, params)

    mul!(phitemp[1], DdagD, x)
    println(dot(x, x))
    set_wing_fermion!(x)
    solve_DinvX!(phitemp[2], DdagD, x)
    set_wing_fermion!(phitemp[2])
    mul!(phitemp[1], DdagD, phitemp[2])
    println(dot(phitemp[1], phitemp[1]))


    dfdU = similar(U)

    # g の中身をそのまま実行して、どこで落ちるか見る

    η = phitemp[1]
    GD = Dirac_operator(U, x, params)
    D = GD.diracop
    DdagD = DdagD_operator(U, x, params)
    solve_DinvX!(η, DdagD, x)
    #solve!(η, DdagD, φ) #η = (DdagD)^-1 φ
    println("solved")
    set_wing_fermion!(η)
    χ = phitemp[2]
    mul!(χ, D, η)

    try
        g(χ, U[1], U[2], U[3], U[4], η, D.p, D.apply, phitemp, temp)
        println("g(primal) ok")
    catch err
        @show err
        rethrow()
    end



    Enzyme.Compiler.VERBOSE_ERRORS[] = true
    try
        dSFdU!(dfdU, GD, x)
    catch err
        ct = code_typed(err)
        open("enzyme_code_typed.txt", "w") do io
            show(io, ct)
        end

        rethrow()
    end

    for μ = 1:4
        display(dfdU[μ].f.A[:, :, 2, 2, 2, 2])
    end


    #dSFdU!(dfdU, D, x)



    return

    # phitemp[1], phitemp[2] の差分を調べる
    a = phitemp[1]
    b = phitemp[2]

    PN = (a.NX, a.NY, a.NZ, a.NT)
    NG = a.NG
    NC = a.NC

    maxdiff = 0.0
    maxidx = nothing
    count = 0
    tol = 1e-12

    @inbounds for α = 1:NG
        for it = 1:PN[4], iz = 1:PN[3], iy = 1:PN[2], ix = 1:PN[1]
            for ic = 1:NC
                va = a.f.A[ic, α, ix+Nwing, iy+Nwing, iz+Nwing, it+Nwing]
                vb = b.f.A[ic, α, ix+Nwing, iy+Nwing, iz+Nwing, it+Nwing]
                d = va - vb
                ad = abs(d)
                if ad > maxdiff
                    maxdiff = ad
                    maxidx = (ic, ix, iy, iz, it, α)
                end
                if ad > tol
                    count += 1
                    if count <= 10
                        println("diff[(ic,ix,iy,iz,it,α)=$((ic,ix,iy,iz,it,α))] = ",
                            d, "  a=", va, " b=", vb)
                    end
                end
            end
        end
    end

    println("physical maxdiff = ", maxdiff, " at ", maxidx, "  count>tol=", count)


    return


    println(s)
end
main()
