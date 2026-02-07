struct GeneralFermionAction{Dim,Dirac,fermion,gauge} <: FermiAction{Dim,Dirac,fermion,gauge}
    DdagD::Dirac
    numtemp::Int64
    function GeneralFermionAction(U::Vector{TG}, x::TF, apply_D::TmulD, apply_Ddag::TmulDdag;
        numcg=4, num=10, numg=10, eps_CG=1e-12, maxsteps=10000, verbose_level=2, numtemp=5) where {
        TmulD,TmulDdag,TG<:Fields_4D_MPILattice,TF<:GeneralFermion}

        Dim = length(U)
        fermion = typeof(x)
        gauge = eltype(U)

        DdagD = DdagDgeneral(U, x, apply_D, apply_Ddag;
            numcg, num, numg, eps_CG, maxsteps, verbose_level, numtemp)
        Dirac = typeof(DdagD)

        return new{Dim,Dirac,fermion,gauge}(DdagD, numtemp)
    end
end
export GeneralFermionAction

function evaluate_FermiAction(
    fermi_action::GeneralFermionAction,
    U,
    ϕ::GeneralFermion,
)
    #u1 = fermi_action.DdagD.U[1].U.A[1, 1, 3, 3, 3, 3]
    #println("d ", fermi_action.DdagD.U[1].U.A[1, 1, 3, 3, 3, 3])
    DdagD = fermi_action.DdagD(U)
    #u2 = DdagD.U[1].U.A[1, 1, 3, 3, 3, 3]

    #println("dd ", DdagD.U[1].U.A[1, 1, 3, 3, 3, 3])
    #println("diff ", u1 - u2)

    η, it_η = get_block(DdagD._temporary_fermion_forCG)
    solve_DinvX!(η, DdagD, ϕ)
    Sf = dot(ϕ, η)

    unused!(DdagD._temporary_fermion_forCG, it_η)
    return real(Sf)
end

function gauss_sampling_in_action!(
    η::GeneralFermion,
    U,
    fermi_action::GeneralFermionAction,
)
    #gauss_distribution_fermion!(η)
    #gauss_distribution_fermion!(η, rand)
    gauss_distribution_fermion!(η)
end


function sample_pseudofermions!(
    ϕ::GeneralFermion,
    U,
    fermi_action::GeneralFermionAction,
    ξ::GeneralFermion,
)
    #W = fermi_action.diracoperator(U)
    DdagD = fermi_action.DdagD(U)
    apply_Ddag = fermi_action.DdagD.apply_Ddag
    set_wing_fermion!(ξ)

    numtemp = fermi_action.numtemp
    temp, ittemp = get_block(DdagD._temporary_gaugefield, numtemp)
    phitemp, itphitemp = get_block(DdagD._temporary_fermion, numtemp)

    apply_Ddag(ϕ, U[1], U[2], U[3], U[4], ξ, phitemp, temp)

    unused!(DdagD._temporary_fermion, itphitemp)
    unused!(DdagD._temporary_gaugefield, ittemp)

    #println("ξ")

    #mul!(ϕ, W', ξ)

    set_wing_fermion!(ϕ)
    #println("ϕ")
end

function vdD(χ, U1, U2, U3, U4, η, apply, phitemp, temp)
    phitemp1 = phitemp[end]
    apply(phitemp1, U1, U2, U3, U4, η, phitemp, temp)
    #Dmul!(phitemp1, U1, U2, U3, U4, D, η)
    s = -2 * real(dot(χ, phitemp1)) * (-0.5)
    return s
end

function calc_UdSfdU!(
    UdSfdU::Vector{TU},
    fermi_action::TG,
    U::Vector{TU},
    ϕ::TA,
) where {TU<:AbstractGaugefields,TG<:GeneralFermionAction,TA<:GeneralFermion}
    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]

    dfdU1 = UdSfdU[1]
    dfdU2 = UdSfdU[2]
    dfdU3 = UdSfdU[3]
    dfdU4 = UdSfdU[4]
    clear_U!(UdSfdU)
    numtemp = fermi_action.numtemp


    apply_D = fermi_action.DdagD.apply_D
    #apply_Ddag = fermi_action.DdagD.apply_Ddag
    DdagD = fermi_action.DdagD(U)
    #update_U!(DdagD, U)

    η, it_η = get_block(DdagD._temporary_fermion_forCG)
    solve_DinvX!(η, DdagD, ϕ)

    set_wing_fermion!(η)
    χ, it_χ = get_block(DdagD._temporary_fermion)

    temp, ittemp = get_block(DdagD._temporary_gaugefield, numtemp)
    phitemp, itphitemp = get_block(DdagD._temporary_fermion, numtemp)
    dtemp, itdtemp = get_block(DdagD._temporary_gaugefield, numtemp)
    dphitemp, itdphitemp = get_block(DdagD._temporary_fermion_forCG, numtemp)

    apply_D(χ, U1, U2, U3, U4, η, phitemp, temp)
    #mul!(χ, D, η)

    func(U1, U2, U3, U4, χ, η, apply, phitemp, temp) = vdD(χ, U1, U2, U3, U4, η, apply, phitemp, temp)

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
        nodiff(χ), nodiff(η), nodiff(apply_D); temp=temp, dtemp=dtemp, phitemp=phitemp, dphitemp=dphitemp)

    #indices = (2, 3, 2, 4)
    #dfdu = dfdU1.U.A[:, :, indices...]
    #display(dfdu)
    #=
    dfdu = dfdU1.U.A[:, :, 3, 3, 3, 3]
    u1 = U[1].U.A[:, :, 3, 3, 3, 3]
    println("u1 * dfdu")
    display(u1 * dfdu)
    println("u1 * dfdu'")
    display(u1 * dfdu')
    println("u1' * dfdu")
    display(u1' * dfdu)
    println("u1' * dfdu'")
    display(u1' * dfdu')
    println("u1 * dfdu^T")
    display(u1 * transpose(dfdu))
    println("u1' * dfdu^T")
    display(u1' * transpose(dfdu))
    println("dfdu")
    display(dfdu)
    =#

    #=
    for μ = 1:4
    println("μ = $μ")
    display(UdSfdU[μ].U.A[:, :, 3, 3, 3, 3])
    end
    =#

    #=
    f(U1, U2, U3, U4) = func(U1, U2, U3, U4, χ, η, apply_D, phitemp, temp)

    nw = 1
    indices_mid = (3, 3, 3, 3)
    indices_halo = (2, 3, 3, 3)
    indices_mid_o = Tuple(collect(indices_mid) .+ nw)
    println(indices_mid_o)
    indices_halo_o = Tuple(collect(indices_halo) .+ nw)
    println(indices_halo_o)


    dSFdUn = Numerical_derivative_Enzyme(f, indices_mid, U1, U2, U3, U4)
    for μ = 1:4
        display(dSFdUn[μ])
    end
    dSFdUn_halo = Numerical_derivative_Enzyme(f, indices_halo, U1, U2, U3, U4)
    for μ = 1:4
        display(dSFdUn_halo[μ])
    end
    =#

    #=
    println("halo ", indices_halo)
    for μ = 1:4
        println("mu = $μ ")
        display(UdSfdU[μ].U.A[:, :, indices_halo_o...])
        println("AD ")
        display(dSFdUn_halo[μ])
    end
    println("mid ", indices_mid)
    for μ = 1:4
        println("mu = $μ ")
        display(UdSfdU[μ].U.A[:, :, indices_mid_o...])
        println("AD ")
        display(dSFdUn[μ])
    end
    =#

    #error("numerical!!!")


    unused!(DdagD._temporary_fermion_forCG, it_η)
    unused!(DdagD._temporary_fermion, it_χ)
    unused!(DdagD._temporary_fermion, itphitemp)
    unused!(DdagD._temporary_gaugefield, ittemp)
    unused!(DdagD._temporary_fermion_forCG, itdphitemp)
    unused!(DdagD._temporary_gaugefield, itdtemp)


    temp, ittemp = get_block(DdagD._temporary_gaugefield)
    for μ = 1:4

        mul!(temp, U[μ], UdSfdU[μ]')
        substitute_U!(UdSfdU[μ], temp)
    end
    unused!(DdagD._temporary_gaugefield, ittemp)
    #dSFdU!(U, dfdU::Vector{TG}, apply_D, apply_Ddag, φ, numtemp, verbose_level)

end

