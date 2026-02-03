import Gaugefields.AbstractGaugefields_module: Gaugefields_4D_MPILattice, Fields_4D_MPILattice
using PreallocatedArrays

import LatticeDiracOperators.Dirac_operators: vdD



function LatticeDiracOperators.dSFdU!(U::Vector{TG}, dfdU::Vector{TG}, apply_D, apply_Ddag, φ::L1; numtemp=5, verbose_level=2) where {TG<:Fields_4D_MPILattice,L1<:GeneralFermion}
    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]

    dfdU1 = dfdU[1]
    dfdU2 = dfdU[2]
    dfdU3 = dfdU[3]
    dfdU4 = dfdU[4]


    DdagD = DdagDgeneral(U, φ, apply_D, apply_Ddag; verbose_level)
    η, it_η = get_block(DdagD._temporary_fermion_forCG)

    solve_DinvX!(η, DdagD, φ)
    #println("solved")
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


    unused!(DdagD._temporary_fermion_forCG, it_η)
    unused!(DdagD._temporary_fermion, it_χ)
    unused!(DdagD._temporary_fermion, itphitemp)
    unused!(DdagD._temporary_gaugefield, ittemp)
    unused!(DdagD._temporary_fermion_forCG, itdphitemp)
    unused!(DdagD._temporary_gaugefield, itdtemp)
end