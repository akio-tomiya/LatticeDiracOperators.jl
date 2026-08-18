using Enzyme
using Gaugefields
using JACC
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using MPI
using Random
using Test

JACC.@init_backend
MPI.Initialized() || MPI.Init()

struct _ApplyWilsonLM{Adjoint,T}
    kappa::T
end

_ApplyWilsonLM{Adjoint}(kappa::T) where {Adjoint,T} =
    _ApplyWilsonLM{Adjoint,T}(kappa)

function (apply::_ApplyWilsonLM{Adjoint})(
    result, U1, U2, U3, U4, source, fermion_temps, gauge_temps,
) where {Adjoint}
    operator = WilsonDiracOperator4D(
        U1.U, U2.U, U3.U, U4.U, apply.kappa)
    mul!(result.field, Adjoint ? operator' : operator, source.field)
    return result
end

@testset "LatticeMatrices Wilson callback AD" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    gauge = Initialize_Gaugefields(
        2, 1, global_size...;
        condition="cold", isMPILattice=true, PEs=process_grid,
        verbose_level=0,
    )

    Random.seed!(812)
    source = GeneralFermion(
        2, 4, global_size, process_grid; nw=1, numtemps=6)
    gauss_distribution_fermion!(source)
    set_wing_fermion!(source)

    action = GeneralFermionAction(
        gauge, source,
        _ApplyWilsonLM{false}(0.1), _ApplyWilsonLM{true}(0.1);
        numtemp=1, num=3, numg=3, numcg=6,
        eps_CG=1e-8, maxsteps=2_000, verbose_level=0,
    )
    force = similar(gauge)
    calc_UdSfdU!(force, action, gauge, source)

    @test Base.get_extension(
        LatticeMatrices, :LatticeMatricesEnzymeExt) !== nothing
    @test all(link -> all(isfinite, link.U.A), force)
    @test sum(link -> sum(abs2, link.U.A), force) > 0
end
