using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using MPI
using Random
using Test
import JACC

JACC.@init_backend
MPI.Initialized() || MPI.Init()

function _general_fermion_test_field(NG, gsize, PEs; nw, seed)
    rng = Random.MersenneTwister(seed)
    NC = 3
    dim = length(gsize)
    phases = ntuple(direction -> direction == 4 ? -1 : 1, dim)
    values = randn(rng, ComplexF64, NC, NG, gsize...)
    return GeneralFermion(LatticeMatrix(values, dim, PEs; nw, phases))
end

function _test_general_fermion_operator(operator, source, left)
    result = similar(source)
    adjoint_result = similar(left)

    @test mul!(result, operator, source) === result
    @test mul!(adjoint_result, adjoint(operator), left) === adjoint_result
    @test isapprox(
        dot(left.field, result.field),
        dot(adjoint_result.field, source.field);
        atol=5e-10,
        rtol=5e-10,
    )
    @test isfinite(real(dot(result.field, result.field)))
end

@testset "GeneralFermion with LatticeMatrices backend" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    gsize = (4 * nprocs, 4, 4, 4)
    PEs = (nprocs, 1, 1, 1)
    NC = 3

    gauge = Initialize_Gaugefields(
        NC,
        1,
        gsize...;
        condition="cold",
        isMPILattice=true,
        PEs,
        verbose_level=0,
    )
    links = [link.U for link in gauge]
    @test all(link -> link isa LatticeMatrix, links)

    wilson_source = _general_fermion_test_field(4, gsize, PEs; nw=1, seed=101)
    wilson_left = _general_fermion_test_field(4, gsize, PEs; nw=1, seed=102)
    @test wilson_source.field isa LatticeMatrix

    @testset "Wilson" begin
        operator = WilsonDiracOperator4D(links, 0.12)
        _test_general_fermion_operator(operator, wilson_source, wilson_left)
    end

    @testset "Wilson clover" begin
        operator = WilsonDiracCloverOperator4D(links, 0.12, 1.0)
        _test_general_fermion_operator(operator, wilson_source, wilson_left)
    end

    staggered_source = _general_fermion_test_field(1, gsize, PEs; nw=1, seed=201)
    staggered_left = _general_fermion_test_field(1, gsize, PEs; nw=1, seed=202)
    @testset "staggered" begin
        operator = StaggeredDiracOperator4D(links, 0.01)
        _test_general_fermion_operator(operator, staggered_source, staggered_left)
    end

    @testset "HISQ" begin
        gauge_hisq = Initialize_Gaugefields(
            NC,
            3,
            gsize...;
            condition="cold",
            isMPILattice=true,
            PEs,
            verbose_level=0,
        )
        links_hisq = [link.U for link in gauge_hisq]
        hisq_source = _general_fermion_test_field(1, gsize, PEs; nw=3, seed=301)
        hisq_left = _general_fermion_test_field(1, gsize, PEs; nw=3, seed=302)
        operator = HISQDiracOperator4D(
            links_hisq,
            0.01;
            naik_epsilon=-0.083,
        )
        _test_general_fermion_operator(operator, hisq_source, hisq_left)
    end

    L5 = 2
    gsize5 = (gsize..., L5)
    PEs5 = (PEs..., 1)
    domainwall_source = _general_fermion_test_field(4, gsize5, PEs5; nw=1, seed=401)
    domainwall_left = _general_fermion_test_field(4, gsize5, PEs5; nw=1, seed=402)

    @testset "domain wall" begin
        operator = D5DW_MobiusDomainwallOperator5D(
            links,
            L5,
            0.01,
            -1.0,
            1.0,
            1.0,
        )
        _test_general_fermion_operator(operator, domainwall_source, domainwall_left)
    end

    @testset "generalized domain wall" begin
        a5 = [0.9, 1.1]
        b5 = [1.4, 1.6]
        c5 = [0.4, 0.6]
        operator = D5DW_GeneralizedDomainwallOperator5D(
            links,
            L5,
            0.01,
            -1.0,
            a5,
            b5,
            c5,
        )
        _test_general_fermion_operator(operator, domainwall_source, domainwall_left)
    end
end
