using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using Random
using Test
import JACC

JACC.@init_backend
include(joinpath(@__DIR__, "test_communicator.jl"))

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

function _general_fermion_core(field)
    ranges = ntuple(
        direction -> (field.nw + 1):(field.nw + field.PN[direction]),
        length(field.PN),
    )
    return @view field.A[:, :, ranges...]
end

struct _ApplyConstructedLMOperator{Adjoint,B}
    builder::B
end

function (apply::_ApplyConstructedLMOperator{Adjoint})(
    result, U1, U2, U3, U4, source, fermion_temps, gauge_temps,
) where {Adjoint}
    operator = apply.builder([U1.U, U2.U, U3.U, U4.U])
    applied = Adjoint ? adjoint(operator) : operator
    mul!(result.field, applied, source.field)
    return result
end

struct _ApplyCachedClover{Adjoint,C}
    cache::C
end

function (apply::_ApplyCachedClover{false})(
    result, U1, U2, U3, U4, source, fermion_temps, gauge_temps,
)
    mul_cached_clover!(
        result.field, apply.cache,
        U1.U, U2.U, U3.U, U4.U, source.field)
    return result
end


function (apply::_ApplyCachedClover{true})(
    result, U1, U2, U3, U4, source, fermion_temps, gauge_temps,
)
    mul_cached_clover_adjoint!(
        result.field, apply.cache,
        U1.U, U2.U, U3.U, U4.U, source.field)
    return result
end


struct _ApplyCachedHISQ{Adjoint,C}
    cache::C
end

function (apply::_ApplyCachedHISQ{false})(
    result, U1, U2, U3, U4, source, fermion_temps, gauge_temps,
)
    mul_cached_hisq!(
        result.field, apply.cache,
        U1.U, U2.U, U3.U, U4.U, source.field)
    return result
end


function (apply::_ApplyCachedHISQ{true})(
    result, U1, U2, U3, U4, source, fermion_temps, gauge_temps,
)
    mul_cached_hisq_adjoint!(
        result.field, apply.cache,
        U1.U, U2.U, U3.U, U4.U, source.field)
    return result
end


function _test_registered_general_fermion_action(
    gauge, operator, source, apply_D, apply_Ddag,
)
    action = GeneralFermionAction(
        gauge, source, apply_D, apply_Ddag;
        numtemp=1, num=3, numg=2, numcg=4,
        eps_CG=1e-10, verbose_level=0,
    )
    registered_result = similar(source)
    mul!(registered_result, action.DdagD, source)

    intermediate = similar(source)
    expected = similar(source)
    mul!(intermediate, operator, source)
    mul!(expected, adjoint(operator), intermediate)
    @test isapprox(
        _general_fermion_core(registered_result.field),
        _general_fermion_core(expected.field);
        atol=5e-10,
        rtol=5e-10,
    )
    return action
end

@testset "GeneralFermion with LatticeMatrices backend" begin
    nprocs = ldo_test_comm_size()
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
        builder = current_links -> WilsonDiracOperator4D(current_links, 0.12)
        _test_registered_general_fermion_action(
            gauge,
            operator,
            wilson_source,
            _ApplyConstructedLMOperator{false,typeof(builder)}(builder),
            _ApplyConstructedLMOperator{true,typeof(builder)}(builder),
        )
    end

    @testset "Wilson clover" begin
        operator = WilsonDiracCloverOperator4D(links, 0.12, 1.0)
        _test_general_fermion_operator(operator, wilson_source, wilson_left)
        _test_registered_general_fermion_action(
            gauge,
            operator,
            wilson_source,
            _ApplyCachedClover{false,typeof(operator)}(operator),
            _ApplyCachedClover{true,typeof(operator)}(operator),
        )
    end

    staggered_source = _general_fermion_test_field(1, gsize, PEs; nw=1, seed=201)
    staggered_left = _general_fermion_test_field(1, gsize, PEs; nw=1, seed=202)
    @testset "staggered" begin
        operator = StaggeredDiracOperator4D(links, 0.01)
        _test_general_fermion_operator(operator, staggered_source, staggered_left)
        builder = current_links -> StaggeredDiracOperator4D(current_links, 0.01)
        _test_registered_general_fermion_action(
            gauge,
            operator,
            staggered_source,
            _ApplyConstructedLMOperator{false,typeof(builder)}(builder),
            _ApplyConstructedLMOperator{true,typeof(builder)}(builder),
        )
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
        cache = HISQDiracCache4D(
            links_hisq, 0.01; naik_epsilon=-0.083)
        _test_registered_general_fermion_action(
            gauge_hisq,
            operator,
            hisq_source,
            _ApplyCachedHISQ{false,typeof(cache)}(cache),
            _ApplyCachedHISQ{true,typeof(cache)}(cache),
        )
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
        builder = current_links -> D5DW_MobiusDomainwallOperator5D(
            current_links, L5, 0.01, -1.0, 1.0, 1.0)
        _test_registered_general_fermion_action(
            gauge,
            operator,
            domainwall_source,
            _ApplyConstructedLMOperator{false,typeof(builder)}(builder),
            _ApplyConstructedLMOperator{true,typeof(builder)}(builder),
        )
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
        builder = current_links -> D5DW_GeneralizedDomainwallOperator5D(
            current_links, L5, 0.01, -1.0, a5, b5, c5)
        _test_registered_general_fermion_action(
            gauge,
            operator,
            domainwall_source,
            _ApplyConstructedLMOperator{false,typeof(builder)}(builder),
            _ApplyConstructedLMOperator{true,typeof(builder)}(builder),
        )
    end
end
