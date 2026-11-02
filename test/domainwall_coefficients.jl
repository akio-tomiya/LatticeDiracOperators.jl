module DomainwallCoefficientTests
using Test, LinearAlgebra, JACC, Gaugefields, LatticeMatrices, LatticeDiracOperators
JACC.@init_backend
include(joinpath(@__DIR__, "test_communicator.jl"))
const comm = LDO_TEST_COMM
const ranks = ldo_test_comm_size()
select_device_by_mpi_rank!(comm)
if lowercase(get(ENV, "LDO_TEST_REQUIRE_GPU", "false")) == "true"
    @test JACC.backend == "cuda"
end

function relative_difference(x, y)
    tmp = similar(x)
    substitute_fermion!(tmp, x)
    axpby!(-1, y, 1, tmp)
    return sqrt(real(dot(tmp, tmp)) / max(real(dot(y, y)), eps(Float64)))
end

@testset "Generalized coefficient renewal, backend=$(JACC.backend), ranks=$ranks" begin
    U = gauge_configuration((2ranks, 2, 2, 4); colors=3, halo=1, start=:hot,
        seed=UInt64(27), process_grid=(ranks, 1, 1, 1), comm, verbose=0)
    x = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5=4)
    gauss_distribution_fermion!(x; seed=71)
    result, reference, original, solution, rhs = (similar(x) for _ in 1:5)
    for eo in (false, true)
        parameters = Dict{String,Any}(
            "Dirac_operator" => "GeneralizedDomainwall", "L5" => 4,
            "mass" => 0.15, "M" => -1.0, "as" => [0.9, 1.0, 1.1, 1.2],
            "bs" => [1.7, 1.6, 1.5, 1.4], "cs" => [0.6, 0.5, 0.4, 0.3],
            "eps_CG" => 1e-22, "MaxCGstep" => 3000, "verbose_level" => 0,
            "method_CG" => "bicg", "evenodd" => eo)
        W = Dirac_operator(U, x, parameters)
        action = FermiAction(W, Dict())
        mul!(original, W.D5DW, x)
        a, b, c = [1.2, 0.85, 1.05, 0.95], [1.6, 1.3, 1.8, 1.2], [0.3, 0.2, 0.55, 0.4]
        W2 = Renew(W, b, c; as=a)
        params2 = merge(parameters, Dict("as" => copy(a), "bs" => copy(b), "cs" => copy(c)))
        fresh = Dirac_operator(U, x, params2)
        # Caller-owned coefficient mutation must not invalidate the new caches.
        fill!(a, 9); fill!(b, 9); fill!(c, 9)
        @test W2.D5DW.as == params2["as"]
        @test W2.bs == params2["bs"]
        @test W2.cs == params2["cs"]
        @test W.bs == parameters["bs"]
        @test W2.D5DW_PV.mass == 1
        for (old, renewed, rebuilt) in ((W.D5DW, W2.D5DW, fresh.D5DW),
                                      (W.D5DW_PV, W2.D5DW_PV, fresh.D5DW_PV))
            @test renewed._temporary_fermi === old._temporary_fermi
            @test renewed._temporary_fermion_forCG === old._temporary_fermion_forCG
            @test renewed.use_eo == eo
            if eo
                @test renewed.eo_cache !== old.eo_cache
                @test Array(renewed.eo_cache.lower) ≈ Array(rebuilt.eo_cache.lower)
                @test Array(renewed.eo_cache.upper) ≈ Array(rebuilt.eo_cache.upper)
            end
            for adj in (false, true)
                op, refop = adj ? (renewed', rebuilt') : (renewed, rebuilt)
                mul!(result, op, x); mul!(reference, refop, x)
                @test relative_difference(result, reference) < 1e-13
                clear_fermion!(solution)
                solve_DinvX!(solution, op, x)
                mul!(rhs, op, solution)
                @test relative_difference(rhs, x) < 1e-10
            end
            @test !any(old._temporary_fermi._data._flagusing)
            @test !any(old._temporary_fermion_forCG._data._flagusing)
        end
        # Reusing an action must agree with a completely new action and force.
        renewed_action = Renew(action, W2)
        fresh_action = FermiAction(fresh, Dict())
        @test renewed_action._temporary_fermionfields === action._temporary_fermionfields
        @test renewed_action._temporary_gaugefields === action._temporary_gaugefields
        @test evaluate_FermiAction(renewed_action, U, x) ≈
              evaluate_FermiAction(fresh_action, U, x) rtol=2e-11
        f1, f2 = [similar(u) for u in U], [similar(u) for u in U]
        calc_UdSfdU!(f1, renewed_action, U, x)
        calc_UdSfdU!(f2, fresh_action, U, x)
        for mu in 1:4
            @test Array(f1[mu].U.A) ≈ Array(f2[mu].U.A) rtol=2e-11 atol=2e-11
        end
        mul!(result, W.D5DW, x)
        @test relative_difference(result, original) < 1e-14
        default_a = Renew(W, params2["bs"], params2["cs"])
        @test default_a.D5DW.as == parameters["as"]
        @test_throws DimensionMismatch Renew(W, ones(3), ones(4))
        @test_throws ArgumentError Renew(W, [NaN, 1., 1., 1.], ones(4))
        @test_throws ArgumentError Renew(action, fresh)
        if eo
            # A singular diagonal must also return its borrowed cache template.
            @test_throws SingularException Renew(W.D5DW, fill(-1/3, 4), fill(1/3, 4))
            @test !any(W.D5DW._temporary_fermi._data._flagusing)
        end
    end
end
end
