using Gaugefields
using JACC
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using Random
using Test

JACC.@init_backend
include(joinpath(@__DIR__, "test_communicator.jl"))

const _STAGGERED_LDO_DIRAC = LatticeDiracOperators.Dirac_operators

function _staggered_wrapper_core(field)
    ranges = ntuple(
        direction -> (field.nw + 1):(field.nw + field.PN[direction]),
        length(field.PN),
    )
    return @view field.A[:, :, ranges...]
end

@testset "Staggered MPILattice wrapper" begin
    nprocs = ldo_test_comm_size()
    global_size = (4 * nprocs, 4, 4, 4)
    process_grid = (nprocs, 1, 1, 1)
    rng = Random.MersenneTwister(901)
    source_values = randn(rng, ComplexF64, 3, 1, global_size...)

    action_gauge = nothing
    action_source = nothing
    action_operator = nothing

    for nw in (0, 1)
        gauge = gauge_configuration(
            global_size;
            colors=3, halo=nw, start=:hot, process_grid=process_grid,
        )
        @test gauge_backend(gauge) isa LatticeMatricesBackend
        source = Initialize_pseudofermion_fields(gauge[1], "staggered")
        @test source isa _STAGGERED_LDO_DIRAC.StaggeredFermion_4D_MPILattice
        @test source.f isa LatticeMatrix
        @test source.f.NC2 == 1
        @test source.f.nw == nw

        source_lattice = LatticeMatrix(
            source_values, 4, process_grid;
            nw, phases=source.f.phases, comm0=source.f.comm,
        )
        substitute!(source.f, source_lattice)
        set_wing_fermion!(source)

        parameters = Dict(
            "Dirac_operator" => "staggered",
            "mass" => 0.17,
            "eps_CG" => 1e-10,
            "MaxCGstep" => 1_000,
            "verbose_level" => 0,
        )
        operator = Dirac_operator(gauge, source, parameters)
        @test operator isa _STAGGERED_LDO_DIRAC.Staggered_Dirac_operator_MPILattice
        @test operator.D isa StaggeredDiracOperator4D
        @test operator.mass == 0.17

        direct = StaggeredDiracOperator4D([link.U for link in gauge], 0.17)
        actual = similar(source)
        expected = similar(source)
        intermediate = similar(source)

        mul!(actual, operator, source)
        mul!(expected.f, direct, source.f)
        @test isapprox(
            _staggered_wrapper_core(actual.f),
            _staggered_wrapper_core(expected.f);
            atol=3e-12, rtol=3e-12,
        )

        mul!(actual, operator', source)
        mul!(expected.f, direct', source.f)
        @test isapprox(
            _staggered_wrapper_core(actual.f),
            _staggered_wrapper_core(expected.f);
            atol=3e-12, rtol=3e-12,
        )

        normal_operator = DdagD_operator(gauge, source, parameters)
        @test normal_operator.dirac.D isa StaggeredDiracOperator4D
        mul!(actual, normal_operator, source)
        mul!(intermediate.f, direct, source.f)
        mul!(expected.f, direct', intermediate.f)
        @test isapprox(
            _staggered_wrapper_core(actual.f),
            _staggered_wrapper_core(expected.f);
            atol=5e-11, rtol=5e-11,
        )

        replacement = gauge_configuration(
            global_size;
            colors=3, halo=nw, start=:hot, process_grid=process_grid,
        )
        rebuilt = operator(replacement)
        @test rebuilt.D isa StaggeredDiracOperator4D
        @test rebuilt.D !== operator.D
        @test rebuilt.mass == operator.mass
        rebuilt_direct = StaggeredDiracOperator4D(
            [link.U for link in replacement], 0.17)
        mul!(actual, rebuilt, source)
        mul!(expected.f, rebuilt_direct, source.f)
        @test isapprox(
            _staggered_wrapper_core(actual.f),
            _staggered_wrapper_core(expected.f);
            atol=3e-12, rtol=3e-12,
        )

        bad_parameters = copy(parameters)
        bad_parameters["boundarycondition"] = [1, 1, 1, 1]
        @test_throws ArgumentError Dirac_operator(
            gauge, source, bad_parameters)

        if nw == 1
            action_gauge = gauge
            action_source = source
            action_operator = operator
        end
    end

    action = FermiAction(action_operator, Dict("Nf" => 4))
    @test isfinite(evaluate_FermiAction(action, action_gauge, action_source))
    force = calc_UdSfdU(action, action_gauge, action_source)
    @test length(force) == 4
    @test all(link -> isfinite(real(dot(link.U, link.U))), force)

    if nprocs == 1
        legacy_source = _STAGGERED_LDO_DIRAC.Initialize_StaggeredFermion(
            3, global_size...; nowing=true)
        for it in 1:global_size[4], iz in 1:global_size[3],
            iy in 1:global_size[2], ix in 1:global_size[1], ic in 1:3
            legacy_source[ic, ix, iy, iz, it, 1] =
                source_values[ic, 1, ix, iy, iz, it]
        end

        parameters = Dict(
            "Dirac_operator" => "staggered",
            "mass" => 0.17,
            "eps_CG" => 1e-10,
            "MaxCGstep" => 1_000,
            "verbose_level" => 0,
        )
        legacy_operator = Dirac_operator(
            action_gauge, legacy_source, parameters)
        @test legacy_operator isa _STAGGERED_LDO_DIRAC.Staggered_Dirac_operator

        lm_result = similar(action_source)
        legacy_result = similar(legacy_source)
        mul!(lm_result, action_operator, action_source)
        mul!(legacy_result, legacy_operator, legacy_source)
        legacy_values = permutedims(legacy_result.f, (1, 6, 2, 3, 4, 5))
        @test isapprox(
            gather_matrix(lm_result.f), legacy_values;
            atol=3e-12, rtol=3e-12,
        )

        legacy_action = FermiAction(legacy_operator, Dict("Nf" => 4))
        lm_kernel_force = [similar(link) for link in action_gauge]
        legacy_kernel_force = [similar(link) for link in action_gauge]
        clear_U!(lm_kernel_force)
        clear_U!(legacy_kernel_force)
        lm_y = similar(action_source)
        legacy_y = similar(legacy_source)
        _STAGGERED_LDO_DIRAC.calc_UdSfdU_fromX!(
            lm_kernel_force, lm_y, action, action_gauge, action_source)
        _STAGGERED_LDO_DIRAC.calc_UdSfdU_fromX!(
            legacy_kernel_force, legacy_y, legacy_action,
            action_gauge, legacy_source)
        for mu in 1:4
            @test isapprox(
                gather_matrix(lm_kernel_force[mu].U),
                gather_matrix(legacy_kernel_force[mu].U);
                atol=3e-12, rtol=3e-12,
            )
        end
    end
end
