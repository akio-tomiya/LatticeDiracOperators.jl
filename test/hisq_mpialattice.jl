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

const _HISQ_LDO_DIRAC = LatticeDiracOperators.Dirac_operators

function _hisq_wrapper_core(field)
    ranges = ntuple(
        direction -> (field.nw + 1):(field.nw + field.PN[direction]),
        length(field.PN),
    )
    return @view field.A[:, :, ranges...]
end

@testset "HISQ MPILattice wrapper" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4 * nprocs, 4, 4, 4)
    process_grid = (nprocs, 1, 1, 1)
    rng = Random.MersenneTwister(1901)

    gauge = gauge_configuration(
        global_size;
        colors=3,
        halo=3,
        start=:hot,
        seed=1900,
        process_grid,
    )
    source = Initialize_pseudofermion_fields(gauge[1], "staggered")
    source_values = randn(rng, ComplexF64, 3, 1, global_size...)
    source_lattice = LatticeMatrix(
        source_values,
        4,
        process_grid;
        nw=3,
        phases=source.f.phases,
        comm0=source.f.comm,
    )
    substitute!(source.f, source_lattice)
    set_wing_fermion!(source)

    parameters = Dict(
        "Dirac_operator" => "HISQ",
        "mass" => 0.17,
        "naik_epsilon" => -0.083,
        "eps_CG" => 1e-10,
        "MaxCGstep" => 2_000,
        "verbose_level" => 0,
    )
    operator = Dirac_operator(gauge, source, parameters)
    @test operator isa _HISQ_LDO_DIRAC.HISQ_Dirac_operator_MPILattice
    @test operator.cache isa HISQDiracCache4D
    @test operator.mass == 0.17
    @test operator.naik_epsilon == -0.083

    links = [link.U for link in gauge]
    direct_cache = HISQDiracCache4D(
        links, parameters["mass"];
        naik_epsilon=parameters["naik_epsilon"],
    )
    actual = similar(source)
    expected = similar(source)
    intermediate = similar(source)

    mul!(actual, operator, source)
    mul_cached_hisq!(
        expected.f,
        direct_cache,
        links[1],
        links[2],
        links[3],
        links[4],
        source.f,
    )
    @test isapprox(
        _hisq_wrapper_core(actual.f),
        _hisq_wrapper_core(expected.f);
        atol=3e-12,
        rtol=3e-12,
    )

    mul!(actual, operator', source)
    mul_cached_hisq_adjoint!(
        expected.f,
        direct_cache,
        links[1],
        links[2],
        links[3],
        links[4],
        source.f,
    )
    @test isapprox(
        _hisq_wrapper_core(actual.f),
        _hisq_wrapper_core(expected.f);
        atol=3e-12,
        rtol=3e-12,
    )

    normal_operator = DdagD_operator(gauge, source, parameters)
    @test normal_operator.dirac isa
        _HISQ_LDO_DIRAC.HISQ_Dirac_operator_MPILattice
    mul!(actual, normal_operator, source)
    mul!(intermediate, operator, source)
    mul!(expected, operator', intermediate)
    @test isapprox(
        _hisq_wrapper_core(actual.f),
        _hisq_wrapper_core(expected.f);
        atol=5e-11,
        rtol=5e-11,
    )

    replacement = gauge_configuration(
        global_size;
        colors=3,
        halo=3,
        start=:hot,
        seed=1902,
        process_grid,
    )
    rebuilt = operator(replacement)
    @test rebuilt.cache === operator.cache
    replacement_links = [link.U for link in replacement]
    replacement_cache = HISQDiracCache4D(
        replacement_links,
        operator.mass;
        naik_epsilon=operator.naik_epsilon,
    )
    mul!(actual, rebuilt, source)
    mul_cached_hisq!(
        expected.f,
        replacement_cache,
        replacement_links[1],
        replacement_links[2],
        replacement_links[3],
        replacement_links[4],
        source.f,
    )
    @test isapprox(
        _hisq_wrapper_core(actual.f),
        _hisq_wrapper_core(expected.f);
        atol=3e-12,
        rtol=3e-12,
    )

    mutated = gauge_configuration(
        global_size;
        colors=3,
        halo=3,
        start=:hot,
        seed=1903,
        process_grid,
    )
    substitute_U!(replacement, mutated)
    mutated_links = [link.U for link in replacement]
    mutated_cache = HISQDiracCache4D(
        mutated_links,
        operator.mass;
        naik_epsilon=operator.naik_epsilon,
    )
    mul!(actual, rebuilt, source)
    mul_cached_hisq!(
        expected.f,
        mutated_cache,
        mutated_links[1],
        mutated_links[2],
        mutated_links[3],
        mutated_links[4],
        source.f,
    )
    @test isapprox(
        _hisq_wrapper_core(actual.f),
        _hisq_wrapper_core(expected.f);
        atol=3e-12,
        rtol=3e-12,
    )

    bad_boundary = copy(parameters)
    bad_boundary["boundarycondition"] = [1, 1, 1, 1]
    @test_throws ArgumentError Dirac_operator(gauge, source, bad_boundary)

    short_halo_gauge = gauge_configuration(
        global_size;
        colors=3,
        halo=1,
        start=:cold,
        process_grid,
    )
    short_halo_source = Initialize_pseudofermion_fields(
        short_halo_gauge[1], "staggered")
    @test_throws ArgumentError Dirac_operator(
        short_halo_gauge, short_halo_source, parameters)

    wrong_color_gauge = gauge_configuration(
        global_size;
        colors=2,
        halo=3,
        start=:cold,
        process_grid,
    )
    wrong_color_source = Initialize_pseudofermion_fields(
        wrong_color_gauge[1], "staggered")
    @test_throws ArgumentError Dirac_operator(
        wrong_color_gauge, wrong_color_source, parameters)

    if nprocs == 1
        expect_no_enzyme = lowercase(get(
            ENV, "LDO_TEST_EXPECT_NO_ENZYME", "true")) in
            ("1", "true", "yes", "on")
        if expect_no_enzyme
            @test Base.get_extension(
                LatticeMatrices, :LatticeMatricesEnzymeExt) === nothing
            @test Base.get_extension(
                LatticeDiracOperators,
                :LatticeDiracOperatorsEnzymeExt,
            ) === nothing
        end
        action = FermiAction(operator, Dict("Nf" => 4))
        force = [similar(link) for link in gauge]
        clear_U!(force)
        Y = similar(source)
        _HISQ_LDO_DIRAC.calc_UdSfdU_fromX!(
            force, Y, action, gauge, source)
        @test all(link -> isfinite(real(dot(link.U, link.U))), force)
        @test any(link -> !iszero(real(dot(link.U, link.U))), force)

        function fixed_left_contraction(current_gauge)
            result = similar(source)
            mul!(result, operator(current_gauge), source)
            return real(dot(Y, result))
        end

        mu = 1
        row = 1
        col = 2
        site = (2, 2, 2, 2)
        halo_site = ntuple(i -> site[i] + gauge[mu].U.nw, 4)
        delta = 1e-6

        gauge_plus = similar(gauge)
        gauge_minus = similar(gauge)
        substitute_U!(gauge_plus, gauge)
        substitute_U!(gauge_minus, gauge)
        gauge_plus[mu].U.A[row, col, halo_site...] += delta
        gauge_minus[mu].U.A[row, col, halo_site...] -= delta
        mark_halo_dirty!(gauge_plus[mu].U)
        mark_halo_dirty!(gauge_minus[mu].U)
        derivative_real = (
            fixed_left_contraction(gauge_plus) -
            fixed_left_contraction(gauge_minus)
        ) / (2delta)

        substitute_U!(gauge_plus, gauge)
        substitute_U!(gauge_minus, gauge)
        gauge_plus[mu].U.A[row, col, halo_site...] += im * delta
        gauge_minus[mu].U.A[row, col, halo_site...] -= im * delta
        mark_halo_dirty!(gauge_plus[mu].U)
        mark_halo_dirty!(gauge_minus[mu].U)
        derivative_imaginary = (
            fixed_left_contraction(gauge_plus) -
            fixed_left_contraction(gauge_minus)
        ) / (2delta)

        thin_link = Matrix(@view gauge[mu].U.A[:, :, halo_site...])
        force_link = Matrix(@view force[mu].U.A[:, :, halo_site...])
        raw_gradient = force_link' * thin_link
        @test isapprox(
            derivative_real,
            real(raw_gradient[row, col]);
            atol=2e-5,
            rtol=2e-5,
        )
        @test isapprox(
            derivative_imaginary,
            imag(raw_gradient[row, col]);
            atol=2e-5,
            rtol=2e-5,
        )

        @test isfinite(evaluate_FermiAction(action, gauge, source))
        full_force = calc_UdSfdU(action, gauge, source)
        @test all(
            link -> isfinite(real(dot(link.U, link.U))), full_force)
        @test any(
            link -> !iszero(real(dot(link.U, link.U))), full_force)
    end
end
