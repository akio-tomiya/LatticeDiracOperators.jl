using Gaugefields
using JACC
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using MPI
using Random
using Test
using Enzyme

JACC.@init_backend
MPI.Initialized() || MPI.Init()

const _LDO_DIRAC = LatticeDiracOperators.Dirac_operators

function _wilson_clover_wrapper_core(field)
    ranges = ntuple(
        direction -> (field.nw + 1):(field.nw + field.PN[direction]),
        length(field.PN),
    )
    return @view field.A[:, :, ranges...]
end

@testset "WilsonClover MPILattice wrapper" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4 * nprocs, 4, 4, 4)
    process_grid = (nprocs, 1, 1, 1)
    Random.seed!(800)
    gauge = Initialize_Gaugefields(
        3, 1, global_size...;
        condition="hot", isMPILattice=true, PEs=process_grid,
        verbose_level=0,
    )
    source = Initialize_pseudofermion_fields(gauge[1], "Wilson")
    rng = Random.MersenneTwister(801)
    source_values = randn(rng, ComplexF64, 3, 4, global_size...)
    source_lattice = LatticeMatrix(
        source_values, 4, process_grid;
        nw=1, phases=source.f.phases, comm0=source.f.comm,
    )
    substitute!(source.f, source_lattice)
    set_wing_fermion!(source)

    parameters = Dict(
        "Dirac_operator" => "WilsonClover",
        "κ" => 0.12,
        "cSW" => 1.17,
        "verbose_level" => 0,
    )
    operator = Dirac_operator(gauge, source, parameters)
    @test operator isa _LDO_DIRAC.Wilson_Dirac_operator_improved
    @test operator.D isa WilsonDiracCloverOperator4D
    @test _LDO_DIRAC.has_cloverterm(operator)
    @test operator.D.cSW == 1.17
    @test !haskey(parameters, "hasclover")

    direct = WilsonDiracCloverOperator4D(
        [link.U for link in gauge], 0.12, 1.17)
    actual = similar(source)
    expected = similar(source)

    mul!(actual, operator, source)
    mul!(expected.f, direct, source.f)
    @test isapprox(
        _wilson_clover_wrapper_core(actual.f),
        _wilson_clover_wrapper_core(expected.f);
        atol=3e-12, rtol=3e-12,
    )

    mul!(actual, operator', source)
    mul!(expected.f, direct', source.f)
    @test isapprox(
        _wilson_clover_wrapper_core(actual.f),
        _wilson_clover_wrapper_core(expected.f);
        atol=3e-12, rtol=3e-12,
    )

    replacement = Initialize_Gaugefields(
        3, 1, global_size...;
        condition="hot", isMPILattice=true, PEs=process_grid,
        verbose_level=0,
    )
    cache_epoch_before = halo_epochs(operator.D.clover[1]).core
    substitute_U!(gauge[1], replacement[1])
    mul!(actual, operator, source)
    cache_epoch_after = halo_epochs(operator.D.clover[1]).core
    @test cache_epoch_after > cache_epoch_before

    refreshed = WilsonDiracCloverOperator4D(
        [link.U for link in gauge], 0.12, 1.17)
    mul!(expected.f, refreshed, source.f)
    @test isapprox(
        _wilson_clover_wrapper_core(actual.f),
        _wilson_clover_wrapper_core(expected.f);
        atol=3e-12, rtol=3e-12,
    )

    rebuilt = operator(replacement)
    @test rebuilt.D isa WilsonDiracCloverOperator4D
    @test rebuilt.D !== operator.D
    @test rebuilt.D.cSW == operator.D.cSW

    normal_operator = DdagD_operator(gauge, source, parameters)
    @test normal_operator.dirac.D isa WilsonDiracCloverOperator4D
    intermediate = similar(source)
    mul!(actual, normal_operator, source)
    mul!(intermediate.f, refreshed, source.f)
    mul!(expected.f, refreshed', intermediate.f)
    @test isapprox(
        _wilson_clover_wrapper_core(actual.f),
        _wilson_clover_wrapper_core(expected.f);
        atol=5e-11, rtol=5e-11,
    )

    hopping_only_parameters = copy(parameters)
    hopping_only_parameters["Donly"] = true
    @test_throws ArgumentError Dirac_operator(
        gauge, source, hopping_only_parameters)

    default_parameters = Dict(
        "Dirac_operator" => "WilsonClover",
        "κ" => 0.12,
        "verbose_level" => 0,
    )
    @test Dirac_operator(gauge, source, default_parameters).D.cSW == 1.5612

    action = FermiAction(operator, Dict("Nf" => 2))
    force = [similar(link) for link in gauge]
    calc_UdSfdU!(force, action, gauge, source)
    @test all(link -> all(isfinite, link.U.A), force)
    @test all(link -> sum(abs2, link.U.A) > 0, force)

    # Check one Lie-algebra direction against the full pseudofermion action.
    direction = Matrix(im * Diagonal([1.0, -1.0, 0.0]))
    direction ./= sqrt(real(tr(direction' * direction)))
    site = ntuple(_ -> gauge[1].U.nw + 1, 4)
    epsilon = 1e-5
    perturbed = similar(gauge)
    original_link = copy(gauge[1].U.A[:, :, site...])

    substitute_U!(perturbed, gauge)
    perturbed[1].U.A[:, :, site...] .= exp(epsilon * direction) * original_link
    mark_halo_dirty!(perturbed[1].U)
    set_wing_U!(perturbed[1])
    action_plus = evaluate_FermiAction(action, perturbed, source)

    substitute_U!(perturbed, gauge)
    perturbed[1].U.A[:, :, site...] .= exp(-epsilon * direction) * original_link
    mark_halo_dirty!(perturbed[1].U)
    set_wing_U!(perturbed[1])
    action_minus = evaluate_FermiAction(action, perturbed, source)

    finite_difference = (action_plus - action_minus) / (2epsilon)
    force_matrix = force[1].U.A[:, :, site...]
    antihermitian = (force_matrix - force_matrix') / 2
    traceless_force = antihermitian - tr(antihermitian) * I / 3
    local_force_directional = -2 * real(tr(traceless_force * direction))
    force_directional = MPI.Allreduce(
        local_force_directional, +, MPI.COMM_WORLD)
    @test isapprox(
        force_directional, finite_difference; atol=2e-5, rtol=2e-4)
end
