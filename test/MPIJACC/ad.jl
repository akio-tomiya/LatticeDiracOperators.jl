using Enzyme
using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "test_communicator.jl"))

include(joinpath(@__DIR__, "..", "..", "examples", "GeneralFermion_Shift_AD.jl"))
using .GeneralFermionShiftADExample

module GeneralFermionQuickstartExample
include(joinpath(@__DIR__, "..", "..", "examples", "GeneralFermion_Quickstart.jl"))
end

module GeneralFermionAllDirectionsExample
include(joinpath(@__DIR__, "..", "..", "examples", "GeneralFermion_AllDirections.jl"))
end

function core_values(field)
    ranges = ntuple(
        direction -> (field.nw + 1):(field.nw + field.PN[direction]),
        length(field.PN),
    )
    return @view field.A[:, :, ranges...]
end

@testset "GeneralFermion apply_D! quickstart" begin
    process_grid = (ldo_test_comm_size(), 1, 1, 1)
    result = GeneralFermionQuickstartExample.run_quickstart(;
        process_grid, comm=LDO_TEST_COMM)
    @test isfinite(real(dot(result.DdagD_source, result.DdagD_source)))
    @test isfinite(result.force_norm)
    @test result.force_norm > 0

    # The quickstart D depends only on U1.
    @test all(iszero, result.force[2].U.A)
    @test all(iszero, result.force[3].U.A)
    @test all(iszero, result.force[4].U.A)
end

@testset "GeneralFermion apply_D! in all directions" begin
    process_grid = (ldo_test_comm_size(), 1, 1, 1)
    result = GeneralFermionAllDirectionsExample.run_all_directions(;
        process_grid, comm=LDO_TEST_COMM)
    @test isfinite(real(dot(result.DdagD_source, result.DdagD_source)))
    @test all(isfinite, result.force_norms)
    @test all(>(0), result.force_norms)
end

@testset "GeneralFermionAction MPI Enzyme force" begin
    kappa = 0.1
    process_grid = (ldo_test_comm_size(), 1, 1, 1)

    removed_parameters = Dict("Dirac_operator" => "GeneralDirac")
    result = run_shift_defined_ad(;
        kappa, process_grid, comm=LDO_TEST_COMM)
    gauge = result.gauge
    source = result.source

    @test_throws ErrorException Dirac_operator(gauge, source, removed_parameters)
    @test_throws ErrorException DdagD_operator(gauge, source, removed_parameters)

    operator = WilsonDiracOperator4D([link.U for link in gauge], kappa)
    intermediate = similar(source)
    expected_result = similar(source)
    mul!(intermediate.field, operator, source.field)
    mul!(expected_result.field, adjoint(operator), intermediate.field)
    @test isapprox(
        core_values(result.ddagd_source.field),
        core_values(expected_result.field);
        atol=5e-10,
        rtol=5e-10,
    )

    @test Base.get_extension(
        LatticeDiracOperators,
        :LatticeDiracOperatorsEnzymeExt,
    ) !== nothing
    @test isfinite(result.potential)
    @test isfinite(result.force_norm)
    @test result.force_norm > 0
end
