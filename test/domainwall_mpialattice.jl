using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using MPI
using Random
using Test

MPI.Initialized() || MPI.Init()

function _domainwall_mpialattice_core(field)
    ranges = ntuple(
        direction -> (field.nw + 1):(field.nw + field.PN[direction]),
        length(field.PN))
    return @view field.A[:, :, ranges...]
end

function _domainwall_mpialattice_finite(field)
    return all(isfinite, _domainwall_mpialattice_core(field))
end

@testset "domain-wall MPILattice standard backend" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    gsize = (2 * nprocs, 2, 2, 2)
    PEs = (nprocs, 1, 1, 1)
    L5 = 2
    U = Initialize_Gaugefields(
        3, 1, gsize...;
        condition="cold",
        isMPILattice=true,
        PEs,
        verbose_level=0,
    )

    shamir_field = Initialize_pseudofermion_fields(
        U[1], "Domainwall"; L5)
    mobius_field = Initialize_pseudofermion_fields(
        U[1], "MobiusDomainwall"; L5)
    generalized_field = Initialize_pseudofermion_fields(
        U[1], "GeneralizedDomainwall"; L5)

    @test shamir_field isa DomainwallFermion_5D_MPILattice
    @test mobius_field isa MobiusDomainwallFermion_5D_MPILattice
    @test generalized_field isa DomainwallFermion_5D_MPILattice
    @test shamir_field.f isa LatticeMatrix{5}
    @test shamir_field.Dirac_operator == "Domainwall"
    @test mobius_field.Dirac_operator == "MobiusDomainwall"
    @test generalized_field.Dirac_operator == "GeneralizedDomainwall"
    @test get_PEs(shamir_field.f) == (PEs..., 1)
    @test shift_fermion(shamir_field, 5).f.shift == (0, 0, 0, 0, 1)

    Random.seed!(502)
    gauss_distribution_fermion!(shamir_field)
    @test _domainwall_mpialattice_finite(shamir_field.f)
    substitute_fermion!(mobius_field, shamir_field)
    substitute_fermion!(generalized_field, shamir_field)

    common_parameters = Dict{String,Any}(
        "mass" => 0.1,
        "L5" => L5,
        "M" => -1.0,
        "eps_CG" => 1e-8,
        "MaxCGstep" => 500,
        "verbose_level" => 0,
    )
    shamir_parameters = copy(common_parameters)
    shamir_parameters["Dirac_operator"] = "Domainwall"
    mobius_parameters = copy(common_parameters)
    mobius_parameters["Dirac_operator"] = "MobiusDomainwall"
    mobius_parameters["b"] = 1.0
    mobius_parameters["c"] = 1.0

    shamir = Dirac_operator(U, shamir_field, shamir_parameters)
    mobius = Dirac_operator(U, mobius_field, mobius_parameters)
    @test shamir.D5DW.D isa D5DW_MobiusDomainwallOperator5D
    @test mobius.D5DW.D isa D5DW_MobiusDomainwallOperator5D
    mismatched_parameters = copy(shamir_parameters)
    mismatched_parameters["L5"] = L5 + 1
    @test_throws DimensionMismatch Dirac_operator(
        U, shamir_field, mismatched_parameters)
    twisted_boundary = ComplexF64[1, 1, 1, cis(0.2), 1]
    twisted_field = DomainwallFermion_5D_MPILattice(
        U[1], L5;
        operator_name="Domainwall",
        boundarycondition=twisted_boundary,
    )
    twisted_parameters = copy(shamir_parameters)
    twisted_parameters["boundarycondition"] = twisted_boundary
    twisted = Dirac_operator(U, twisted_field, twisted_parameters)
    @test twisted.boundarycondition ≈ twisted_boundary

    shamir_result = similar(shamir_field)
    mobius_result = similar(mobius_field)
    mul!(shamir_result, shamir.D5DW, shamir_field)
    mul!(mobius_result, mobius.D5DW, mobius_field)
    @test _domainwall_mpialattice_core(shamir_result.f) ≈
        _domainwall_mpialattice_core(mobius_result.f) atol=2e-12 rtol=2e-12

    left = similar(shamir_field)
    Random.seed!(503)
    gauss_distribution_fermion!(left)
    adjoint_result = similar(left)
    mul!(adjoint_result, shamir.D5DW', left)
    @test dot(left, shamir_result) ≈
        dot(adjoint_result, shamir_field) atol=2e-9 rtol=2e-10

    DdagD = DdagD_operator(U, shamir_field, shamir_parameters)
    ddagd_result = similar(shamir_field)
    mul!(ddagd_result, DdagD, shamir_field)
    @test _domainwall_mpialattice_finite(ddagd_result.f)

    action = FermiAction(shamir, Dict())
    @test isfinite(evaluate_FermiAction(action, U, shamir_field))
    force = calc_UdSfdU(action, U, shamir_field)
    @test length(force) == 4
    @test all(link -> all(isfinite, link.U.A), force)

    generalized_parameters = copy(common_parameters)
    generalized_parameters["Dirac_operator"] = "GeneralizedDomainwall"
    generalized_parameters["as"] = [0.9, 1.1]
    generalized_parameters["bs"] = [1.4, 1.6]
    generalized_parameters["cs"] = [0.4, 0.6]
    generalized = Dirac_operator(
        U, generalized_field, generalized_parameters)
    @test generalized.D5DW.D isa D5DW_GeneralizedDomainwallOperator5D
    generalized_result = similar(generalized_field)
    generalized_adjoint_result = similar(left)
    mul!(generalized_result, generalized.D5DW, generalized_field)
    mul!(generalized_adjoint_result, generalized.D5DW', left)
    @test isapprox(
        dot(left, generalized_result),
        dot(generalized_adjoint_result, generalized_field);
        atol=3e-9,
        rtol=3e-10,
    )
    generalized_action = FermiAction(generalized, Dict())
    @test isfinite(evaluate_FermiAction(
        generalized_action, U, generalized_field))
    generalized_force = calc_UdSfdU(
        generalized_action, U, generalized_field)
    @test length(generalized_force) == 4
    @test all(link -> all(isfinite, link.U.A), generalized_force)

    # LM defines the generalized coefficients so that a=1,
    # b_s=(b+c)/2, c_s=(b-c)/2 reproduces the Möbius operator.  Check the
    # same identity through the LDO action and analytic force wrappers.
    scaled_mobius_parameters = copy(common_parameters)
    scaled_mobius_parameters["Dirac_operator"] = "MobiusDomainwall"
    scaled_mobius_parameters["b"] = 2.0
    scaled_mobius_parameters["c"] = 1.0
    compatible_parameters = copy(common_parameters)
    compatible_parameters["Dirac_operator"] = "GeneralizedDomainwall"
    compatible_parameters["as"] = ones(L5)
    compatible_parameters["bs"] = fill(1.5, L5)
    compatible_parameters["cs"] = fill(0.5, L5)

    scaled_mobius = Dirac_operator(
        U, mobius_field, scaled_mobius_parameters)
    compatible_generalized = Dirac_operator(
        U, generalized_field, compatible_parameters)
    mul!(mobius_result, scaled_mobius.D5DW, mobius_field)
    mul!(generalized_result, compatible_generalized.D5DW, generalized_field)
    @test _domainwall_mpialattice_core(mobius_result.f) ≈
        _domainwall_mpialattice_core(generalized_result.f) atol=2e-12 rtol=2e-12

    scaled_action = FermiAction(scaled_mobius, Dict())
    compatible_action = FermiAction(compatible_generalized, Dict())
    @test evaluate_FermiAction(scaled_action, U, mobius_field) ≈
        evaluate_FermiAction(compatible_action, U, generalized_field) atol=2e-9 rtol=2e-10
    scaled_force = calc_UdSfdU(scaled_action, U, mobius_field)
    compatible_force = calc_UdSfdU(
        compatible_action, U, generalized_field)
    @test all(1:4) do mu
        scaled_force[mu].U.A ≈ compatible_force[mu].U.A
    end
end
