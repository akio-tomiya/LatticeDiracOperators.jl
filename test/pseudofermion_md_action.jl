import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LatticeMatrices: gather_and_bcast_matrix
using LinearAlgebra
using Test

function _ldo_md_maximum_difference(left, right)
    return maximum(
        maximum(abs, left[direction] .- right[direction])
        for direction in eachindex(left)
    )
end

_ldo_md_global_links(U) = gather_and_bcast_matrix.(getproperty.(U, :U))
_ldo_md_global_momenta(p) = gather_and_bcast_matrix.(getproperty.(p, :a))

@testset "Stout MD action matches the legacy README path" begin
    # Keep the lattice, Wilson parameters, and single stout layer from the
    # legacy README example.  Two smearing objects are used because each owns
    # mutable work buffers: one for the provider and one for the old manual
    # calc_smearedU/back_prop sequence used as the regression oracle.
    U = gauge_configuration(
        (4, 4, 4, 4);
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(0x53544f55),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    provider_smearing = CovNeuralnet(U)
    push!(
        provider_smearing,
        STOUT_Layer(["plaquette"], [0.1], U),
    )
    legacy_smearing = CovNeuralnet(U)
    push!(
        legacy_smearing,
        STOUT_Layer(["plaquette"], [0.1], U),
    )

    field = Initialize_pseudofermion_fields(U[1], "Wilson")
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.141139,
        "eps_CG" => 1e-8,
        "MaxCGstep" => 2000,
        "verbose_level" => 0,
    )
    dirac = Dirac_operator(U, field, parameters)
    action = FermiAction(
        dirac,
        Dict{String,Any}();
        covneuralnet=provider_smearing,
    )
    pseudofermion = similar(field)
    noise = similar(field)
    provider = PseudofermionMDAction(action, pseudofermion)
    @test provider.smearing === provider_smearing

    legacy_links, _, _ = calc_smearedU(U, legacy_smearing)
    legacy_noise = similar(field)
    legacy_pseudofermion = similar(field)
    gauss_sampling_in_action!(
        legacy_noise,
        legacy_links,
        action;
        seed=0x52454144,
        sweep=7,
        subgroup=1,
    )
    sample_pseudofermions!(
        legacy_pseudofermion,
        legacy_links,
        action,
        legacy_noise,
    )
    refresh_pseudofermion!(
        provider,
        U,
        noise;
        seed=0x52454144,
        sweep=7,
        subgroup=1,
    )
    @test gather_and_bcast_matrix(noise.f) ==
          gather_and_bcast_matrix(legacy_noise.f)
    @test isapprox(
        gather_and_bcast_matrix(pseudofermion.f),
        gather_and_bcast_matrix(legacy_pseudofermion.f);
        rtol=2e-12,
        atol=2e-12,
    )

    workspace = md_action_workspace(provider, U)
    legacy_links, _, _ = calc_smearedU(U, legacy_smearing)
    legacy_potential = evaluate_FermiAction(
        action,
        legacy_links,
        pseudofermion,
    )
    @test isapprox(
        md_potential(provider, U, workspace),
        legacy_potential;
        rtol=2e-11,
        atol=2e-11,
    )

    provider_force = initialize_TA_Gaugefields(U)
    md_force!(provider_force, provider, U, workspace)

    legacy_links, link_history, _ = calc_smearedU(U, legacy_smearing)
    raw_derivative = calc_UdSfdU(action, legacy_links, pseudofermion)
    smeared_derivative = map(similar, U)
    for direction in eachindex(U)
        mul!(
            smeared_derivative[direction],
            legacy_links[direction]',
            raw_derivative[direction],
        )
    end
    thin_derivative = back_prop(
        smeared_derivative,
        legacy_smearing,
        link_history,
        U,
    )
    expected_force = initialize_TA_Gaugefields(U)
    conversion = similar(U[1])
    for direction in eachindex(U)
        clear_U!(expected_force[direction])
        mul!(conversion, U[direction], thin_derivative[direction])
        Traceless_antihermitian_add!(
            expected_force[direction],
            -1,
            conversion,
        )
    end
    @test _ldo_md_maximum_difference(
        _ldo_md_global_momenta(provider_force),
        _ldo_md_global_momenta(expected_force),
    ) < 2e-9
end

@testset "Pseudofermion MD action provider" begin
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=2,
        halo=1,
        start=:hot,
        seed=UInt64(0x10203040),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    field = Initialize_pseudofermion_fields(U[1], "Wilson")
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.08,
        "eps_CG" => 1e-11,
        "MaxCGstep" => 2000,
        "verbose_level" => 0,
    )
    dirac = Dirac_operator(U, field, parameters)
    action = FermiAction(dirac, Dict("Nf" => 2))
    pseudofermion = similar(field)
    noise = similar(field)
    provider = PseudofermionMDAction(action, pseudofermion)

    @test refresh_pseudofermion!(
        provider,
        U,
        noise;
        seed=0x314159,
        sweep=11,
        subgroup=1,
    ) === provider

    repeated_noise = similar(field)
    repeated_pseudofermion = similar(field)
    repeated_provider = PseudofermionMDAction(action, repeated_pseudofermion)
    refresh_pseudofermion!(
        repeated_provider,
        U,
        repeated_noise;
        seed=0x314159,
        sweep=11,
        subgroup=1,
    )
    @test gather_and_bcast_matrix(repeated_noise.f) ==
          gather_and_bcast_matrix(noise.f)
    @test gather_and_bcast_matrix(repeated_pseudofermion.f) ≈
          gather_and_bcast_matrix(pseudofermion.f)

    workspace = md_action_workspace(provider, U)
    @test md_potential(provider, U, workspace) ≈
          evaluate_FermiAction(action, U, pseudofermion)

    provider_force = initialize_TA_Gaugefields(U)
    md_force!(provider_force, provider, U, workspace)

    raw_derivative = calc_UdSfdU(action, U, pseudofermion)
    expected_force = initialize_TA_Gaugefields(U)
    for direction in eachindex(U)
        clear_U!(expected_force[direction])
        Traceless_antihermitian_add!(
            expected_force[direction],
            -1,
            raw_derivative[direction],
        )
    end
    @test _ldo_md_maximum_difference(
        _ldo_md_global_momenta(provider_force),
        _ldo_md_global_momenta(expected_force),
    ) < 2e-11

    gauge_action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(gauge_action, 0.5, plaquettes)
    actions = MDActionSet(; gauge=gauge_action, fermion=provider)
    integrator = SextonWeingarten(;
        slow=:fermion,
        fast=:gauge,
        n_fast=2,
    )
    momenta = gaussian_momenta(U; seed=UInt64(0x55667788))
    initial_links = _ldo_md_global_links(U)
    initial_momenta = _ldo_md_global_momenta(momenta)

    forward = md_driver(
        U,
        actions;
        steps=1,
        trajectory_length=0.002,
        integrator,
    )
    diagnostics = md_trajectory!(U, momenta, forward)
    @test isfinite(diagnostics.initial_hamiltonian)
    @test isfinite(diagnostics.final_hamiltonian)
    @test isfinite(diagnostics.delta_hamiltonian)

    backward = md_driver(
        U,
        actions;
        steps=1,
        trajectory_length=-0.002,
        integrator,
    )
    md_trajectory!(U, momenta, backward; diagnostics=false)
    @test _ldo_md_maximum_difference(
        _ldo_md_global_links(U),
        initial_links,
    ) < 2e-8
    @test _ldo_md_maximum_difference(
        _ldo_md_global_momenta(momenta),
        initial_momenta,
    ) < 2e-8
end
