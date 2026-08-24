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
