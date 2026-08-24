import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LatticeMatrices: gather_and_bcast_matrix
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "Pseudofermion MD action and Z4 on two ranks" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    U = gauge_configuration(
        global_size;
        colors=2,
        halo=1,
        start=:hot,
        seed=UInt64(0x10203040),
        process_grid,
        verbose=0,
    )

    distributed_z4 = Initialize_pseudofermion_fields(U[1], "Wilson")
    Z4_distribution_fermi!(distributed_z4; seed=0x1234, sweep=9)
    distributed_values = gather_and_bcast_matrix(distributed_z4.f)
    roots = (1.0 + 0.0im, 0.0 + 1.0im, -1.0 + 0.0im, 0.0 - 1.0im)
    @test all(value -> value in roots, distributed_values)

    reference_U = gauge_configuration(
        global_size;
        colors=2,
        halo=1,
        start=:cold,
        process_grid=(1, 1, 1, 1),
        comm=MPI.COMM_SELF,
        verbose=0,
    )
    reference_z4 = Initialize_pseudofermion_fields(reference_U[1], "Wilson")
    Z4_distribution_fermi!(reference_z4; seed=0x1234, sweep=9)
    @test distributed_values == gather_and_bcast_matrix(reference_z4.f)

    field = Initialize_pseudofermion_fields(U[1], "Wilson")
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.08,
        "eps_CG" => 1e-10,
        "MaxCGstep" => 2000,
        "verbose_level" => 0,
    )
    dirac = Dirac_operator(U, field, parameters)
    action = FermiAction(dirac, Dict("Nf" => 2))
    provider = PseudofermionMDAction(action, similar(field))
    noise = similar(field)
    refresh_pseudofermion!(
        provider,
        U,
        noise;
        seed=0x55667788,
        sweep=4,
        subgroup=2,
    )

    workspace = md_action_workspace(provider, U)
    @test isfinite(md_potential(provider, U, workspace))
    force = initialize_TA_Gaugefields(U)
    md_force!(force, provider, U, workspace)
    @test all(
        all(isfinite, gather_and_bcast_matrix(momentum.a))
        for momentum in force
    )

    gauge_action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(gauge_action, 0.5, plaquettes)
    actions = MDActionSet(; gauge=gauge_action, fermion=provider)
    driver = md_driver(
        U,
        actions;
        steps=1,
        trajectory_length=0.001,
        integrator=SextonWeingarten(
            slow=:fermion,
            fast=:gauge,
            n_fast=2,
        ),
    )
    momenta = gaussian_momenta(U; seed=UInt64(0xabcdef01))
    diagnostics = md_trajectory!(U, momenta, driver)
    @test isfinite(diagnostics.delta_hamiltonian)
end
