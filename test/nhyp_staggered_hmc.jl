using LatticeMatrices: gather_and_bcast_matrix
using Random

function _nhyp_staggered_test_system(; zero_smearing=false)
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(0x4e485950),
        process_grid=(1, 1, 1, 1),
        comm=SerialCommunicator(),
        verbose=0,
    )
    gauge = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(gauge, 5.7 / 2, plaquettes)

    pseudofermion = Initialize_pseudofermion_fields(U[1], "staggered")
    gaussian = similar(pseudofermion)
    D = Dirac_operator(U, pseudofermion, Dict(
        "Dirac_operator" => "staggered",
        "mass" => 0.5,
        "eps" => 1e-13,
        "MaxCGstep" => 2_000,
        "verbose_level" => 0,
    ))
    fermion = FermiAction(D, Dict("Nf" => 4))
    coefficients = zero_smearing ? (0.0, 0.0, 0.0) : (0.5, 0.5, 0.4)
    fermion_md = NHYPSmearedFermiAction(
        fermion,
        pseudofermion;
        alpha_outer=coefficients[1],
        alpha_middle=coefficients[2],
        alpha_inner=coefficients[3],
    )
    return (; U, gauge, pseudofermion, gaussian, fermion, fermion_md)
end

function _nhyp_staggered_global_links(U)
    return gather_and_bcast_matrix.(getproperty.(U, :U))
end

function _nhyp_staggered_global_momenta(momentum)
    return gather_and_bcast_matrix.(getproperty.(momentum, :a))
end

function _nhyp_staggered_maximum_difference(left, right)
    return maximum(
        maximum(abs, left[direction] .- right[direction])
        for direction in eachindex(left)
    )
end

@testset "nHYP staggered MD provider" begin
    zero = _nhyp_staggered_test_system(; zero_smearing=true)
    zero_actions = MDActionSet(gauge=zero.gauge, fermion=zero.fermion_md)
    zero_driver = md_driver(
        zero.U,
        zero_actions;
        steps=1,
        trajectory_length=0.001,
        integrator=QPQ(),
    )
    Random.seed!(0x53544147)
    @test refresh_nhyp_pseudofermions!(
        zero.gaussian,
        zero.U,
        zero_driver,
        :fermion,
    ) === zero.pseudofermion

    nhyp_force = gauge_momenta(zero.U)
    md_force!(
        nhyp_force,
        zero.fermion_md,
        zero.U,
        zero_driver.action_workspace.terms.fermion,
    )
    direct_link_force = similar(zero.U)
    calc_UdSfdU!(
        direct_link_force,
        zero.fermion,
        zero.U,
        zero.pseudofermion,
    )
    direct_force = gauge_momenta(zero.U)
    for direction in 1:4
        Traceless_antihermitian_add!(
            direct_force[direction],
            -1,
            direct_link_force[direction],
        )
    end
    @test _nhyp_staggered_maximum_difference(
        _nhyp_staggered_global_momenta(nhyp_force),
        _nhyp_staggered_global_momenta(direct_force),
    ) < 2e-11

    system = _nhyp_staggered_test_system()
    actions = MDActionSet(gauge=system.gauge, fermion=system.fermion_md)
    forward = md_driver(
        system.U,
        actions;
        steps=2,
        trajectory_length=0.005,
        integrator=QPQ(),
    )
    Random.seed!(0x53544147)
    refresh_nhyp_pseudofermions!(
        system.gaussian,
        system.U,
        forward,
        :fermion,
    )
    initial_links = _nhyp_staggered_global_links(system.U)
    momentum = gaussian_momenta(system.U; seed=UInt64(0x484d43))
    initial_momenta = _nhyp_staggered_global_momenta(momentum)
    result = md_trajectory!(system.U, momentum, forward)
    @test isfinite(result.delta_hamiltonian)
    @test abs(result.delta_hamiltonian) < 1e-3
    @test _nhyp_staggered_maximum_difference(
        _nhyp_staggered_global_links(system.U),
        initial_links,
    ) > 1e-6

    backward = md_driver(
        system.U,
        actions;
        steps=2,
        trajectory_length=-0.005,
        integrator=QPQ(),
    )
    md_trajectory!(system.U, momentum, backward; diagnostics=false)
    @test _nhyp_staggered_maximum_difference(
        _nhyp_staggered_global_links(system.U),
        initial_links,
    ) < 2e-10
    @test _nhyp_staggered_maximum_difference(
        _nhyp_staggered_global_momenta(momentum),
        initial_momenta,
    ) < 2e-9
end

@testset "Generic analytically smeared staggered MD provider" begin
    system = _nhyp_staggered_test_system()
    for smearing in (APESmearing(), HYPSmearing())
        exception = try
            SmearedFermiAction(
                system.fermion, system.pseudofermion, smearing)
            nothing
        catch caught
            caught
        end
        @test exception isa ArgumentError
        @test occursin(
            "projection=:polar", sprint(showerror, exception))
    end

    for smearing in (
        StoutSmearing(rho=0.07), HEXSmearing(),
        APESmearing(alpha=0.5, projection=:polar),
        HYPSmearing(projection=:polar),
    )
        action = SmearedFermiAction(
            system.fermion,
            system.pseudofermion,
            smearing,
        )
        workspace = md_action_workspace(action, system.U)
        Random.seed!(0x53564d44)
        @test refresh_smeared_pseudofermions!(
            system.gaussian,
            action,
            system.U,
            workspace,
        ) === system.pseudofermion
        @test isfinite(md_potential(action, system.U, workspace))
        force = gauge_momenta(system.U)
        @test md_force!(force, action, system.U, workspace) === nothing
        @test all(field -> all(isfinite, field.a.A), force)
    end

    action = SmearedFermiAction(
        system.fermion,
        system.pseudofermion,
        StoutSmearing(rho=0.07),
    )
    actions = MDActionSet(gauge=system.gauge, fermion=action)
    driver = md_driver(system.U, actions; steps=1)
    Random.seed!(0x53565246)
    @test refresh_smeared_pseudofermions!(
        system.gaussian,
        system.U,
        driver,
        :fermion,
    ) === system.pseudofermion
end
