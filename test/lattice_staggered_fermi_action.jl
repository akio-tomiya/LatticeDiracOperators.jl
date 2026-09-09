using Random

function _lattice_staggered_action_system(; discretization, nw)
    extent = nw >= 3 ? 4 : 2
    lattice = (extent, extent, extent, extent)
    U = gauge_configuration(
        lattice;
        colors=3,
        halo=nw,
        start=:hot,
        seed=UInt64(0x53544147 + nw),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    fermion = GeneralFermion(
        3, 1, lattice, (1, 1, 1, 1);
        nw,
        phases=(1, 1, 1, -1),
    )
    pseudofermion = similar(fermion)
    gaussian = similar(fermion)
    action = StaggeredFermiAction(
        U,
        fermion;
        mass=0.5,
        Nf=4,
        discretization,
        naik_epsilon=-0.083,
        eps_CG=1e-10,
        maxsteps=2_000,
        verbose_level=0,
    )
    return (; U, fermion, pseudofermion, gaussian, action)
end

function _all_finite_gaugefields(fields)
    return all(field -> all(isfinite, field.U.A), fields)
end

function _lattice_staggered_force_directional(force, U, direction)
    gradient = similar(U[1])
    value = 0.0
    for mu in 1:4
        mul!(gradient, force[mu]', U[mu])
        value += real(dot(gradient.U, direction[mu].U))
    end
    # LDO's stored link force is -1/2 of the ordinary Euclidean-gradient
    # representation; undo that convention for a finite-difference check.
    return -2value
end

@testset "LatticeMatrices StaggeredFermiAction" begin
    for (discretization, nw) in ((:staggered, 1), (:hisq, 3))
        system = _lattice_staggered_action_system(; discretization, nw)
        Random.seed!(0x4c4d5354 + nw)
        @test gauss_sampling_in_action!(
            system.gaussian, system.U, system.action) === system.gaussian
        @test sample_pseudofermions!(
            system.pseudofermion,
            system.U,
            system.action,
            system.gaussian,
        ) === system.pseudofermion

        potential = evaluate_FermiAction(
            system.action, system.U, system.pseudofermion)
        @test isfinite(potential)
        @test potential > 0

        force = similar(system.U)
        @test calc_UdSfdU!(
            force,
            system.action,
            system.U,
            system.pseudofermion,
        ) === force
        @test _all_finite_gaugefields(force)
        @test any(field -> !iszero(sum(abs2, field.U.A)), force)

        direction = gauge_configuration(
            ntuple(_ -> nw >= 3 ? 4 : 2, 4);
            colors=3,
            halo=nw,
            start=:hot,
            seed=UInt64(0x44495200 + nw),
            process_grid=(1, 1, 1, 1),
            verbose=0,
        )
        # A moderately wide step avoids amplifying GPU Krylov reduction
        # noise while retaining O(epsilon^2) central-difference accuracy.
        epsilon = 3e-4
        plus_U = copy_configuration(system.U)
        minus_U = copy_configuration(system.U)
        for mu in 1:4
            add_U!(plus_U[mu], epsilon, direction[mu])
            add_U!(minus_U[mu], -epsilon, direction[mu])
        end
        finite_difference = (
            evaluate_FermiAction(
                system.action, plus_U, system.pseudofermion) -
            evaluate_FermiAction(
                system.action, minus_U, system.pseudofermion)
        ) / (2epsilon)
        analytic = _lattice_staggered_force_directional(
            force, system.U, direction)
        @test isapprox(
            analytic, finite_difference;
            atol=2e-5,
            rtol=2e-4,
        )

        if discretization == :staggered
            smeared_action = NHYPSmearedFermiAction(
                system.action,
                system.pseudofermion,
            )
            workspace = md_action_workspace(smeared_action, system.U)
            @test isfinite(md_potential(
                smeared_action, system.U, workspace))
            momentum_force = gauge_momenta(system.U)
            @test md_force!(
                momentum_force,
                smeared_action,
                system.U,
                workspace,
            ) === nothing
            @test all(field -> all(isfinite, field.a.A), momentum_force)
        end
    end

    one_link = _lattice_staggered_action_system(
        discretization=:staggered, nw=1)
    @test_throws ArgumentError StaggeredFermiAction(
        one_link.U, one_link.fermion;
        mass=0.5, discretization=:hisq)
    @test_throws ArgumentError StaggeredFermiAction(
        one_link.U, one_link.fermion;
        mass=0.5, discretization=:unknown)
end
