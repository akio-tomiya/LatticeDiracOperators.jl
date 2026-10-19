# Generalized domain-wall even/odd preconditioning tests.
# Run with an environment containing the EO implementation:
#   julia --startup-file=no --project=<environment> test/domainwall_eo.jl
# See test/eo/README.md in the EO checkout for environment setup and MPI/GPU use.
# LDO_TEST_MPI=true enables MPI; LDO_TEST_REQUIRE_GPU=true rejects CPU fallback.
# Optional: LDO_EO_BENCH_N=4, LDO_EO_BENCH_L5=4, LDO_EO_BENCH_REPEATS=5.
# Speedups are informational; solution agreement and residuals are assertions.

import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using Test

if !isdefined(@__MODULE__, :LDO_TEST_COMM)
    const LDO_TEST_MPI_ENABLED =
        lowercase(get(ENV, "LDO_TEST_MPI", "false")) == "true"

    if LDO_TEST_MPI_ENABLED
        @eval using MPI
        MPI.Initialized() || MPI.Init()
        const LDO_TEST_COMM = MPI.COMM_WORLD
        ldo_test_comm_size() = MPI.Comm_size(LDO_TEST_COMM)
        ldo_test_comm_rank() = MPI.Comm_rank(LDO_TEST_COMM)
        ldo_test_allreduce_sum(value) =
            MPI.Allreduce(value, MPI.SUM, LDO_TEST_COMM)
        ldo_test_allreduce_max(value) =
            MPI.Allreduce(value, MPI.MAX, LDO_TEST_COMM)
        ldo_test_barrier() = MPI.Barrier(LDO_TEST_COMM)
    else
        const LDO_TEST_COMM = LatticeMatrices.SerialCommunicator()
        ldo_test_comm_size() = 1
        ldo_test_comm_rank() = 0
        ldo_test_allreduce_sum(value) = value
        ldo_test_allreduce_max(value) = value
        ldo_test_barrier() = nothing
    end
end

const EO_REQUIRE_GPU = lowercase(get(ENV, "LDO_TEST_REQUIRE_GPU", "false")) == "true"
EO_REQUIRE_GPU && JACC.backend == "threads" && error(
    "GPU test requested, but JACC selected threads; configure JACC in the active project first")
const EO_EXPECTED_RANKS = parse(Int, get(ENV, "LDO_TEST_EXPECT_RANKS", string(ldo_test_comm_size())))
ldo_test_comm_size() == EO_EXPECTED_RANKS || error(
    "Expected $EO_EXPECTED_RANKS ranks, got $(ldo_test_comm_size()); check LDO_TEST_MPI and the launcher")

# Device selection belongs to the backend layer and uses node-local rank.
# It runs before allocation and also supports scheduler-isolated devices.
const EO_DEVICE_SELECTION = select_device_by_mpi_rank!(LDO_TEST_COMM)
@info "EO backend test" backend=JACC.backend rank=ldo_test_comm_rank() ranks=ldo_test_comm_size() selection=EO_DEVICE_SELECTION


const EO_Dop = LatticeDiracOperators.Dirac_operators

# The small dense oracle uses an explicit serial communicator. Distributed
# runs below already compare gathered fields against an undistributed oracle.
if ldo_test_comm_size() == 1
    include(joinpath(@__DIR__, "eo", "dense_reference.jl"))
end

function _eo_parameters(L5=4)
    Dict{String,Any}(
        "Dirac_operator" => "GeneralizedDomainwall", "L5" => L5,
        "mass" => 0.15, "M" => -1.0,
        "as" => collect(range(0.9, 1.1; length=L5)),
        "bs" => collect(range(1.7, 0.9; length=L5)),
        "cs" => collect(range(0.6, 0.1; length=L5)),
        "eps_CG" => 1e-20, "MaxCGstep" => 3000, "verbose_level" => 0,
        "method_CG" => "bicg", "evenodd" => true)
end

function _eo_relative_difference(x, y)
    tmp = similar(x)
    substitute_fermion!(tmp, x)
    axpby!(-1, y, 1, tmp)
    sqrt(real(dot(tmp, tmp)) / max(real(dot(y, y)), eps(Float64)))
end

_eo_pool_free(pool) = !any(pool._data._flagusing)

@testset "Generalized domain-wall EO blocks and solves" begin
    nprocs = ldo_test_comm_size()
    U = gauge_configuration((2nprocs, 2, 2, 2); colors=3, halo=1,
        start=:hot, seed=UInt64(47), process_grid=(nprocs, 1, 1, 1),
        comm=LDO_TEST_COMM, verbose=0)
    x = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5=4)
    y, image, adj_image = similar(x), similar(x), similar(x)
    gauss_distribution_fermion!(x; seed=42)
    gauss_distribution_fermion!(y; seed=43)
    parameters = _eo_parameters()
    A = Dirac_operator(U, x, parameters).D5DW
    @test A.use_eo
    @test A(U).use_eo
    @test A(U).eo_cache === A.eo_cache
    @test adjoint(adjoint(A)) === A

    # Independent on-site check against the full unpreconditioned operator, including
    # nonuniform a_s, b_s and c_s, both chiralities and both parities.
    for even in (true, false), adj in (false, true)
        rhs = similar(x)
        substitute_fermion!(rhs, x)
        EO_Dop.GeneralizedD5DW_keep_parity!(rhs, even)
        if adj
            EO_Dop.D5DW_diag_solve_adjoint_direct!(image, A, rhs, even)
        else
            EO_Dop.D5DW_diag_solve_block!(image, A, rhs, even)
        end
        mul!(adj_image, adj ? A' : A, image)
        EO_Dop.GeneralizedD5DW_keep_parity!(adj_image, even)
        @test _eo_relative_difference(adj_image, rhs) < 2e-13
    end
    S = EO_Dop.D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)
    @test adjoint(adjoint(S)) === S
    # Also check projection of arbitrary inputs, not only pre-masked vectors.
    mul!(image, S, x)
    mul!(adj_image, S', y)
    @test dot(y, image) ≈ dot(adj_image, x) rtol=2e-12 atol=2e-12
    substitute_fermion!(adj_image, image)
    EO_Dop.GeneralizedD5DW_keep_parity!(adj_image, false)
    @test real(dot(adj_image, adj_image)) < 1e-25

    @testset "EO Krylov halo deferral" begin
        Q = EO_Dop.DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice(S)
        Qadj = EO_Dop.DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice(S')
        composed = similar(x)
        for op in (S, S', Q, Qadj)
            # Internal vector algebra changes only core data. Public field
            # operations provide an independent reference with a current halo.
            substitute_fermion!(adj_image, x)
            EO_Dop._solver_copy!(op, image, x)
            @test halo_is_dirty(image.f)
            @test _eo_relative_difference(image, adj_image) < 2e-13
            substitute_fermion!(image, x)
            axpby!(0.3 + 0.2im, y, -0.7, adj_image)
            EO_Dop._solver_axpby!(op, 0.3 + 0.2im, y, -0.7, image)
            @test halo_is_dirty(image.f)
            @test _eo_relative_difference(image, adj_image) < 2e-13

            mul!(adj_image, op, x)
            @test !halo_is_dirty(adj_image.f)
            EO_Dop._solver_mul!(image, op, x)
            @test halo_is_dirty(image.f)
            @test _eo_relative_difference(image, adj_image) < 2e-12
            if op isa EO_Dop.DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice
                # Q(S) is S†S, while Q(S†) is SS†; check the order using
                # two public matvecs independent of the normal-operator path.
                mul!(composed, op.dirac, x)
                mul!(adj_image, op.dirac', composed)
                @test _eo_relative_difference(image, adj_image) < 2e-12
            end
            @test _eo_pool_free(A._temporary_fermion_forCG)
        end

        # Test the direct Krylov return contract before full-system
        # reconstruction can refresh the solution halo itself.
        substitute_fermion!(y, x)
        EO_Dop.GeneralizedD5DW_keep_parity!(y, true)
        rhs = similar(x)
        for (method, op) in ((:bicg, S), (:bicgstab, S'), (:cg, Q))
            mul!(rhs, op, y)
            clear_fermion!(adj_image)
            diagnostics = getfield(EO_Dop, method)(adj_image, op, rhs;
                eps=A.eps_CG, maxsteps=A.MaxCGstep, verbose=A.verbose_print)
            @test diagnostics.method == method
            @test diagnostics.iterations > 0
            @test diagnostics.recursive_residual_squared < A.eps_CG
            @test !halo_is_dirty(adj_image.f)
            @test _eo_relative_difference(adj_image, y) < 1e-8
            mul!(image, op, adj_image)
            @test _eo_relative_difference(image, rhs) < 1e-8
            @test _eo_pool_free(A._temporary_fermion_forCG)
        end
    end

    # Non-EO reference and unchanged non-EO solver selection.
    for method in ("bicg", "bicgstab")
        p = merge(parameters, Dict("evenodd" => false, "method_CG" => method))
        full = Dirac_operator(U, x, p).D5DW
        @test !full.use_eo
        @test full.eo_cache === nothing
        clear_fermion!(y)
        diagnostics = solve_DinvX!(y, full, x)
        @test diagnostics.method == Symbol(method)
        mul!(image, full, y)
        @test _eo_relative_difference(image, x) < 1e-8
    end
    full_outer = Dirac_operator(U, x, merge(parameters, Dict("evenodd" => false)))
    full = full_outer.D5DW
    references = map((false, true)) do adj
        solution = similar(x)
        clear_fermion!(solution)
        solve_DinvX!(solution, adj ? full' : full, x)
        solution
    end
    for method in ("bicg", "bicgstab", "cg")
        outer = Dirac_operator(U, x, merge(parameters, Dict("method_CG" => method)))
        D = outer.D5DW
        for adj in (false, true)
            # In-place source/destination exercises the reconstruction source.
            substitute_fermion!(y, x)
            diagnostics = solve_DinvX!(y, adj ? D' : D, y)
            @test diagnostics.method == Symbol(method)
            @test !halo_is_dirty(y.f)
            mul!(image, adj ? D' : D, y)
            residual = _eo_relative_difference(image, x)
            @test residual < 1e-8
            @test _eo_relative_difference(y, references[adj ? 2 : 1]) < 1e-8
            @test _eo_pool_free(D._temporary_fermion_forCG)
            @info "EO full-system residual" method adj residual

            # The public PV-composed wrapper also routes inversions via EO,
            # while retaining its full-field definition W=D(m)D(PV)^{-1}.
            diagnostics = solve_DinvX!(y, adj ? outer' : outer, x)
            @test diagnostics.method == Symbol(method)
            @test !halo_is_dirty(y.f)
            mul!(image, adj ? full_outer' : full_outer, y)
            @test _eo_relative_difference(image, x) < 1e-8
            mul!(adj_image, adj ? outer' : outer, y)
            @test _eo_relative_difference(adj_image, image) < 1e-8
        end
        # Failure must release nested Schur and Krylov workspaces.
        @test_throws ErrorException EO_Dop.solve_DinvX_eo!(y, D, x; maxsteps=0)
        @test _eo_pool_free(D._temporary_fermion_forCG)
        clear_fermion!(image)
        @test solve_DinvX!(y, D, image).iterations == 0
        @test !halo_is_dirty(y.f)
        @test real(dot(y, y)) == 0
    end
    @test_throws ErrorException EO_Dop.solve_DdagD_even!(y, A, x; maxsteps=0)
    @test _eo_pool_free(A._temporary_fermion_forCG)
    @test_throws ArgumentError Dirac_operator(U, x,
        merge(parameters, Dict("method_CG" => "unsupported")))
    twisted = DomainwallFermion_5D_MPILattice(U[1], 4;
        operator_name="GeneralizedDomainwall", boundarycondition=[1, 1, 1, -1, -1])
    @test_throws ArgumentError Dirac_operator(U, twisted, parameters)
    # Odd global extents break the parity graph through periodic boundaries.
    if nprocs == 1
        odd_U = gauge_configuration((3, 2, 2, 2); colors=3, halo=1,
            start=:cold, process_grid=(1, 1, 1, 1), comm=LDO_TEST_COMM, verbose=0)
        odd_x = Initialize_pseudofermion_fields(odd_U[1], "GeneralizedDomainwall"; L5=4)
        @test_throws ArgumentError Dirac_operator(odd_U, odd_x, parameters)
    end
    # When L5=1 the diagonal and fifth-neighbour entries coincide.
    one_slice = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5=1)
    gauss_distribution_fermion!(one_slice; seed=45)
    single_parameters = merge(parameters, Dict("L5" => 1,
        "as" => [1.0], "bs" => [1.5], "cs" => [0.5]))
    single = Dirac_operator(U, one_slice, single_parameters).D5DW
    single_image, single_inverse = similar(one_slice), similar(one_slice)
    EO_Dop.GeneralizedD5DW_keep_parity!(one_slice, true)
    EO_Dop.D5DW_diag_solve_block!(single_inverse, single, one_slice, true)
    mul!(single_image, single, single_inverse)
    EO_Dop.GeneralizedD5DW_keep_parity!(single_image, true)
    @test _eo_relative_difference(single_image, one_slice) < 2e-13
end

@testset "Generalized domain-wall EO hopping and halo" begin
    nprocs = ldo_test_comm_size()
    # Length four distinguishes the forward and backward neighbours. Both
    # the specialized SU(3) hopping and generic-color path must agree with D.
    @testset "colors=$colors L5=$L5" for colors in (3, 2), L5 in (1, 3)
        U = gauge_configuration((4nprocs, 2, 2, 2); colors, halo=1,
            start=:hot, seed=UInt64(71), process_grid=(nprocs, 1, 1, 1),
            comm=LDO_TEST_COMM, verbose=0)
        x = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5)
        gauss_distribution_fermion!(x; seed=72)
        source, masked, image, reference = similar(x), similar(x), similar(x), similar(x)
        parameters = merge(_eo_parameters(), Dict(
            "L5" => L5, "as" => [0.9, 1.1, 0.95][1:L5],
            "bs" => [1.7, 1.2, 0.9][1:L5], "cs" => [0.6, 0.3, 0.1][1:L5]))
        A = Dirac_operator(U, x, parameters).D5DW

        for adj in (false, true), even in (false, true)
            op = adj ? A' : A
            hop! = adj ? EO_Dop.GeneralizedD5DW_offdiag_adjoint_blockx_clean! :
                EO_Dop.GeneralizedD5DW_offdiag_blockx_clean!
            diag! = adj ? EO_Dop.D5DW_diag_solve_adjoint_direct! :
                EO_Dop.D5DW_diag_solve_block!

            # Retain nonzero values on both source parities. Scaling only the
            # core also leaves stale halo values for the hopping to refresh.
            substitute_fermion!(source, x)
            mul!(source.f, 1.25, x.f)
            @test halo_is_dirty(source.f)
            substitute_fermion!(masked, source)
            EO_Dop.GeneralizedD5DW_keep_parity!(masked, !even)
            mul!(reference, op, masked)
            EO_Dop.GeneralizedD5DW_keep_parity!(reference, even)
            substitute_fermion!(image, x)
            hop!(image, A, source, even; sethalo=false)
            @test halo_is_dirty(image.f)
            @test _eo_relative_difference(image, reference) < 2e-12
            substitute_fermion!(masked, image)
            EO_Dop.GeneralizedD5DW_keep_parity!(masked, !even)
            @test real(dot(masked, masked)) == 0

            # The inverse diagonal needs only physical fifth slices, so it
            # must neither consume nor synchronize its source halo.
            mul!(source.f, 0.75, x.f)
            @test halo_is_dirty(source.f)
            diag!(image, A, source, even; sethalo=false)
            @test halo_is_dirty(source.f)
            @test halo_is_dirty(image.f)
            mul!(reference, op, image)
            EO_Dop.GeneralizedD5DW_keep_parity!(reference, even)
            substitute_fermion!(masked, source)
            EO_Dop.GeneralizedD5DW_keep_parity!(masked, even)
            @test _eo_relative_difference(reference, masked) < 2e-12
        end

        S = EO_Dop.D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)
        for adj in (false, true)
            op = adj ? S' : S
            mul!(reference, op, x)
            substitute_fermion!(masked, x)
            EO_Dop.GeneralizedD5DW_keep_parity!(masked, true)
            mul!(image, op, masked)
            @test _eo_relative_difference(image, reference) < 2e-12

            # The fused final subtraction must preserve the even input until
            # every hopping and diagonal step has finished, also in-place.
            substitute_fermion!(image, x)
            mul!(image, op, image)
            @test _eo_relative_difference(image, reference) < 2e-12
            EO_Dop.GeneralizedD5DW_keep_parity!(image, false)
            @test real(dot(image, image)) == 0
            @test _eo_pool_free(A._temporary_fermion_forCG)
        end
    end
end

@testset "Generalized domain-wall EO local hopping faces" begin
    if ldo_test_comm_size() == 1
        # A wider halo exposes untouched outer layers; complex phases check
        # that the two nearest faces use the correct wrap factors.
        U = gauge_configuration((4, 2, 2, 2); colors=3, halo=2,
            start=:hot, seed=UInt64(81), process_grid=(1, 1, 1, 1),
            comm=LDO_TEST_COMM, verbose=0)
        phases = [cis(0.31), cis(-0.47), cis(0.19), -1, 1]
        x = DomainwallFermion_5D_MPILattice(U[1], 3;
            operator_name="GeneralizedDomainwall", boundarycondition=phases)
        gauss_distribution_fermion!(x; seed=82)
        set_halo!(x.f)
        A = Dirac_operator(U, x, _eo_parameters(3)).D5DW
        source, masked, image = similar(x), similar(x), similar(x)
        reference, full_reference = similar(x), similar(x)
        nw = x.f.nw

        for adj in (false, true), even in (false, true)
            op = adj ? A' : A
            hop! = adj ? EO_Dop.GeneralizedD5DW_offdiag_adjoint_blockx_clean! :
                EO_Dop.GeneralizedD5DW_offdiag_blockx_clean!
            substitute_fermion!(masked, x)
            EO_Dop.GeneralizedD5DW_keep_parity!(masked, !even)
            mul!(reference, op, masked)
            EO_Dop.GeneralizedD5DW_keep_parity!(reference, even)

            # Restore only the physical data after poisoning every padded
            # value. The partial exchange must retain the dirty full halo.
            fill!(source.f.A, ComplexF64(NaN, NaN))
            LatticeMatrices.substitute!(source.f, x.f)
            epochs = halo_epochs(source.f)
            EO_Dop._generalized_eo_hopping_halo!(source)
            @test halo_is_dirty(source.f)
            @test halo_epochs(source.f) == epochs
            padded = Array(source.f.A)
            @test isnan(real(padded[1, 1, nw, nw, nw + 1, nw + 1, nw + 1]))
            @test isnan(real(padded[1, 1, nw + 1, nw + 1, nw + 1, nw + 1, nw]))
            @test isnan(real(padded[1, 1, 1, nw + 1, nw + 1, nw + 1, nw + 1]))
            hop!(image, A, source, even; sethalo=false)
            @test halo_is_dirty(source.f)
            @test _eo_relative_difference(image, reference) < 2e-12

            # A later public stencil must refresh the complete halo,
            # including fifth ghosts, corners, and the second halo layer.
            mul!(full_reference, op, x)
            mul!(image, op, source)
            @test !halo_is_dirty(source.f)
            @test Array(source.f.A) ≈ Array(x.f.A)
            @test _eo_relative_difference(image, full_reference) < 2e-12
            @test _eo_pool_free(A._temporary_fermion_forCG)
        end
    end
end

@testset "Generalized domain-wall EO action and force" begin
    nprocs = ldo_test_comm_size()
    U = gauge_configuration((2nprocs, 2, 2, 2); colors=3, halo=1,
        start=:hot, seed=UInt64(48), process_grid=(nprocs, 1, 1, 1),
        comm=LDO_TEST_COMM, verbose=0)
    phi = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5=4)
    noise, even_noise, mixed = similar(phi), similar(phi), similar(phi)
    action = FermiAction(Dirac_operator(U, phi, _eo_parameters()), Dict())
    provider = PseudofermionMDAction(action, phi)
    workspace = md_action_workspace(provider, U)
    refresh_pseudofermion!(provider, U, noise; seed=49)
    substitute_fermion!(even_noise, noise)
    EO_Dop.GeneralizedD5DW_keep_parity!(even_noise, true)
    noise_norm = real(dot(even_noise, even_noise))
    value = evaluate_FermiAction(action, U, phi)
    @test value ≈ noise_norm rtol=1e-10 atol=1e-8
    @test md_potential(provider, U, workspace) ≈ value rtol=1e-12
    substitute_fermion!(mixed, phi)
    EO_Dop.GeneralizedD5DW_keep_parity!(mixed, false)
    @test real(dot(mixed, mixed)) < 1e-25
    substitute_fermion!(mixed, noise)
    EO_Dop.GeneralizedD5DW_keep_parity!(mixed, false)
    axpby!(1, phi, 1, mixed)
    @test evaluate_FermiAction(action, U, mixed) ≈ value rtol=1e-12

    force = calc_UdSfdU(action, U, phi)
    # Perturb links both within a rank and on a periodic time boundary, in
    # diagonal and off-diagonal SU(3) directions.
    directions = (Matrix(im * Diagonal([1.0, -1.0, 0.0])) / sqrt(2),
        ComplexF64[0 1 0; -1 0 0; 0 0 0] / sqrt(2))
    for (mu, local_site, direction) in ((1, (1, 1, 1, 1), directions[1]),
        (4, (2, 1, 1, 2), directions[2]))
        site = local_site .+ U[mu].U.nw
        # The small finite-difference oracle runs on the host; production
        # action/force evaluation continues to use the selected JACC backend.
        link_data = Array(U[mu].U.A)
        link = copy(link_data[:, :, site...])
        F = Array(force[mu].U.A)[:, :, site...]
        anti = (F - F') / 2
        ta = anti - tr(anti) * I / 3
        local_derivative = ldo_test_comm_rank() == 0 ? -2real(tr(ta * direction)) : 0.0
        analytic = ldo_test_allreduce_sum(local_derivative)
        for h in (1e-4, 3e-5)
            up, um = similar(U), similar(U)
            substitute_U!(up, U)
            substitute_U!(um, U)
            if ldo_test_comm_rank() == 0
                plus, minus = copy(link_data), copy(link_data)
                plus[:, :, site...] .= exp(h * direction) * link
                minus[:, :, site...] .= exp(-h * direction) * link
                copyto!(up[mu].U.A, plus)
                copyto!(um[mu].U.A, minus)
            end
            mark_halo_dirty!(up[mu].U)
            mark_halo_dirty!(um[mu].U)
            set_wing_U!(up)
            set_wing_U!(um)
            numerical = (evaluate_FermiAction(action, up, phi) -
                evaluate_FermiAction(action, um, phi)) / (2h)
            @test analytic ≈ numerical rtol=2e-5 atol=2e-7
            @info "EO force finite difference" mu h analytic numerical
        end
    end

    md_force = initialize_TA_Gaugefields(U)
    md_force!(md_force, provider, U, workspace)
    momentum = initialize_TA_Gaugefields(U)
    clear_U!(momentum)
    EO_Dop.calc_p_UdSfdU!(momentum, action, U, phi, 0.25)
    EO_Dop.calc_p_UdSfdU!(momentum, action, U, phi, 0.75)
    for mu in 1:4
        @test Array(momentum[mu].a.A) ≈ Array(md_force[mu].a.A) rtol=2e-10 atol=2e-11
    end
    reset_trajectory_state!(provider)
    @test evaluate_FermiAction(action, U, phi) ≈ value rtol=1e-12
    for pool in (action._temporary_fermionfields, action._temporary_gaugefields,
        action.diracoperator.D5DW._temporary_fermion_forCG,
        action.diracoperator.D5DW_PV._temporary_fermion_forCG)
        @test _eo_pool_free(pool)
    end
end

function _eo_backend_serial_reference(links, source, phases, halo)
    comm = SerialCommunicator()
    grid = (1, 1, 1, 1)
    gsize = size(links[1])[3:end]
    U = gauge_configuration(gsize; colors=3, halo, start=:cold,
        process_grid=grid, comm, verbose=0)
    for mu in 1:4
        field = LatticeMatrix(links[mu], 4, grid;
            nw=halo, comm0=comm, device_mapping=:current)
        LatticeMatrices.substitute!(U[mu].U, field)
    end
    set_wing_U!(U)
    L5 = size(source, 7)
    x = DomainwallFermion_5D_MPILattice(U[1], L5;
        operator_name="GeneralizedDomainwall", boundarycondition=phases)
    field = LatticeMatrix(source, 5, (grid..., 1);
        nw=halo, phases, comm0=comm, device_mapping=:current)
    LatticeMatrices.substitute!(x.f, field)
    set_wing_fermion!(x)
    parameters = merge(_eo_parameters(L5), Dict("method_CG" => "bicgstab"))
    A = Dirac_operator(U, x, parameters).D5DW
    full = Dirac_operator(U, x, merge(parameters, Dict("evenodd" => false))).D5DW
    S = EO_Dop.D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)
    return (; x, image=similar(x), A, S, full)
end

function _eo_backend_decomposition_test(axis, halo)
    nprocs = ldo_test_comm_size()
    rank = ldo_test_comm_rank()
    # On two ranks, local extent 3 makes the parity origin differ on rank 1.
    # The global periodic graph stays even. Both spatial and time cuts are used.
    local_extent = iseven(nprocs) ? 3 : 4
    gsize = ntuple(d -> d == axis ? local_extent * nprocs : 2, 4)
    grid = ntuple(d -> d == axis ? nprocs : 1, 4)
    phases = [cis(0.31), 1, 1, -1, 1]
    L5 = 3
    U = gauge_configuration(gsize; colors=3, halo, start=:hot, seed=UInt64(91),
        process_grid=grid, comm=LDO_TEST_COMM, verbose=0)
    x = DomainwallFermion_5D_MPILattice(U[1], L5;
        operator_name="GeneralizedDomainwall", boundarycondition=phases)
    gauss_distribution_fermion!(x; seed=92)
    saved, image, solution = similar(x), similar(x), similar(x)
    substitute_fermion!(saved, x)
    A = Dirac_operator(U, x,
        merge(_eo_parameters(L5), Dict("method_CG" => "bicgstab"))).D5DW
    S = EO_Dop.D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)

    @test x.f.A isa JACC.array_type()
    @test all(link -> link.U.A isa JACC.array_type(), U)
    @test A.eo_cache.lower isa JACC.array_type()
    @test A.eo_cache.upper_adjoint isa JACC.array_type()
    if EO_REQUIRE_GPU
        @test !(x.f.A isa Array)
        @test !(A.eo_cache.lower isa Array)
    end
    @test x.f.PN[axis] == local_extent
    @test x.f.dims == (grid..., 1)
    @info "EO lattice layout" rank axis halo global_size=gsize local_size=x.f.PN storage=typeof(x.f.A) transport=mpi_transport_info(x.f)

    # Every rank participates in gathers; only rank 0 constructs the serial
    # oracle, so another rank's selected GPU is never changed to device zero.
    links = [gather_matrix(U[mu].U) for mu in 1:4]
    source = gather_matrix(x.f)
    reference = rank == 0 ? _eo_backend_serial_reference(links, source, phases, halo) : nothing
    for which in (:full, :schur), adj in (false, true)
        op = which === :full ? A : S
        # Restore just the core after poisoning all ghosts. Communication must
        # repair remote and periodic faces before a stencil reads them.
        fill!(x.f.A, ComplexF64(NaN, NaN))
        LatticeMatrices.substitute!(x.f, saved.f)
        @test halo_is_dirty(x.f)
        mul!(image, adj ? op' : op, x)
        @test !halo_is_dirty(image.f)
        result = gather_matrix(image.f)
        if rank == 0
            refop = which === :full ? reference.A : reference.S
            mul!(reference.image, adj ? refop' : refop, reference.x)
            expected = gather_matrix(reference.image.f)
            @test result ≈ expected rtol=3e-11 atol=3e-11
        end
    end

    for adj in (false, true)
        op = adj ? A' : A
        clear_fermion!(solution)
        solve_DinvX!(solution, op, saved)
        @test !halo_is_dirty(solution.f)
        mul!(image, op, solution)
        @test _eo_relative_difference(image, saved) < 1e-8
        result = gather_matrix(solution.f)
        if rank == 0
            clear_fermion!(reference.image)
            refop = adj ? reference.full' : reference.full
            solve_DinvX!(reference.image, refop, reference.x)
            expected = gather_matrix(reference.image.f)
            @test norm(result - expected) / norm(expected) < 1e-8
        end
    end
    @test _eo_pool_free(A._temporary_fermion_forCG)
    return nothing
end


@testset "Generalized domain-wall EO decomposition" begin
    @testset "decomposition axis=$axis halo=$halo" for axis in (1, 4), halo in (1, 2)
        _eo_backend_decomposition_test(axis, halo)
    end
end

function _eo_comparison_timed_solve!(solution, D, source)
    # Every sample starts from zero. Setup, clearing, and pending backend work
    # are excluded; EO reduction/reconstruction and CG RHS formation are not.
    clear_fermion!(solution)
    JACC.synchronize()
    ldo_test_barrier()
    elapsed = @elapsed begin
        solve_DinvX!(solution, D, source)
        JACC.synchronize()
    end
    # All ranks report the slowest rank's time, including solver communication
    # and device completion but excluding the initial barrier and this reduce.
    return ldo_test_allreduce_max(elapsed)
end

function _eo_comparison_relative_difference!(work, left, right)
    substitute_fermion!(work, left)
    axpby!(-1, right, 1, work)
    return sqrt(real(dot(work, work)) / max(real(dot(right, right)), eps(Float64)))
end

function _eo_comparison_residual!(work, D, solution, source)
    mul!(work, D, solution)
    axpby!(-1, source, 1, work)
    return sqrt(real(dot(work, work)) / real(dot(source, source)))
end

function _eo_comparison_median(values)
    ordered = sort(values)
    midpoint = fld(length(ordered), 2) + 1
    return isodd(length(ordered)) ? ordered[midpoint] :
           (ordered[midpoint - 1] + ordered[midpoint]) / 2
end

@testset "Generalized domain-wall EO/non-EO solve comparison" begin
    N = parse(Int, get(ENV, "LDO_EO_BENCH_N", "4"))
    L5 = parse(Int, get(ENV, "LDO_EO_BENCH_L5", "4"))
    repeats = parse(Int, get(ENV, "LDO_EO_BENCH_REPEATS", "5"))
    axis = parse(Int, get(ENV, "LDO_EO_BENCH_AXIS", "1"))
    nprocs = ldo_test_comm_size()
    N >= 2 && iseven(N) || throw(ArgumentError("LDO_EO_BENCH_N must be positive and even"))
    L5 >= 2 || throw(ArgumentError("LDO_EO_BENCH_L5 must be at least 2"))
    repeats >= 1 || throw(ArgumentError("LDO_EO_BENCH_REPEATS must be positive"))
    axis in 1:4 || throw(ArgumentError("LDO_EO_BENCH_AXIS must be in 1:4"))
    N % nprocs == 0 || throw(ArgumentError(
        "global extent LDO_EO_BENCH_N must be divisible by the MPI rank count"))

    gsize = (N, N, N, N)
    process_grid = ntuple(d -> d == axis ? nprocs : 1, 4)
    U = gauge_configuration(
        gsize; colors=3, halo=1, start=:hot, seed=UInt64(47),
        process_grid, comm=LDO_TEST_COMM, verbose=0)
    source = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5)
    gauss_distribution_fermion!(source; seed=42)
    solution_full, solution_eo = similar(source), similar(source)
    work = similar(source)
    references = (similar(source), similar(source))
    tolerance = 1e-8

    parameters = Dict{String,Any}(
        "Dirac_operator" => "GeneralizedDomainwall",
        "L5" => L5, "mass" => 0.15, "M" => -1.0,
        "as" => collect(range(0.9, 1.1; length=L5)),
        "bs" => collect(range(1.7, 0.9; length=L5)),
        "cs" => collect(range(0.6, 0.1; length=L5)),
        "eps_CG" => 1e-20, "MaxCGstep" => 3000, "verbose_level" => 0)

    if ldo_test_comm_rank() == 0
        @info "EO solve comparison setup (setup/warmup excluded)" lattice=gsize L5 repeats ranks=nprocs process_grid backend=JACC.backend threads=Threads.nthreads() storage=typeof(source.f.A) eps_CG=parameters["eps_CG"] tolerance
    end

    for method in ("cg", "bicg", "bicgstab")
        @testset "$method" begin
            full_parameters = merge(parameters, Dict("method_CG" => method, "evenodd" => false))
            eo_parameters = merge(parameters, Dict("method_CG" => method, "evenodd" => true))
            full_outer = Dirac_operator(U, source, full_parameters)
            eo_outer = Dirac_operator(U, source, eo_parameters)
            D_full, D_eo = full_outer.D5DW, eo_outer.D5DW

            for adj in (false, true)
                direction = adj ? "adjoint" : "forward"
                @testset "$direction" begin
                    full_operator = adj ? D_full' : D_full
                    eo_operator = adj ? D_eo' : D_eo
                    reference = references[adj ? 2 : 1]

                    # Warm the same helper and both paths before sampling.
                    _eo_comparison_timed_solve!(solution_full, full_operator, source)
                    _eo_comparison_timed_solve!(solution_eo, eo_operator, source)

                    times_full, times_eo = zeros(repeats), zeros(repeats)
                    residuals_full, residuals_eo = zeros(repeats), zeros(repeats)
                    differences = zeros(repeats)
                    for sample in 1:repeats
                        # Alternate the order to reduce systematic timing bias.
                        order = isodd(sample) ? (false, true) : (true, false)
                        for use_eo in order
                            GC.gc()
                            if use_eo
                                times_eo[sample] = _eo_comparison_timed_solve!(
                                    solution_eo, eo_operator, source)
                            else
                                times_full[sample] = _eo_comparison_timed_solve!(
                                    solution_full, full_operator, source)
                            end
                        end

                        # Use the SAME full equation for both residuals, outside
                        # timing; reduced/normal residuals alone do not suffice.
                        residuals_full[sample] = _eo_comparison_residual!(
                            work, full_operator, solution_full, source)
                        residuals_eo[sample] = _eo_comparison_residual!(
                            work, full_operator, solution_eo, source)
                        differences[sample] = _eo_comparison_relative_difference!(
                            work, solution_eo, solution_full)
                        @test residuals_full[sample] < tolerance
                        @test residuals_eo[sample] < tolerance
                        @test differences[sample] < tolerance
                    end

                    # Also compare CG, BiCG, and BiCGStab against each other.
                    if method == "cg"
                        substitute_fermion!(reference, solution_full)
                    end
                    @test _eo_comparison_relative_difference!(work, solution_full, reference) < tolerance
                    @test _eo_comparison_relative_difference!(work, solution_eo, reference) < tolerance

                    full_ms = 1000 * _eo_comparison_median(times_full)
                    eo_ms = 1000 * _eo_comparison_median(times_eo)
                    speedup = full_ms / eo_ms
                    @test isfinite(speedup) && speedup > 0
                    # Above 1 means EO is faster; below 1 means EO is slower.
                    # The speed ratio is informational, not a pass criterion.
                    if ldo_test_comm_rank() == 0
                        @info "EO/non-EO solve timing (median; speedup = non-EO / EO)" method direction non_eo_ms=full_ms eo_ms speedup
                        @info "EO/non-EO solution agreement" method direction non_eo_residual=maximum(residuals_full) eo_residual=maximum(residuals_eo) relative_solution_difference=maximum(differences)
                    end

                    # Exercise the new non-EO CG public API beyond timing:
                    # in-place inputs, diagnostics, and the PV-composed wrapper.
                    if method == "cg"
                        substitute_fermion!(solution_full, source)
                        diagnostics = solve_DinvX!(solution_full, full_operator, solution_full)
                        @test diagnostics isa SolverDiagnostics
                        @test diagnostics.method === :cg
                        @test _eo_comparison_relative_difference!(work, solution_full, reference) < tolerance

                        full_wrapped = adj ? full_outer' : full_outer
                        eo_wrapped = adj ? eo_outer' : eo_outer
                        diagnostics = solve_DinvX!(solution_full, full_wrapped, source)
                        solve_DinvX!(solution_eo, eo_wrapped, source)
                        @test diagnostics isa SolverDiagnostics
                        @test _eo_comparison_residual!(work, full_wrapped, solution_full, source) < tolerance
                        @test _eo_comparison_relative_difference!(work, solution_eo, solution_full) < tolerance
                    end
                end
            end

            if method == "cg"
                # A zero RHS and a failed normal solve must release all leases.
                clear_fermion!(work)
                clear_fermion!(solution_full)
                @test solve_DinvX!(solution_full, D_full, work).iterations == 0
                @test real(dot(solution_full, solution_full)) == 0
                for adj in (false, true)
                    failing = Dirac_operator(U, source,
                        merge(full_parameters, Dict("MaxCGstep" => 0))).D5DW
                    @test_throws ErrorException solve_DinvX!(
                        solution_full, adj ? failing' : failing, source)
                    @test !any(failing._temporary_fermi._data._flagusing)
                    @test !any(failing._temporary_fermion_forCG._data._flagusing)
                end
            end
        end
    end
end
