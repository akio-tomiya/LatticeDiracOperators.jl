module DomainwallPhysicalCoefficientTests
using Test, LinearAlgebra, JACC, Gaugefields, LatticeMatrices, LatticeDiracOperators
JACC.@init_backend
include(joinpath(@__DIR__, "test_communicator.jl"))
select_device_by_mpi_rank!(LDO_TEST_COMM)

function parameters(L5)
    Dict{String,Any}("Dirac_operator" => "GeneralizedDomainwall", "L5" => L5,
        "mass" => 0.15, "M" => -1.0, "as" => ones(L5),
        "bs" => [1.6 - 0.1s for s in 1:L5], "cs" => [0.5 - 0.07s for s in 1:L5],
        "eps_CG" => 1e-25, "MaxCGstep" => 5000, "method_CG" => "bicg", "verbose_level" => 0)
end

global_array(x) = gather_and_bcast_matrix(x.f)

@testset "Physical propagators with variable coefficients" begin
    ranks = ldo_test_comm_size()
    U = gauge_configuration((4, 2, 2, 4); colors=3, halo=1, start=:hot,
        seed=UInt64(91), process_grid=(ranks, 1, 1, 1), comm=LDO_TEST_COMM, verbose=0)
    x = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5=4)
    p = parameters(4)
    W = Dirac_operator(U, x, p)
    source4 = Initialize_WilsonFermion(U[1]; nowing=true)
    gauss_distribution_fermion!(source4; seed=UInt64(81))
    source5, embedded, raw_image, solution, scaled_solution = (similar(x) for _ in 1:5)
    physical = similar(source4)
    shamir = Dirac_operator(U, x, merge(p, Dict("Dirac_operator" => "Domainwall")))
    LatticeMatrices.domainwall_import_physical_source!(embedded.f, source4.f)
    mul!(raw_image, shamir.D5DW, embedded)
    eta = global_array(embedded)
    # Independent D_W reference: D_Shamir = I + D_W - F_m.
    # Form the fifth hopping explicitly on host arrays, including mass wraps.
    Feta = similar(eta)
    for s in 1:4, spin in 1:4
        neighbor = spin <= 2 ? mod1(s-1, 4) : mod1(s+1, 4)
        factor = (spin <= 2 && s == 1) || (spin > 2 && s == 4) ? -W.mass : 1.0
        Feta[:, spin, :, :, :, :, s] .= factor .* eta[:, spin, :, :, :, :, neighbor]
    end
    DWeta = global_array(raw_image) - eta + Feta
    a = [0.8, 1.2, 0.9, 1.1]
    scaled = Renew(W, W.bs, W.cs; as=a)
    import_domainwall_physical_source!(source5, scaled, source4)
    expected_source = similar(eta)
    for s in 1:4
        expected_source[:, :, :, :, :, :, s] .= a[s] .* (
            eta[:, :, :, :, :, :, s] .- W.cs[s] .* DWeta[:, :, :, :, :, :, s])
    end
    @test global_array(source5) ≈ expected_source rtol=2e-13 atol=2e-13
    solve_domainwall_physical_propagator!(solution, physical, source5, W, source4)
    for eo in (false, true)
        scaled_eo = Dirac_operator(U, x, merge(p, Dict("as" => a, "evenodd" => eo)))
        solve_domainwall_physical_propagator!(scaled_solution, physical, source5, scaled_eo, source4)
        @test global_array(scaled_solution) ≈ global_array(solution) rtol=3e-11 atol=3e-12
        mul!(raw_image, scaled_eo.D5DW, scaled_solution)
        @test global_array(raw_image) ≈ global_array(source5) rtol=3e-11 atol=3e-12
        @test !any(scaled_eo.D5DW._temporary_fermi._data._flagusing)
    end
    # Scalar b,c are sums/differences of the generalized coefficients.
    scalar = Dirac_operator(U, x, merge(p,
        Dict("Dirac_operator" => "MobiusDomainwall", "b" => 1.8, "c" => 1.0)))
    constant = Renew(W, fill(1.4, 4), fill(0.4, 4))
    solve_domainwall_physical_propagator!(solution, physical, source5, scalar, source4)
    solve_domainwall_physical_propagator!(scaled_solution, physical, source5, constant, source4)
    @test global_array(scaled_solution) ≈ global_array(solution) rtol=3e-11 atol=3e-12
    @test_throws ArgumentError import_domainwall_physical_source!(source5,
        Renew(W, W.bs, W.cs; as=zeros(4)), source4)
    @test_throws ArgumentError solve_domainwall_physical_propagator!(source5, physical, source5, W, source4)

    @testset "Adjoint mres coefficient gradient" begin
        x2 = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5=2)
        W2 = Dirac_operator(U, x2, merge(parameters(2),
            Dict("as" => [0.85, 1.15], "evenodd" => true)))
        slices = [2, 3]
        source_position = (1, 1, 1, 2)
        g = domainwall_residual_mass_gradient(W2, x2; timeslices=slices, source_position)
        @test all(isfinite, g.db) && all(isfinite, g.dc)
        @test g.value ≈ real(sum(g.correlators.J5qP[slices]) / sum(g.correlators.PP[slices]))
        function loss(bs, cs)
            op = Renew(W2, bs, cs)
            prop = domainwall_physical_point_propagators(op, x2; source_position)
            corr = domainwall_residual_mass_correlator(prop.five_dimensional; origin=source_position)
            real(sum(corr.J5qP[slices]) / sum(corr.PP[slices]))
        end
        h = 2e-5
        for (coefficients, analytic, which) in ((W2.bs, g.db, :b), (W2.cs, g.dc, :c)), s in 1:2
            plus, minus = copy(coefficients), copy(coefficients)
            plus[s] += h; minus[s] -= h
            numerical = which == :b ? (loss(plus, W2.cs)-loss(minus, W2.cs))/(2h) :
                                       (loss(W2.bs, plus)-loss(W2.bs, minus))/(2h)
            @test analytic[s] ≈ numerical rtol=2e-5 atol=2e-8
        end
        # A small learning step should reduce exactly the selected new objective.
        @test loss(W2.bs - 1e-3g.db, W2.cs - 1e-3g.dc) < g.value
        @test !any(W2.D5DW._temporary_fermi._data._flagusing)
        @test !any(W2.D5DW._temporary_fermion_forCG._data._flagusing)
        @test_throws ArgumentError domainwall_residual_mass_gradient(W2, x2; timeslices=Int[])
        @test_throws ArgumentError domainwall_residual_mass_gradient(W2, x2; timeslices=[0])
        @test_throws ArgumentError domainwall_residual_mass_gradient(W2, x2; timeslices=[2, 2])
    end
end
end
