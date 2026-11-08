module WilsonActionCleanupTests
using Test, LinearAlgebra, Gaugefields, LatticeDiracOperators, LatticeMatrices, JACC
const DO = LatticeDiracOperators.Dirac_operators
JACC.@init_backend
include("test_communicator.jl")

flags(pool) = copy(pool._data._flagusing)
function action_flags(action)
    (flags(action._temporary_fermionfields), flags(action._temporary_gaugefields))
end

@testset "Wilson action scratch ownership" begin
    # Exhaust a real pool during batch acquisition while preserving a caller's slot.
    pool = DO.Temporalfields(zeros(3); num=3, Nmax=3)
    _, held = DO.get_temp(pool)
    before = flags(pool)
    @test_throws AssertionError DO._with_wilson_temporary_fields(identity, pool, 3)
    @test flags(pool) == before
    @test_throws ErrorException DO._with_wilson_temporary_fields(pool, 2) do fields
        error("injected callback failure")
    end
    @test flags(pool) == before
    @test DO._with_wilson_temporary_fields(length, pool, 2) == 2
    @test flags(pool) == before
    DO.unused!(pool, held)

    nprocs = ldo_test_comm_size()
    # Legacy arrays have no distributed/GPU implementation; the standard path
    # below runs on every configured JACC and MPI backend.
    backends = get(ENV, "LDO_CLEANUP_STANDARD_ONLY", "false") == "true" ?
        (:standard,) : (:legacy, :standard)
    for backend in backends
        U = if backend == :legacy
            Initialize_4DGaugefields(3, 0, 2, 2, 2, 2; condition="cold", verbose_level=0)
        else
            Initialize_Gaugefields(3, 1, 4, 2, 2, 2; condition="cold",
                isMPILattice=true, PEs=(nprocs, 1, 1, 1), verbose_level=0)
        end
        source = Initialize_pseudofermion_fields(U[1], "Wilson"; nowing=true)
        clear_fermion!(source)
        if backend == :legacy
            setindex_global!(source, 1, 1, 1, 1, 1, 1, 1)
        else
            values = zeros(ComplexF64, 3, 4, 4, 2, 2, 2)
            values[1, 1, 1, 1, 1, 1] = 1
            lattice = LatticeMatrix(values, 4, (nprocs, 1, 1, 1);
                nw=1, phases=source.f.phases, comm0=source.f.comm)
            substitute!(source.f, lattice)
        end
        set_wing_fermion!(source)
        parameters = Dict{String,Any}("Dirac_operator" =>
            backend == :legacy ? "Wilson" : "WilsonClover",
            "κ" => 0.1, "cSW" => 1.17, "faster version" => false,
            "eps_CG" => 1e-20, "MaxCGstep" => 0, "verbose_level" => 0)
        failed = FermiAction(Dirac_operator(U, source, parameters), Dict())
        parameters["MaxCGstep"] = 1000
        good = FermiAction(Dirac_operator(U, source, parameters), Dict())
        # Reuse exactly the pools involved in each failed solve.
        good._temporary_fermionfields._data = failed._temporary_fermionfields._data
        good._temporary_gaugefields._data = failed._temporary_gaugefields._data
        # Freeze capacity to turn leaked slots into a reproducible later failure.
        for pool in (good._temporary_fermionfields, good._temporary_gaugefields)
            pool._data.Nmax = length(pool)
        end
        foreach(clear_fermion!, good._temporary_fermionfields._data._data)
        force = [similar(link) for link in U]
        momentum = initialize_TA_Gaugefields(U)
        calls = (
            action -> evaluate_FermiAction(action, U, source),
            action -> calc_UdSfdU!(force, action, U, source),
            action -> DO.calc_p_UdSfdU!(momentum, action, U, source),
        )
        initial = action_flags(good)
        for call in calls
            for _ in 1:2
                foreach(clear_fermion!, good._temporary_fermionfields._data._data)
                err = try
                    call(failed)
                    nothing
                catch err
                    err
                end
                @test err isa ErrorException
                @test occursin("not converged", sprint(showerror, err))
                @test action_flags(good) == initial
                call(good)
                @test action_flags(good) == initial
            end
        end
        # Failure after the solve, inside raw force / momentum accumulation.
        @test_throws BoundsError calc_UdSfdU!(typeof(force)(), good, U, source)
        @test action_flags(good) == initial
        @test_throws BoundsError DO.calc_p_UdSfdU!(typeof(momentum)(), good, U, source)
        @test action_flags(good) == initial
        Y = similar(source)
        @test_throws BoundsError DO.calc_p_UdSfdU_fromX!(
            typeof(momentum)(), Y, good, U, source)
        @test action_flags(good) == initial
        # Repeated successful calls still fit within the original capacity.
        for _ in 1:2, call in calls
            call(good)
            @test action_flags(good) == initial
        end
        @test isfinite(evaluate_FermiAction(good, U, source))
        @test action_flags(good) == initial
    end
end
end
