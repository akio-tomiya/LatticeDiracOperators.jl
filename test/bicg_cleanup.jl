module BiCGCleanupTests
using Test, LinearAlgebra, LatticeDiracOperators, Gaugefields
const DO = LatticeDiracOperators.Dirac_operators

mutable struct Pool
    fields::Vector{Vector{Float64}}
    used::Set{Int}
    calls::Int
    fail_at::Int
end
function DO.get_temp(pool::Pool)
    pool.calls += 1
    pool.calls == pool.fail_at && error("injected acquisition failure")
    i = findfirst(i -> !(i in pool.used), eachindex(pool.fields))
    i === nothing && error("pool exhausted")
    push!(pool.used, i)
    return pool.fields[i], i
end
DO.unused!(pool::Pool, i) = delete!(pool.used, i)

mutable struct Op{P}
    pool::P
    failure::Symbol
    applies::Int
    copies::Int
end
DO.get_temporaryvectors_forCG(A::Op) = A.pool
Base.adjoint(A::Op) = A
function DO._solver_copy!(A::Op, y, x)
    A.copies += 1
    A.failure == Symbol("copy_", A.copies) && error("injected copy failure")
    copyto!(y, x)
end
function DO._solver_mul!(y, A::Op, x)
    A.applies += 1
    A.failure == Symbol("apply_", A.applies) && error("injected operator failure")
    y .= [1., 2., 4.] .* x
end
function DO._solver_axpby!(A::Op, a, x, b, y)
    A.failure == :axpby && error("injected residual failure")
    axpby!(a, x, b, y)
end
function DO._solver_finish!(A::Op, x)
    A.failure == :finish && error("injected finish failure")
    nothing
end

@testset "BiCG scratch ownership on every exit" begin
    for (solve, slots) in ((DO.bicg, 5), (DO.bicgstab, 7))
        modes = [:normal, :zero, :nonconverged, :copy_1, :apply_1,
                 :apply_2, :axpby, :finish]
        solve === DO.bicgstab && append!(modes, [:copy_2, :copy_3])
        for acquisition in 1:slots
            push!(modes, Symbol("allocation_", acquisition))
        end
        for mode in modes
            fail_at = startswith(string(mode), "allocation_") ?
                parse(Int, split(string(mode), '_')[end]) : 0
            # Slot 1 belongs to a caller and must never be released by the solver.
            pool = Pool([zeros(3) for _ in 1:slots+1], Set([1]), 0, fail_at)
            A = Op(pool, mode, 0, 0)
            x, b = zeros(3), mode == :zero ? zeros(3) : ones(3)
            if mode in (:normal, :zero)
                result = solve(x, A, b; eps=1e-24, verbose=Verbose_print(0))
                @test result isa SolverDiagnostics
                @test x ≈ b ./ [1., 2., 4.]
            else
                err = try
                    solve(x, A, b; eps=1e-24,
                        maxsteps=mode == :nonconverged ? 0 : 20,
                        verbose=Verbose_print(0))
                    nothing
                catch err
                    err
                end
                @test err isa ErrorException
                @test occursin(mode == :nonconverged ? "not converged" : "injected",
                    sprint(showerror, err))
            end
            @test pool.used == Set([1])
            A.failure = :none
            pool.fail_at = 0
            fill!(x, 0)
            fill!(b, 1)
            @test solve(x, A, b; eps=1e-24, verbose=Verbose_print(0)) isa SolverDiagnostics
            @test x ≈ [1., 0.5, 0.25]
            @test pool.used == Set([1])
        end
        # Check the production Temporalfields implementation as well.
        pool = DO.Temporalfields(zeros(3); num=slots+1, Nmax=slots+1)
        _, held = DO.get_temp(pool)
        before = copy(pool._data._flagusing)
        A = Op(pool, :apply_1, 0, 0)
        @test_throws ErrorException solve(zeros(3), A, ones(3); verbose=Verbose_print(0))
        @test pool._data._flagusing == before
        A.failure = :none
        x = zeros(3)
        @test solve(x, A, ones(3); eps=1e-24, verbose=Verbose_print(0)) isa SolverDiagnostics
        @test x ≈ [1., 0.5, 0.25]
        @test pool._data._flagusing == before
        DO.unused!(pool, held)
    end
end
end
