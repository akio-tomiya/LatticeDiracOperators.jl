module GenericCGCleanupTests
using Test, LinearAlgebra, LatticeDiracOperators
const DO = LatticeDiracOperators.Dirac_operators

struct Field <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(x::Field) = size(x.data)
Base.getindex(x::Field, i::Int) = x.data[i]
Base.setindex!(x::Field, v, i::Int) = (x.data[i] = v)
Base.IndexStyle(::Type{Field}) = IndexLinear()

mutable struct Pool
    fields::Vector{Field}
    used::Set{Int}
    calls::Int
    fail_at::Int
end
function DO.get_temp(pool::Pool)
    pool.calls += 1
    pool.calls == pool.fail_at && error("injected scratch acquisition failure")
    index = findfirst(i -> !(i in pool.used), eachindex(pool.fields))
    index === nothing && error("scratch pool exhausted")
    push!(pool.used, index)
    return pool.fields[index], index
end
DO.unused!(pool::Pool, index) = delete!(pool.used, index)
DO.substitute_fermion!(x::Field, y::Field) = copyto!(x.data, y.data)
DO.add!(x::Field, a, y::Field) = (x.data .+= a .* y.data)
DO.add!(a, x::Field, b, y::Field) = (x.data .= a .* x.data .+ b .* y.data)

mutable struct Op
    pool::Pool
    calls::Int
    fail_at::Int
end
DO.get_temporaryvectors_forCG(A::Op) = A.pool
function LinearAlgebra.mul!(y::Field, A::Op, x::Field)
    A.calls += 1
    A.calls == A.fail_at && error("injected operator failure")
    y.data .= [1., 2., 4.] .* x.data
    return y
end

@testset "Generic CG returns scratch on every exit" begin
    for mode in (:normal, :zero, :nonconverged, :initial_apply, :iteration_apply,
                 :allocation_1, :allocation_2, :allocation_3, :allocation_4)
        fail = startswith(string(mode), "allocation_") ? parse(Int, string(last(string(mode)))) : 0
        pool = Pool([Field(zeros(3)) for _ in 1:4], Set{Int}(), 0, fail)
        A = Op(pool, 0, mode == :initial_apply ? 1 : mode == :iteration_apply ? 2 : 0)
        x, b = Field(zeros(3)), Field(mode == :zero ? zeros(3) : ones(3))
        if mode in (:normal, :zero)
            @test DO.cg(x, A, b; eps=1e-24, verbose=Verbose_print(0)) === nothing
            @test x.data ≈ b.data ./ [1., 2., 4.]
        else
            @test_throws ErrorException DO.cg(x, A, b;
                maxsteps=mode == :nonconverged ? 0 : 10, verbose=Verbose_print(0))
        end
        @test isempty(pool.used)
        # A successful solve after each failure must fit in the same four slots.
        A.fail_at = pool.fail_at = 0
        fill!(x.data, 0)
        fill!(b.data, 1)
        DO.cg(x, A, b; eps=1e-24, verbose=Verbose_print(0))
        @test x.data ≈ [1., 0.5, 0.25]
        @test isempty(pool.used)
    end
end
end
