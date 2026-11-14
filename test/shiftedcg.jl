module ShiftedCGTests
using Test, LinearAlgebra, LatticeDiracOperators, Gaugefields, LatticeMatrices, JACC
JACC.@init_backend
include("test_communicator.jl")
const DO = LatticeDiracOperators.Dirac_operators
const snapshots = Dict{String,Vector{Float64}}()

@testset "Shifted CG contracts and true residuals" begin
    b = ComplexF64[1, 2im, -3, 4-im]
    A = Diagonal([0.0, 0.2, 1.0, 5.0])
    shifts = [2.0, 0.001, 0.3, 0.001]
    xs = [fill(ComplexF64(NaN), length(b)) for _ in shifts]
    DO.shiftedcg(xs, shifts, A, b; eps=1e-22)
    for (x, shift) in zip(xs, shifts)
        @test norm((A + shift*I)*x - b)^2 <= 1e-22
        @test x ≈ (A + shift*I) \ b rtol=1e-12
    end
    # The five-argument API still returns the unshifted inverse in x.
    x = similar(b)
    B = A + I
    DO.shiftedcg(xs, shifts, x, B, b; eps=1e-22)
    @test x ≈ B \ b rtol=1e-12
    for (solution, shift) in zip(xs, shifts)
        @test norm((B + shift*I)*solution - b)^2 <= 1e-22
    end
    complex_shifts = ComplexF64[0.1+0.2im, -0.1im]
    complex_xs = [similar(b) for _ in complex_shifts]
    DO.shiftedcg(complex_xs, complex_shifts, x, B, b; eps=1e-22)
    @test x ≈ B \ b rtol=1e-12
    for (solution, shift) in zip(complex_xs, complex_shifts)
        @test norm((B + shift*I)*solution - b)^2 <= 1e-22
    end
    # A singular unshifted system cannot provide the promised x.
    @test_throws ErrorException DO.shiftedcg(xs, shifts, x, A, b; maxsteps=20)
    DO.shiftedcg(xs, shifts, x, A, b; seed_shift=minimum(shifts), eps=1e-22)
    @test x ≈ (A + minimum(shifts)*I) \ b rtol=1e-12
    DO.shiftedcg(xs, shifts, x, A, zero(b); seed_shift=minimum(shifts))
    @test iszero(norm(x)) && all(v -> iszero(norm(v)), xs)
    @test_throws ArgumentError DO.shiftedcg(xs, Float64[], A, b)
    @test_throws ArgumentError DO.shiftedcg(xs, [NaN], A, b)
    @test_throws ArgumentError DO.shiftedcg(xs, shifts, A, b; eps=0)
    @test_throws ErrorException DO.shiftedcg(xs, shifts, A, b; eps=1e-30, maxsteps=1)
end

@testset "Massless Wilson shifted solves" begin
    shifts = LatticeDiracOperators.Rhmc.RHMC(-1//4; n=15).coeffs.β
    grid = (1, 1, 1, ldo_test_comm_size())
    for dims in ((4,4,4,4), (4,4,4,8))
        U = gauge_configuration(dims; colors=3, halo=1, start=:cold,
            process_grid=grid, comm=LDO_TEST_COMM, verbose=0)
        b = Initialize_pseudofermion_fields(U[1], "Wilson"; boundarycondition=[1,1,1,1])
        values = ComplexF64[1 + sin(i) + im*cos(2i) for i in 1:length(b)]
        substitute_fermion!(b, values)
        parameters = Dict{String,Any}("Dirac_operator"=>"Wilson", "κ"=>1/8,
            "boundarycondition"=>[1,1,1,1], "verbose_level"=>0)
        A = DdagD_operator(U, b, parameters)
        xs = [similar(b) for _ in shifts]
        target = 1e-20 * real(dot(b,b))
        DO.shiftedcg(xs, shifts, A, b; eps=target, maxsteps=3000)
        r = similar(b)
        for (j, shift) in enumerate(shifts)
            mul!(r, A, xs[j])
            axpby!(shift, xs[j], 1, r)
            axpby!(-1, b, 1, r)
            residual = real(dot(r,r))
            @test residual <= target
            v = convert_to_normalvector(xs[j])
            snapshots["$(dims[end])-shift-$j"] = vcat(real.(v), imag.(v))
        end
        # An exactly constant spinor is a zero mode with known shifted solutions.
        substitute_fermion!(b, ones(ComplexF64, length(b)))
        DO.shiftedcg(xs, shifts, A, b; eps=1e-20 * real(dot(b,b)))
        for (solution, shift) in zip(xs, shifts)
            @test convert_to_normalvector(solution) ≈ fill(1/shift, length(b)) rtol=1e-11
        end
    end
end
end
