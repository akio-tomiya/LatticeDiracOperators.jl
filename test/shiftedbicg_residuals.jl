module ShiftedBiCGResidualTests
using LatticeDiracOperators, LinearAlgebra, Test
const DO = LatticeDiracOperators.Dirac_operators

@testset "Shifted BiCG true residuals for ordinary vectors" begin
    n = 12
    A = Matrix(Tridiagonal(fill(-0.3+0.1im,n-1),
        [0.4+0.01i+0.07im*i for i in 1:n],fill(0.2-0.05im,n-1)))
    b = ComplexF64[sin(i)+im*cos(2i) for i in 1:n]
    shifts = [0.1+0.2im,-0.35+0.3im,10.0+0im]
    solutions = DO.shiftedbicg_2003(shifts,A,b;eps=1e-24,verboselevel=0)
    @test length(solutions) == length(shifts)
    for (shift,x) in zip(shifts,solutions)
        shifted = A + shift*I
        @test x ≈ shifted\b rtol=1e-10 atol=1e-11
        @test norm(shifted*x-b)/norm(b) < 1e-10
    end
end
end
