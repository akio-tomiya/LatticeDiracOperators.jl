using LatticeDiracOperators
using Test

import LatticeDiracOperators.Dirac_operators:
    WilsonFermion_2D_wing, mul_γ5x!, apply_γ5!

@testset "2D Wilson gamma5" begin
    for NC in (2, 3)
        # Use a rectangular lattice and check the halo entries as well.
        x = WilsonFermion_2D_wing(NC, 2, 3)
        for index in CartesianIndices(x.f)
            x.f[index] = complex(
                index[1] + 10index[2] + 100index[3] + 1000index[4],
                index[1] - index[4],
            )
        end
        original = copy(x.f)
        y = similar(x)

        mul_γ5x!(y, x)
        @test x.f == original
        for index in CartesianIndices(x.f)
            expected_sign = index[4] == 2 ? -1 : 1
            @test y.f[index] == expected_sign * original[index]
        end

        apply_γ5!(x)
        for index in CartesianIndices(x.f)
            expected_sign = index[4] == 2 ? -1 : 1
            @test x.f[index] == expected_sign * original[index]
        end
        apply_γ5!(x)
        @test x.f == original
    end
end
