using LatticeDiracOperators
using Test

import LatticeDiracOperators.Dirac_operators: apply_γ5!
import LatticeDiracOperators.Dirac_operators: WilsonFermion_4D_wing

function test_apply_gamma5_wilson4d()
    x = WilsonFermion_4D_wing{3}(1,1,1,1)
    original = similar(x.f)

    for index in CartesianIndices(x.f)
        value = complex(Float64(index[1] + 10index[2] + 100index[3] +
                                1000index[4] + 10000index[5] + 100000index[6]),
                        Float64(index[1] - index[6]))
        x.f[index] = value
        original[index] = value
    end

    apply_γ5!(x)

    for index in CartesianIndices(x.f)
        expected_sign = index[6] <= 2 ? -1 : 1
        @test x.f[index] == expected_sign * original[index]
    end

    apply_γ5!(x)
    @test x.f == original
end

@testset "Wilson gamma5" begin
    test_apply_gamma5_wilson4d()
end
