using LatticeDiracOperators
using Test

@eval using MPI

@testset "deprecated MPI field lifecycle" begin
    @test !MPI.Initialized()
    @test isdefined(
        LatticeDiracOperators.Dirac_operators,
        :WilsonFermion_4D_mpi,
    )

    legacy_field = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_mpi(
        2, 2, 2, 2, 2, (1, 1, 1, 1))
    @test MPI.Initialized()
    @test legacy_field.mpi
    @test legacy_field.nprocs == 1
end
