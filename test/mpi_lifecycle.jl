using LatticeDiracOperators
using Test

@testset "MPI weak dependency lifecycle" begin
    @test Base.get_extension(
        LatticeDiracOperators, :LatticeDiracOperatorsMPIExt) === nothing

    @eval using MPI

    @test Base.get_extension(
        LatticeDiracOperators, :LatticeDiracOperatorsMPIExt) !== nothing
    @test !MPI.Initialized()

    process_grid = (1, 1, 1, 1)
    serial_field = WilsonFermion_4D_MPILattice(
        2, 2, 2, 2, 2;
        PEs=process_grid,
        comm=SerialCommunicator(),
    )
    @test !MPI.Initialized()
    @test serial_field.f.comm isa SerialCommunicator

    mpi_field = WilsonFermion_4D_MPILattice(
        2, 2, 2, 2, 2;
        PEs=process_grid,
    )
    @test MPI.Initialized()
    @test MPI.Comm_size(mpi_field.f.comm) == 1
    @test isdefined(
        LatticeDiracOperators.Dirac_operators,
        :WilsonFermion_4D_nowing_mpi,
    )
end
