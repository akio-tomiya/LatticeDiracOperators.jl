using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using Test

@testset "MPI-free standard fermion fields" begin
    @test Base.get_extension(
        LatticeDiracOperators, :LatticeDiracOperatorsMPIExt) === nothing
    @test Base.get_extension(Gaugefields, :GaugefieldsMPIExt) === nothing
    @test Base.get_extension(LatticeMatrices, :LatticeMatricesMPIExt) === nothing

    communicator = SerialCommunicator()
    process_grid = (1, 1, 1, 1)

    wilson = WilsonFermion_4D_MPILattice(
        2, 2, 2, 2, 2; PEs=process_grid, comm=communicator)
    staggered = StaggeredFermion_4D_MPILattice(
        2, 2, 2, 2, 2; PEs=process_grid, comm=communicator)
    domainwall = DomainwallFermion_5D_MPILattice(
        2, 2, 2, 2, 2, 2; PEs=process_grid, comm=communicator)
    general = GeneralFermion(
        2, 1, (2, 2, 2, 2), process_grid; comm0=communicator)

    for field in (wilson, staggered, domainwall, general)
        @test Gaugefields.get_myrank(field) == 0
        @test Gaugefields.get_nprocs(field) == 1
    end
    @test wilson.f.comm isa SerialCommunicator
    @test staggered.f.comm isa SerialCommunicator
    @test domainwall.f.comm isa SerialCommunicator
    @test general.field.comm isa SerialCommunicator
end
