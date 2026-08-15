import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LatticeDiracOperators MPILattice halo epochs" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    global_size = (4 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    U = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    source = Initialize_pseudofermion_fields(U[1], "Wilson")
    result = similar(source)
    ones_global = ones(ComplexF64, 3, 4, global_size...)
    ones_lattice = LatticeMatrix(
        ones_global,
        4,
        process_grid;
        nw=1,
        phases=source.f.phases,
        comm0=source.f.comm,
    )
    LatticeMatrices.substitute!(source.f, ones_lattice)
    set_halo!(source.f)

    @test !halo_is_dirty(result.f)
    mul!(result.f, 2, source.f)
    @test halo_is_dirty(result.f)

    shifted = shift_fermion(result, 1)
    @test !halo_is_dirty(result.f)
    shifted_copy = similar(result)
    substitute_fermion!(shifted_copy, shifted)
    gathered = gather_matrix(shifted_copy.f)
    if rank == 0
        @test gathered == 2 .* ones_global
    end

    long_shifted = shift_fermion(result, (2, 0, 0, 0))
    @test isopen(long_shifted)
    LatticeMatrices.release!(long_shifted)
    @test !isopen(long_shifted)

    params = Dict(
        "Dirac_operator" => "Wilson",
        "κ" => 0.1,
        "verbose_level" => 0,
    )
    D = Dirac_operator(U, source, params)
    epochs_before_dirac = halo_epochs(result.f)
    mul!(result, D, source)
    @test halo_epochs(result.f).core > epochs_before_dirac.core
    shift_fermion(result, -1)
    @test !halo_is_dirty(result.f)
    dirac_result = gather_matrix(result.f)
    if rank == 0
        @test all(isfinite, dirac_result)
    end
end
