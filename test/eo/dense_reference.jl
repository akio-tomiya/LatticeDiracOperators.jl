# Standalone, backend-neutral reference test; also included by domainwall_eo.jl
# for single-process runs. Dense reference algebra is intentionally on the host.
module DomainwallEODenseReferenceTests

import JACC
JACC.@init_backend

using Gaugefields, LatticeDiracOperators, LatticeMatrices, LinearAlgebra, Test

const Dop = LatticeDiracOperators.Dirac_operators

function parameters(L5)
    return Dict{String,Any}(
        "Dirac_operator" => "GeneralizedDomainwall", "L5" => L5,
        "mass" => 0.15, "M" => -1.3,
        "as" => [0.8 + 0.2s for s in 1:L5],
        "bs" => [1.7 - 0.1s for s in 1:L5],
        "cs" => [0.4 + 0.1s for s in 1:L5],
        "evenodd" => true, "method_CG" => "bicgstab",
        "eps_CG" => 1e-22, "MaxCGstep" => 3000, "verbose_level" => 0)
end

function dense_matrix(operator, source)
    image = similar(source)
    shape = (source.NC, 4, source.NX, source.NY, source.NZ, source.NT, source.L5)
    matrix = zeros(ComplexF64, prod(shape), prod(shape))
    # Explicit host-to-device copies also work with CUDA.allowscalar(false).
    basis = zeros(eltype(source.f.A), size(source.f.A))
    for (column, index) in enumerate(CartesianIndices(shape))
        fill!(basis, 0)
        indices = Tuple(index)
        basis[indices[1], indices[2], (indices[3:end] .+ source.f.nw)...] = 1
        copyto!(source.f.A, basis)
        mark_halo_dirty!(source.f)
        mul!(image, operator, source)
        matrix[:, column] = vec(gather_matrix(image.f))
    end
    return matrix
end

function dense_schur(matrix, even)
    odd = .!even
    E, B = matrix[even, even], matrix[even, odd]
    C, O = matrix[odd, even], matrix[odd, odd]
    return E \ (E - B * (O \ C))
end

@testset "Independent dense EO reference" begin
    @testset "colors=$NC L5=$L5" for NC in (1, 3), L5 in (1, 2)
        determinant_offsets = Float64[]
        for seed in (UInt64(100), UInt64(101))
            U = gauge_configuration((2, 2, 2, 2); colors=NC, halo=1,
                start=:hot, seed, process_grid=(1, 1, 1, 1),
                comm=SerialCommunicator(), verbose=0)
            source = DomainwallFermion_5D_MPILattice(U[1], L5;
                operator_name="GeneralizedDomainwall",
                boundarycondition=[cis(0.3), 1, 1, -1, 1])
            W = Dirac_operator(U, source, parameters(L5))
            shape = (NC, 4, 2, 2, 2, 2, L5)
            even = vec([iseven(sum(Tuple(i)[3:6])) for i in CartesianIndices(shape)])

            # Assemble the full operators from basis vectors, independently of
            # EO block inverses, hopping helpers, and reconstruction routines.
            D = dense_matrix(W.D5DW, source)
            P = dense_matrix(W.D5DW_PV, source)
            S, SP = dense_schur(D, even), dense_schur(P, even)
            schur_operator = Dop.D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(W.D5DW)
            @test dense_matrix(schur_operator, source)[even, even] ≈ S rtol=3e-12
            @test dense_matrix(schur_operator', source)[even, even] ≈ S' rtol=3e-12

            gauss_distribution_fermion!(source; seed=102)
            rhs = vec(gather_matrix(source.f))
            solution = similar(source)
            for adj in (false, true)
                solve_DinvX!(solution, adj ? W.D5DW' : W.D5DW, source)
                expected = (adj ? D' : D) \ rhs
                @test vec(gather_matrix(solution.f)) ≈ expected rtol=1e-9
            end
            action = FermiAction(W, Dict())
            expected_action = norm(S' \ (SP' * rhs[even]))^2
            @test evaluate_FermiAction(action, U, source) ≈ expected_action rtol=1e-10

            # det(D) = det(E) det(O) det(S). The omitted diagonal factors may
            # depend on mass/coefficients, but must not depend on gauge links.
            push!(determinant_offsets, 2 * (logabsdet(D)[1] - logabsdet(P)[1] -
                logabsdet(S)[1] + logabsdet(SP)[1]))
        end
        @test determinant_offsets[1] ≈ determinant_offsets[2] atol=2e-10
    end
end

end # module
