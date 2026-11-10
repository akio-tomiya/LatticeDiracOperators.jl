import JACC
JACC.@init_backend

using Gaugefields, LatticeDiracOperators, Test, LinearAlgebra
import LatticeMatrices

function test_wilson_sources(; process_grid=(1, 1, 1, 1))
    z2_values = nothing
    @testset "Wilson source input" begin
        for T in (ComplexF64, ComplexF32)
            U = gauge_configuration((4, 4, 4, 4); colors=3, start=:cold,
                                    eltype=T, process_grid, verbose=0)
            field = Initialize_pseudofermion_fields(U[1], "Wilson")
            T = eltype(field.f.A)
            values = zeros(T, 3, 4, 4, 4, 4, 4) # color, x, y, z, t, spin
            clear_fermion!(field)
            setvalue_fermion!(field, 2+3im, 2, 1, 2, 3, 4, 3)
            values[2, 1, 2, 3, 4, 3] = 2+3im
            @test LatticeMatrices.halo_is_dirty(field.f)
            set_wing_fermion!(field)
            @test convert_to_normalvector(field) == vec(values)
            @test dot(field, field) == 13
            @test !LatticeMatrices.halo_is_dirty(field.f)
            field[1, 3, 4, 1, 2, 2] = 4-1im
            values[1, 3, 4, 1, 2, 2] = 4-1im
            set_wing_fermion!(field)
            @test convert_to_normalvector(field) == vec(values)
            @test dot(field, field) == 30
            @test_throws BoundsError setvalue_fermion!(field, 1, 0, 1, 1, 1, 1, 1)
            @test_throws BoundsError setvalue_fermion!(field, 1, 1, 1, 1, 1, 5, 1)
            @test_throws BoundsError setvalue_fermion!(field, 1, 1, 1, 1, 1, 1, 5)
            @test_throws DimensionMismatch substitute_fermion!(field, zeros(T, 3))
            # Distinguish spin and space dimensions to catch ordering mistakes.
            values .= reshape(T.(1:length(values)), size(values)) .* T(1+2im)
            substitute_fermion!(field, vec(values))
            @test convert_to_normalvector(field) == vec(values)
            @test eltype(convert_to_normalvector(field)) == T
            @test !LatticeMatrices.halo_is_dirty(field.f)
            @test dot(field, field) ≈ sum(abs2, values)
            # Compare an actual Wilson stencil with a collectively loaded point
            # source, so halo refresh and temporal boundary phases are exercised.
            reference = similar(field)
            clear_fermion!(field)
            fill!(values, 0); values[2, 1, 2, 3, 4, 3] = 1
            setvalue_fermion!(field, 1, 2, 1, 2, 3, 4, 3)
            set_wing_fermion!(field)
            substitute_fermion!(reference, vec(values))
            D = Dirac_operator(U, field, Dict("Dirac_operator"=>"Wilson", "κ"=>0.12))
            out, expected = similar(field), similar(field)
            mul!(out, D, field); mul!(expected, D, reference)
            @test convert_to_normalvector(out) == convert_to_normalvector(expected)
            @test any(!iszero, convert_to_normalvector(out))
            Z2_distribution_fermion!(field; seed=0x1234, sweep=7)
            first = convert_to_normalvector(field)
            @test all(x -> x == 1 || x == -1, first)
            @test any(==(1), first) && any(==(-1), first)
            @test dot(field, field) == length(field)
            @test !LatticeMatrices.halo_is_dirty(field.f)
            Z2_distribution_fermion!(field; seed=0x1234, sweep=7)
            @test convert_to_normalvector(field) == first
            Z2_distribution_fermion!(field; seed=0x1234, sweep=8)
            @test convert_to_normalvector(field) != first
            Z2_distribution_fermion!(field)
            @test all(x -> x == 1 || x == -1, convert_to_normalvector(field))
            if z2_values === nothing
                z2_values = first
            else
                @test z2_values == first
            end
        end
    end
    return z2_values
end

function test_legacy_fermion_fixes()
    @testset "Legacy JACC Wilson dot" begin
        U = Initialize_Gaugefields(3, 0, 4, 4, 4, 4; condition="cold", accelerator="JACC")
        a = Initialize_pseudofermion_fields(U[1], "Wilson")
        b = similar(a)
        host_a = reshape(ComplexF64.(1:length(a.f)), size(a.f)) .* (1+2im)
        host_b = reverse(host_a) .* (2-3im)
        copyto!(a.f, host_a); copyto!(b.f, host_b)
        @test dot(a, b) ≈ dot(host_a, host_b)
        @test dot(b, a) ≈ conj(dot(a, b))
        @test dot(a, a) ≈ sum(abs2, host_a)
        fill!(a.f, 0)
        @test dot(a, b) == 0
    end
    @testset "Legacy domain-wall setter" begin
        U = Initialize_Gaugefields(3, 0, 4, 4, 4, 4; condition="cold")
        for nowing in (true, false)
            field = Initialize_pseudofermion_fields(U[1], "Domainwall"; L5=4, nowing)
            clear_fermion!(field)
            setvalue_fermion!(field, 2+3im, 2, 1, 2, 3, 4, 3, 2)
            set_wing_fermion!(field)
            @test field[2, 1, 2, 3, 4, 3, 2] == 2+3im
            @test dot(field, field) == 13
            @test_throws BoundsError setvalue_fermion!(field, 1, 1, 1, 1, 1, 1, 1, 5)
        end
    end
end
