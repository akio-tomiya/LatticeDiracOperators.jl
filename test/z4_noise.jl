import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LatticeMatrices: gather_and_bcast_matrix
using Test

function _test_z4_values(field)
    values = gather_and_bcast_matrix(field.f)
    roots = (1.0 + 0.0im, 0.0 + 1.0im, -1.0 + 0.0im, 0.0 - 1.0im)
    @test all(value -> value in roots, values)
    return values
end

@testset "Z4 noise" begin
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=2,
        halo=1,
        start=:cold,
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )

    for family in ("Wilson", "staggered")
        field = Initialize_pseudofermion_fields(U[1], family)
        Z4_distribution_fermi!(field; seed=0x1234, sweep=7)
        first = _test_z4_values(field)

        repeated = similar(field)
        Z4_distribution_fermi!(repeated; seed=0x1234, sweep=7)
        @test gather_and_bcast_matrix(repeated.f) == first

        Z4_distribution_fermi!(repeated; seed=0x1234, sweep=8)
        @test gather_and_bcast_matrix(repeated.f) != first
    end

    domainwall = Initialize_pseudofermion_fields(
        U[1],
        "Domainwall";
        L5=4,
    )
    Z4_distribution_fermi!(domainwall; seed=0x5678, sweep=3)
    domainwall_values = gather_and_bcast_matrix(domainwall.f)
    roots = (1.0 + 0.0im, 0.0 + 1.0im, -1.0 + 0.0im, 0.0 - 1.0im)
    @test all(value -> value in roots, selectdim(domainwall_values, 7, 1))
    @test all(iszero, selectdim(domainwall_values, 7, 2))
    @test all(iszero, selectdim(domainwall_values, 7, 3))
    @test all(iszero, selectdim(domainwall_values, 7, 4))

    U2 = Initialize_Gaugefields(
        2,
        0,
        2,
        2;
        condition="cold",
        verbose_level=0,
    )
    field2 = Initialize_pseudofermion_fields(U2[1], "Wilson")
    Z4_distribution_fermi!(field2)
    @test all(value -> value in roots, field2)
end
