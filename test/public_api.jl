using LatticeDiracOperators
using Test

const _LDO = LatticeDiracOperators
const _DIRAC = LatticeDiracOperators.Dirac_operators

@testset "v1 public API" begin
    undefined_exports = filter(
        name -> !isdefined(_LDO, name), names(_LDO; all=false, imported=false))
    @test isempty(undefined_exports)

    @test _LDO.cg === _DIRAC.cg
    @test _LDO.WilsonFermion_4D_MPILattice ===
          _DIRAC.WilsonFermion_4D_MPILattice
    @test _LDO.StaggeredFermion_4D_MPILattice ===
          _DIRAC.StaggeredFermion_4D_MPILattice
    @test _LDO.DomainwallFermion_5D_MPILattice ===
          _DIRAC.DomainwallFermion_5D_MPILattice
    @test _LDO.solve_domainwall_physical_propagator! ===
          _DIRAC.solve_domainwall_physical_propagator!
    @test _LDO.domainwall_physical_point_propagators ===
          _DIRAC.domainwall_physical_point_propagators
    @test _LDO.domainwall_residual_mass_correlator ===
          _DIRAC.domainwall_residual_mass_correlator

    # Compatibility names remain bound after moving their implementations.
    @test _LDO.WilsonFermion_4D_wing === _DIRAC.WilsonFermion_4D_wing
    @test isdefined(_LDO, :Wilson_Dirac_operator_faster)
end
