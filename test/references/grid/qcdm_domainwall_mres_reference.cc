// Independent Shamir domain-wall PP/J5q reference for LatticeDiracOperators.jl.
//
// The gauge field is cold except for
// U_1(0)=diag(exp(i theta),exp(-i theta),1), so the complete input can be
// reconstructed without a binary gauge fixture. The program records the
// physical-wall pseudoscalar correlator, midpoint J5q correlator, and ratio.

#include <Grid/Grid.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Grid;

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  const Coordinate lattice({4, 4, 4, 4});
  const Coordinate mpi({1, 1, 1, 1});
  const Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  auto *grid = SpaceTimeGrid::makeFourDimGrid(lattice, simd, mpi);
  auto *rb_grid = SpaceTimeGrid::makeFourDimRedBlackGrid(grid);

  constexpr int Ls = 4;
  auto *fermion_grid = SpaceTimeGrid::makeFiveDimGrid(Ls, grid);
  auto *fermion_rb_grid =
      SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, grid);

  LatticeGaugeField gauge(grid);
  SU<Nc>::ColdConfiguration(gauge);
  LorentzColourMatrixD origin_links;
  const Coordinate origin({0, 0, 0, 0});
  peekSite(origin_links, gauge, origin);
  constexpr RealD theta = 1.1;
  origin_links(0)()(0, 0) = ComplexD(std::cos(theta), std::sin(theta));
  origin_links(0)()(1, 1) = ComplexD(std::cos(theta), -std::sin(theta));
  origin_links(0)()(2, 2) = ComplexD(1.0, 0.0);
  pokeSite(origin_links, gauge, origin);

  constexpr RealD mass = 0.1;
  constexpr RealD M5 = 1.0;
  DomainWallFermionD::ImplParams params;
  params.boundary_phases[Nd - 1] = -1.0;
  DomainWallFermionD dirac(
      gauge, *fermion_grid, *fermion_rb_grid, *grid, *rb_grid,
      mass, M5, params);

  LatticePropagator source(grid);
  LatticePropagator propagator(grid);
  LatticePropagator propagator5(fermion_grid);
  source = Zero();
  propagator = Zero();
  propagator5 = Zero();
  SpinColourMatrix unit;
  unit = 1.0;
  pokeSite(unit, source, origin);

  ConjugateGradient<LatticeFermion> cg(1.0e-14, 100000);
  SchurRedBlackDiagMooeeSolve<LatticeFermion> solve(cg);
  ZeroGuesser<LatticeFermion> guess;

  for (int spin = 0; spin < Ns; ++spin) {
    for (int colour = 0; colour < Nc; ++colour) {
      LatticeFermion source4(grid);
      LatticeFermion source5(fermion_grid);
      LatticeFermion solution4(grid);
      LatticeFermion solution5(fermion_grid);
      PropToFerm<DomainWallFermionD>(source4, source, spin, colour);
      dirac.ImportPhysicalFermionSource(source4, source5);
      solution5 = Zero();
      solve(dirac, source5, solution5, guess);
      dirac.ExportPhysicalFermionSolution(solution5, solution4);
      FermToProp<DomainWallFermionD>(propagator, solution4, spin, colour);
      FermToProp<DomainWallFermionD>(propagator5, solution5, spin, colour);
    }
  }

  LatticeComplex pp = trace(adj(propagator) * propagator);
  LatticeComplex j5q(grid);
  dirac.ContractJ5q(propagator5, j5q);
  std::vector<TComplex> pp_timeslices;
  std::vector<TComplex> j5q_timeslices;
  sliceSum(pp, pp_timeslices, Nd - 1);
  sliceSum(j5q, j5q_timeslices, Nd - 1);

  std::cout << std::setprecision(17);
  std::cout << "GRID_COMMIT "
            << "0ac72cb6a30ccdc41d664e7e0759f0c8833078f1\n";
  std::cout << "GAUGE localized theta=" << theta << "\n";
  std::cout << "PLAQUETTE "
            << WilsonLoops<PeriodicGimplR>::avgPlaquette(gauge) << "\n";
  std::cout << "SHAMIR mass=" << mass << " M5=" << M5
            << " Ls=" << Ls << "\n";
  std::cout << "BOUNDARY 1 1 1 -1\n";
  for (int t = 0; t < static_cast<int>(pp_timeslices.size()); ++t) {
    const Complex pp_value = TensorRemove(pp_timeslices[t]);
    const Complex j5q_value = TensorRemove(j5q_timeslices[t]);
    std::cout << "MRES t=" << t
              << " pp=" << pp_value
              << " j5q=" << j5q_value
              << " ratio=" << j5q_value / pp_value << "\n";
  }

  delete fermion_rb_grid;
  delete fermion_grid;
  delete rb_grid;
  delete grid;
  Grid_finalize();
  return 0;
}
