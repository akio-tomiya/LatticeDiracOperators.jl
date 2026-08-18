# Deprecated Wilson fermion implementations

This directory contains the Wilson fermion field and operator implementations
that are not part of the standard `LatticeMatrices.jl`-backed path. They remain
loaded to preserve the existing public types, constructors, and function
behavior.

The v1 standard implementation is
`../WilsonFermion_4D_MPILattice.jl`, together with the
`../WilsonFermion_improved.jl` wrapper around
`LatticeMatrices.WilsonDiracOperator4D`.
