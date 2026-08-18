# Deprecated Möbius domain-wall implementations

The historical four-dimensional-slice representation remains included for
compatibility.  New MPILattice code uses the common
`DomainwallFermion_5D_MPILattice` field and the LatticeMatrices Möbius
operator.

`MobiusDomainwallFermion_MPILattice_compat.jl` contains only conversions from
the historical `w[s]` storage to the common field. The standard field itself
is loaded before, and independently of, this directory.
