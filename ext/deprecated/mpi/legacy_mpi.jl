# Compatibility loader for the hand-written MPI fermion fields.  These files
# are intentionally loaded only after users load MPI.jl.  Supported code uses
# the LatticeMatrices-backed fields and LatticeDiracOperatorsMPIExt instead.
# Requires.jl supplies the MPI module binding while evaluating this loader.

include("Wilson/WilsonFermion_4D_wing_mpi.jl")
include("Wilson/WilsonFermion_4D_nowing_mpi.jl")
include("Staggered/StaggeredFermion_4D_nowing_mpi.jl")
include("Domainwall/DomainwallFermion_5d_wing_mpi.jl")
include("Domainwall/DomainwallFermion_5d_mpi.jl")
