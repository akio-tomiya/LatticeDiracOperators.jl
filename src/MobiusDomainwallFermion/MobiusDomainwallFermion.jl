# Load the v1 standard field independently of the compatibility layer.  The
# legacy file that follows defines the historical w[s] fields and the shared
# public high-level wrappers.
include("../DomainwallFermion/DomainwallFermion_5D_MPILattice.jl")
include("./deprecated/MobiusDomainwallFermion_legacy.jl")
