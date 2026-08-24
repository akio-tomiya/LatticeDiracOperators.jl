module LatticeDiracOperatorsMPIExt

using LatticeDiracOperators
using MPI

import LatticeDiracOperators.Communication:
    barrier,
    broadcast!,
    comm_rank,
    comm_size,
    prepare_communicator

default_communicator() = MPI.COMM_WORLD

@inline function prepare_communicator(comm::MPI.Comm)
    MPI.Finalized() && throw(ArgumentError(
        "MPI has already been finalized; restart Julia before constructing " *
        "an MPI fermion field"))
    MPI.Initialized() || MPI.Init()
    return comm
end

@inline comm_size(comm::MPI.Comm) = MPI.Comm_size(comm)
@inline comm_rank(comm::MPI.Comm) = MPI.Comm_rank(comm)
@inline barrier(comm::MPI.Comm) = MPI.Barrier(comm)
@inline broadcast!(value, root::Integer, comm::MPI.Comm) =
    MPI.Bcast!(value, root, comm)
end
