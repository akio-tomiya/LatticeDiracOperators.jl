# Run with an active project configured for JACC's CUDA backend.
# Device visibility and MPI transport are inherited from the caller.
import JACC
JACC.backend == "cuda" || error(
    "CUDA EO test requested, but JACC selected $(JACC.backend); configure cuda first")
JACC.@init_backend

using CUDA
using LatticeMatrices

CUDA.functional(true) || error("CUDA is not functional")
CUDA.allowscalar(false)
include(joinpath(@__DIR__, "..", "test_communicator.jl"))
device_selection = select_device_by_mpi_rank!(LDO_TEST_COMM)
@info "EO CUDA hardware" rank=ldo_test_comm_rank() selection=device_selection device=CUDA.name(CUDA.device()) uuid=string(CUDA.uuid(CUDA.device())) runtime=CUDA.runtime_version() driver=CUDA.driver_version()

withenv("LDO_TEST_REQUIRE_GPU" => "true") do
    include(joinpath(@__DIR__, "..", "domainwall_eo.jl"))
end
