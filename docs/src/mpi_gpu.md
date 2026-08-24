# MPI, GPU, and multi-GPU execution

The v1 standard fields always use LatticeMatrices storage. The same LDO code
therefore runs on CPU threads, one GPU, MPI ranks, or multiple GPUs. JACC
selects the execution backend, while Gaugefields' `process_grid` selects the
domain decomposition.

## Select a JACC backend

Record the desired backend in the active Julia environment, then restart
Julia:

```julia
import JACC

JACC.set_backend("threads") # CPU
# JACC.set_backend("cuda")   # NVIDIA GPU
# JACC.set_backend("amdgpu") # AMD GPU
# JACC.set_backend("oneapi") # Intel GPU
```

Every simulation script should initialize JACC before loading Gaugefields and
LDO:

```julia
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
```

No CUDA-, AMDGPU-, or accelerator-specific fermion constructor is needed.

## MPI decomposition

For two ranks split in the first lattice direction:

```julia
using MPI
# MPI.Init() # Optional; field construction initializes MPI lazily.

U = gauge_configuration(
    (8, 4, 4, 4);
    colors=3,
    halo=1,
    start=:cold,
    process_grid=(2, 1, 1, 1),
)

x = Initialize_pseudofermion_fields(U[1], "Wilson")
```

Launch the script with the MPI launcher configured by MPI.jl, for example:

```text
mpiexec -n 2 julia --project=. simulation.jl
```

The fermion field inherits `process_grid`, communicator, precision, global
lattice size, and halo width from `U[1]`. `process_grid` must multiply to the
communicator size and each global extent must be divisible by its process-grid
extent.

MPI.jl is not installed or loaded for serial CPU and single-GPU execution.
Loading MPI.jl activates `MPI.COMM_WORLD`; LDO initializes it when the first
MPI-backed field is constructed if necessary, and never finalizes it. Call
`MPI.Init(...)` yourself before field construction when custom initialization
options are required. With one rank, pass `SerialCommunicator()` explicitly to
`comm` or `comm0` to select the serial path after `using MPI`.

## Halo requirements

- Wilson, Wilson--clover, staggered, and domain-wall examples normally use
  `halo=1`.
- HISQ dynamical force evaluation requires `halo>=3`.
- `halo=0` is a supported serial/fallback staggered and HISQ operator path,
  but not the standard distributed force setup.
- The domain-wall fifth coordinate is not MPI-distributed; its process-grid
  extent is one.

Gauge-link initialization, device-to-rank mapping, reproducible random fields,
and backend installation are Gaugefields/LatticeMatrices responsibilities.
See the Gaugefields v1
[four-dimensional tutorial](https://akio-tomiya.github.io/Gaugefields.jl/v1/tutorial4d/)
and [randomness guide](https://akio-tomiya.github.io/Gaugefields.jl/v1/randomness/).

## Compatibility types

Names containing `wing`, `nowing`, or the historical hand-written MPI and
accelerator implementations remain available for compatibility. They are not
alternate v1 backend choices. New CPU/GPU/MPI work should target the standard
MPILattice fields and the LatticeMatrices/JACC execution path. Legacy MPI
implementations load only after `using MPI` and are scheduled for removal in a
future breaking release.
