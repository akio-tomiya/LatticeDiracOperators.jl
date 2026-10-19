# Generalized domain-wall EO: JACC and MPI tests

Run these commands from the LatticeDiracOperators repository root. The test
uses the active project's JACC backend. The shared numerical tests do not
import a vendor package directly; the optional `run_cuda.jl` entry point adds
CUDA-specific checks. LatticeMatrices v1 supplies device arrays,
portable kernel launches, rank-to-device selection, and halo transport.

## Environment

Create an isolated test environment with this checkout as the LDO dependency:

```sh
julia --project=test/eo -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

To test local Gaugefields/LatticeMatrices changes as well, `Pkg.develop` their
checkouts in the same environment. Configure JACC there, then start a fresh
Julia process. For example, the following selects the CPU backend:

```sh
backend=threads
julia --project=test/eo -e 'import JACC; JACC.set_backend(ARGS[1])' "$backend"
```

Set `backend` to the JACC backend installed on the machine to use a GPU. JACC
loads its backend with `JACC.@init_backend`. The tests use `ComplexF64`, so the
device/backend must support double precision. Selecting a backend alone does
not establish that these EO kernels have been validated on that hardware.

## One process

```sh
julia --startup-file=no --project=test/eo test/domainwall_eo.jl
```

For a GPU run, add `LDO_TEST_REQUIRE_GPU=true`. This rejects the Threads backend
and verifies that fields and the EO diagonal-inverse cache use JACC device
arrays rather than CPU arrays:

```sh
LDO_TEST_REQUIRE_GPU=true julia --startup-file=no --project=test/eo test/domainwall_eo.jl
```

For CUDA, install the optional dependency and configure JACC, then start a
fresh process with the CUDA entry point:

```sh
julia --project=test/eo -e 'using Pkg; Pkg.add("CUDA"); import JACC; JACC.set_backend("cuda")'
julia --startup-file=no --project=test/eo test/eo/run_cuda.jl
```

`run_cuda.jl` rejects a non-CUDA backend, checks `CUDA.functional(true)`,
disables scalar indexing with `CUDA.allowscalar(false)`, and enables the GPU
storage assertions. It logs the selected device, UUID, and CUDA versions after
MPI rank-to-device selection. Device visibility is inherited from the caller;
use `CUDA_VISIBLE_DEVICES` or the scheduler's assignment to select free GPUs.
The script does not select a machine-specific GPU number or Julia depot.

## MPI, with the same selected JACC backend

The launcher below uses the MPI binary selected by MPI.jl, avoiding a mismatch
between a system `mpiexec` and the MPI library loaded in Julia:

```sh
LDO_TEST_MPI=true LDO_TEST_EXPECT_RANKS=2 \
julia --startup-file=no --project=test/eo -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --project=test/eo test/domainwall_eo.jl`)'
```

Add `LDO_TEST_REQUIRE_GPU=true` to that command for MPI+GPU. Device assignment
is handled by `LatticeMatrices.select_device_by_mpi_rank!`, which uses the
node-local rank and respects schedulers exposing one device per process.
Available devices, double precision, and MPI transport support remain backend
requirements. In the current v1 implementation, automatic multi-rank Metal
device mapping is unsupported. The runner prints the array type, process
layout, and the resolved halo transport on every rank; it does not assume
device-direct MPI or that distinct physical GPUs were assigned by a scheduler.

The existing `test/runtests_mpi.jl` includes this runner in its two-rank CI
tests. It selects one timing sample by default to limit CI time.

The stricter CUDA entry point can be used with the same launcher:

```sh
LDO_TEST_MPI=true LDO_TEST_EXPECT_RANKS=2 \
julia --startup-file=no --project=test/eo -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --project=test/eo test/eo/run_cuda.jl`)'
```

If only one GPU is visible, both ranks share it. This exercises MPI with GPU
fields, but does not establish correctness of communication between distinct
physical GPUs. Check the printed `transport` to distinguish host-staged from
device-direct MPI.

## Checks and timings

The common entry point `domainwall_eo.jl` covers block inverses,
forward/adjoint Schur operators, solver cleanup, pseudofermion refresh, and
finite-difference force checks. Reference construction and numerical/halo
assertions may transfer arrays to the host; production solves, actions, and
forces use the selected backend.

Single-process runs also include `test/eo/dense_reference.jl` (44 assertions).
It assembles full operators by applying them to basis vectors on a `2^4`
lattice, with colors 1 and 3, fifth extents 1 and 2, nonuniform coefficients,
and a complex boundary phase. Host dense linear algebra independently checks
the normalized Schur operator and its adjoint, full forward/adjoint solutions,
the pseudofermion action, and gauge independence of the omitted diagonal
determinant factors. Explicit host/device transfers avoid GPU scalar indexing.
Run just this oracle in one process with:

```sh
julia --startup-file=no --project=test/eo test/eo/dense_reference.jl
```

Multi-rank runs retain the distributed-versus-serial checks below and do not
repeat the dense oracle on each rank.

It adds comparisons against an undistributed reference using exactly the same
gauge links and source. The reference runs on rank zero with an explicit serial
communicator. Spatial and temporal splits, complex/antiperiodic boundary phases,
halo widths 1 and 2, poisoned ghosts, and odd local extents are exercised. The
two-rank case has local extent 3 and global extent 6 in the split direction.

The same file checks CG, BiCG, and BiCGStab, with and without EO, for both
forward and adjoint equations. Its settings are:

| Variable | Default | Meaning |
|---|---:|---|
| `LDO_EO_BENCH_N` | 4 | Global extent in each of the four directions |
| `LDO_EO_BENCH_L5` | 4 | Local fifth-direction extent |
| `LDO_EO_BENCH_REPEATS` | 5 | Samples per solver and direction |
| `LDO_EO_BENCH_AXIS` | 1 | Physical direction split across ranks, 1–4 |

The global extent must be even and divisible by the number of ranks. Each
timing sample starts after backend synchronization and an MPI barrier, includes
the complete solve and final device synchronization, and reports the maximum
rank time. The median of those samples is printed on rank zero. Full-system
residuals and solution agreement are pass criteria; speedups are informational.

## Validation of this change

With Julia 1.11.8, Gaugefields 1.1.9, LatticeMatrices 1.2.8, JACC 1.3.1
(`threads`), and MPI.jl 0.20.26 (MPICH 5.0.1), the runner passed with one Julia
thread per process and `LDO_EO_BENCH_REPEATS=1`:

- Serial: 526 assertions passed.
- Two MPI ranks: 481 assertions on rank zero and 457 on rank one passed.
  Rank zero also checks the gathered results against the serial reference.
- Requesting a GPU while the Threads backend was selected failed as intended.

These assertion counts belong to this runner and configuration; repetitions
and rank-specific checks change the totals. They predate the 44 dense-reference
assertions now included in single-process runs.

### H100 validation (2026-09-28)

Independent validation used Julia 1.11.8, Gaugefields 1.1.9, LatticeMatrices
1.2.8, JACC 1.4.0 (`cuda`), CUDA.jl 6.4.0, MPI.jl 0.20.27, and MPICH 5.0.1.
The device was an NVIDIA H100 NVL, using `ComplexF64`, one Julia/OpenBLAS thread
per process, and `LDO_EO_BENCH_REPEATS=1`. CUDA reported runtime 13.4.0 and
driver API 13.3.0; the host NVIDIA driver was 590.48.01.

This run used commit `cd5ac29589e6e189595dc23d5326342a3c2e235e`. Its production
Julia source and package dependency declarations are identical to
`c0385c20b2646b95bb6101e7231603fc2538187b`, which consolidated the test files.
These are results for the earlier runners, not a claim that the newly arranged
CUDA entry point was run on that date:

- Single GPU: 534/534 assertions passed with GPU storage checks and scalar
  indexing disabled.
- Independent dense reference, run separately: 44/44 assertions passed.
- Two MPI ranks sharing one H100: rank zero 489/489 and rank one 465/465 passed.
  MPI selected `host_staged`; device-direct and distinct-GPU communication
  were not exercised.
- On the `4^4`, `L5=4` comparison, the maximum original-system relative residual
  of the EO GPU solutions was `4.69e-12`, and the maximum relative difference
  between EO and non-EO solutions was `7.03e-12`.
- The maximum absolute difference between analytic and finite-difference
  gauge derivatives on the single GPU was `1.25e-9`.

The full package suite, long HMC trajectories, and production performance were
not validated by these focused runs. One timing sample on these small lattices
is not a production-performance measurement.

### Integrated CPU validation (2026-10-19)

The consolidated entry point with the dense reference included passed 570/570
assertions, also exercising reuse of `test/test_communicator.jl`. This run used
Julia 1.11.8, Gaugefields 1.1.9, LatticeMatrices 1.2.8, JACC 1.4.0 (`threads`),
one Julia/OpenBLAS thread, and `LDO_EO_BENCH_REPEATS=1`.
The same common entry point passed on both CPU MPI ranks with MPI.jl 0.20.27
and MPICH 5.0.1, including distributed-versus-serial comparisons and force
finite differences; the dense reference is deliberately limited to one process.
The CUDA entry point was checked to fail explicitly with the Threads backend.
The newly arranged CUDA entry point was not rerun on GPU hardware on this date;
the H100 results above belong to the earlier validation.
