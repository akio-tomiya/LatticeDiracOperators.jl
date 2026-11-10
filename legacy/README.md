# Legacy samples

The scripts in `legacy/run/` were written for LatticeDiracOperators before v1.
They create gauge fields with `Initialize_Gaugefields` or
`Initialize_4DGaugefields`, choose CUDA or JACC through the old `cuda=true` and
`accelerator="JACC"` keywords, or drive MPI by hand with `mpi=true`. They are
kept for reference and are not maintained.

v1 keeps the old functions as a compatibility API, so some of these scripts
still run. With the registered LatticeDiracOperators 1.1.6 and Gaugefields
1.1.13 on one CPU process (checked on 2026-11-10):

- `wilsonhmc.jl` (including its Sakurai-Sugiura eigenvalue part) and
  `wilsonhmc_cold.jl` run unchanged;
- `mpiwilsontest_imp.jl` in serial mode (`1 1 1 1 false`) solves
  `D^{-1}(D x) = x` correctly, but its 16^3 x 32 lattice is slow on this path;
- `mpidomainwalltest_imp.jl` in serial mode stops at `setvalue_fermion!`,
  which has no method for the domain-wall field;
- `wilsonhmc_jacc.jl` and `staggeredhmc_jacc.jl` load CUDA.jl. Without that
  line, the JACC Wilson field stops in `dot` with JACC 1.4.
- The CUDA scripts (`wilsonhmc_cuda.jl`, `gputest/`) and multi-process MPI
  were not checked.

New code should start from the v1 samples instead: `run/wilsonhmc.jl`,
`run/staggeredhmc.jl`, `run/solvertest.jl`, the scripts in `examples/`, and the
README of this repository. The
[Migration map](https://github.com/akio-tomiya/Gaugefields.jl/blob/master/docs/src/legacyapi.md#migration-map)
on the Legacy API page of the Gaugefields documentation gives the v1
replacement for each pre-v1 gauge-field function, for example
`gauge_configuration` for `Initialize_Gaugefields` and `measure_plaquette` for
`calculate_Plaquette`. For GPUs, select the JACC backend before creating the
fields instead of passing `cuda=true`; see
[MPI, GPU, and multi-GPU](../docs/src/mpi_gpu.md).
