# LatticeDiracOperators

[![CI](https://github.com/akio-tomiya/LatticeDiracOperators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/LatticeDiracOperators.jl/actions/workflows/CI.yml)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://akio-tomiya.github.io/LatticeDiracOperators.jl/dev/)

LatticeDiracOperators.jl provides lattice Dirac operators, pseudofermion
actions, solvers, and fermion forces for lattice QCD. Version 1 uses
[Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl) v1 and
[LatticeMatrices.jl](https://github.com/cometscome/LatticeMatrices.jl) v1.2
as its standard backend.

Version 1.1.3 adds analytic staggered/HISQ actions and native APE,
stout/EXP, HYP, HEX, and nHYP-smeared fermion HMC; see [changes.md](changes.md).

Version 1.1.2 adds a public trajectory-state reset hook for portable HMC
restart; see [changes.md](changes.md).

Version 1.1.1 enables HISQ for SU(N); see [changes.md](changes.md).

Version 1.1.0 makes MPI.jl optional. Serial CPU and single-GPU calculations no
longer install or load MPI. See [changes.md](changes.md) for details.

Version 1.0.2 adds stout-smearing support to the pseudofermion MD driver. See [changes.md](changes.md) for details.

Version 1.0.1 adds Gaugefields MD-driver integration and corrected Z4 noise. See [changes.md](changes.md) for details.

The package supports Julia 1.11 and 1.12, threaded CPU execution, MPI domain
decomposition, and the GPU backends provided by JACC and LatticeMatrices.

## What's new in v1

Compared with the 0.6 release line, v1:

- uses the LatticeMatrices-backed MPILattice fields as the standard path;
- provides Wilson, Wilson--clover, staggered, HISQ, Shamir domain-wall,
  Möbius domain-wall, and generalized domain-wall operators;
- keeps historical wing/nowing/accelerator implementations in
  family-specific `deprecated/` directories for source compatibility;
- implements the standard HISQ and Wilson--clover forces analytically through
  LatticeMatrices pullbacks;
- provides automatic forces for user-defined callbacks through an optional
  Enzyme extension;
- supports user-defined `apply_D!` and `apply_Ddag!` callbacks through
  `GeneralFermionAction`; and
- tests the core, Enzyme, and two-rank MPI paths separately on Julia 1.11 and
  1.12.

New code should create gauge links with Gaugefields'
`gauge_configuration`. The historical `Initialize_Gaugefields` and LDO
concrete field names remain available, but select the compatibility path when
given legacy gauge fields.

## Breaking changes from v0.6

Most legacy public names remain available through the compatibility layer, but:

- v1.1 requires Julia 1.11 or later, Gaugefields 1.1, and LatticeMatrices 1.2
  or later;
- Gaugefields v1 inputs now select the LatticeMatrices-backed MPILattice path;
  code that depends on legacy wing/nowing storage or writes directly to `.A`
  must use the standard field API and halo-epoch contract;
- the standard domain-wall field uses one five-dimensional `LatticeMatrix`
  rather than the legacy `w[s]` slice representation; and
- Enzyme is optional and is no longer loaded by LDO; automatic forces for
  user-defined `GeneralFermionAction` callbacks require explicitly loading it.

## Install

In Julia package mode:

```text
pkg> add Gaugefields LatticeDiracOperators JACC
```

Enzyme is optional. Add it only for automatic differentiation of user-defined
`GeneralFermionAction` callbacks:

```text
pkg> add Enzyme
```

MPI is optional. Serial CPU and single-GPU applications do not need to install
it. For MPI execution, add MPI.jl and load it before constructing distributed
fields:

```text
using MPI
MPI.Init() # Optional: the first MPI field also initializes MPI lazily.
```

After `using MPI`, the default communicator is `MPI.COMM_WORLD`. Even with one
rank, pass `SerialCommunicator()` as `comm`/`comm0` to force the serial path.
LDO never calls `MPI.Finalize()`.

## Recommended high-level API

The common workflow is:

1. create gauge links with `gauge_configuration`;
2. create a pseudofermion field with
   `Initialize_pseudofermion_fields`;
3. construct `Dirac_operator` or `DdagD_operator`; and
4. construct `FermiAction` when an action or force is needed.

### Wilson and Wilson--clover

This complete example applies and solves the Wilson operator, evaluates the
pseudofermion action and force, and applies the Wilson--clover operator:

```julia
# README_V1_WILSON
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

U = gauge_configuration(
    (4, 4, 4, 4);
    colors=3,
    halo=1,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)

x = Initialize_pseudofermion_fields(U[1], "Wilson")
Random.seed!(101)
gauss_distribution_fermion!(x)

parameters = Dict{String,Any}(
    "Dirac_operator" => "Wilson",
    "κ" => 0.12,
    "eps_CG" => 1e-10,
    "MaxCGstep" => 3000,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)

DdagD = DdagD_operator(U, x, parameters)
solve_DinvX!(y, DdagD, x)

action = FermiAction(D, Dict("Nf" => 2))
action_value = evaluate_FermiAction(action, U, x)
force = calc_UdSfdU(action, U, x)

@assert isfinite(action_value)
@assert all(link -> isfinite(real(dot(link.U, link.U))), force)

clover_parameters = merge(
    parameters,
    Dict{String,Any}(
        "Dirac_operator" => "WilsonClover",
        "cSW" => 1.17,
    ),
)
Dclover = Dirac_operator(U, x, clover_parameters)
mul!(y, Dclover, x)
@assert isfinite(real(dot(y, y)))

clover_action = FermiAction(Dclover, Dict("Nf" => 2))
clover_force = calc_UdSfdU(clover_action, U, x)
@assert all(link -> isfinite(real(dot(link.U, link.U))), clover_force)
```

Wilson--clover application, inversion, and force evaluation use the analytic
LatticeMatrices pullback and do not require an automatic-differentiation
package.

### HMC with and without stout smearing

The following examples show both supported styles:

1. assemble the gauge update, gauge force, fermion force, Hamiltonian, and
   Metropolis step explicitly, as in the historical README; or
2. give the same gauge and fermion actions to the Gaugefields MD driver.

The shared setup below uses a small lattice so it can be run as-is. The
functions `conventional_hmc!` and `driver_hmc!` each perform one complete HMC
trajectory. The fixed `accept_uniform` makes the example reproducible; a
production application should draw one uniform random number per trajectory.

```julia
# README_V1_HMC_COMMON
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra

function wilson_hmc_setup(; stout=false)
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=2,
        halo=1,
        start=:hot,
        seed=UInt64(0x10203040),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )

    gauge_action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(gauge_action, 0.95, plaquettes)

    field = Initialize_pseudofermion_fields(U[1], "Wilson")
    parameters = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.08,
        "eps_CG" => 1e-11,
        "MaxCGstep" => 2000,
        "verbose_level" => 0,
    )
    D = Dirac_operator(U, field, parameters)

    smearing = if stout
        network = CovNeuralnet(U)
        push!(network, STOUT_Layer(["plaquette"], [0.1], U))
        network
    else
        nothing
    end
    action = FermiAction(
        D,
        Dict("Nf" => 2);
        covneuralnet=smearing,
    )
    return U, gauge_action, action, similar(field), similar(field), smearing
end

function conventional_workspace(U)
    return (
        gauge_derivative=similar(U[1]),
        gauge_product=similar(U[1]),
        raw_fermion_derivative=map(similar, U),
        smeared_derivative=map(similar, U),
        thin_derivative=map(similar, U),
        conversion=similar(U[1]),
        exponential=similar(U[1]),
        link_product=similar(U[1]),
        exponential_temps=[similar(U[1]), similar(U[1])],
        gauge_force=initialize_TA_Gaugefields(U),
        fermion_force=initialize_TA_Gaugefields(U),
    )
end

fermion_links(U, ::Nothing) = U
fermion_links(U, smearing) = first(calc_smearedU(U, smearing))

function conventional_gauge_force!(force, gauge_action, U, workspace)
    factor = -1 / U[1].NC
    for direction in eachindex(U)
        calc_dSdUμ!(
            workspace.gauge_derivative,
            gauge_action,
            direction,
            U,
        )
        mul!(
            workspace.gauge_product,
            U[direction],
            workspace.gauge_derivative,
        )
        clear_U!(force[direction])
        Traceless_antihermitian_add!(
            force[direction],
            factor,
            workspace.gauge_product,
        )
    end
    return nothing
end

function conventional_fermion_force!(
    force,
    action,
    U,
    pseudofermion,
    ::Nothing,
    workspace,
)
    calc_UdSfdU!(
        workspace.raw_fermion_derivative,
        action,
        U,
        pseudofermion,
    )
    for direction in eachindex(U)
        clear_U!(force[direction])
        Traceless_antihermitian_add!(
            force[direction],
            -1,
            workspace.raw_fermion_derivative[direction],
        )
    end
    return nothing
end

function conventional_fermion_force!(
    force,
    action,
    U,
    pseudofermion,
    smearing,
    workspace,
)
    Uout, link_history, _ = calc_smearedU(U, smearing)
    calc_UdSfdU!(
        workspace.raw_fermion_derivative,
        action,
        Uout,
        pseudofermion,
    )
    for direction in eachindex(U)
        mul!(
            workspace.smeared_derivative[direction],
            Uout[direction]',
            workspace.raw_fermion_derivative[direction],
        )
    end
    back_prop!(
        workspace.thin_derivative,
        workspace.smeared_derivative,
        smearing,
        link_history,
        U,
    )
    for direction in eachindex(U)
        mul!(
            workspace.conversion,
            U[direction],
            workspace.thin_derivative[direction],
        )
        clear_U!(force[direction])
        Traceless_antihermitian_add!(
            force[direction],
            -1,
            workspace.conversion,
        )
    end
    return nothing
end

function conventional_update_links!(U, momenta, step_size, workspace)
    for direction in eachindex(U)
        exptU!(
            workspace.exponential,
            step_size,
            momenta[direction],
            workspace.exponential_temps,
        )
        mul!(
            workspace.link_product,
            workspace.exponential,
            U[direction],
        )
        substitute_U!(U[direction], workspace.link_product)
    end
    return nothing
end

function conventional_update_momenta!(
    momenta,
    U,
    step_size,
    gauge_action,
    action,
    pseudofermion,
    smearing,
    workspace,
)
    conventional_gauge_force!(workspace.gauge_force, gauge_action, U, workspace)
    conventional_fermion_force!(
        workspace.fermion_force,
        action,
        U,
        pseudofermion,
        smearing,
        workspace,
    )
    for direction in eachindex(U)
        add_U!(
            momenta[direction],
            step_size,
            workspace.gauge_force[direction],
        )
        add_U!(
            momenta[direction],
            step_size,
            workspace.fermion_force[direction],
        )
    end
    return nothing
end

function conventional_hamiltonian(
    U,
    momenta,
    gauge_action,
    action,
    pseudofermion,
    smearing,
)
    gauge = -real(evaluate_GaugeAction(gauge_action, U)) / U[1].NC
    fermion = real(evaluate_FermiAction(
        action,
        fermion_links(U, smearing),
        pseudofermion,
    ))
    return gauge + real(momenta * momenta) / 2 + fermion
end

function conventional_hmc!(
    U,
    gauge_action,
    action,
    pseudofermion,
    noise,
    smearing;
    steps=1,
    trajectory_length=0.002,
    accept_uniform=0.5,
)
    gauss_sampling_in_action!(
        noise,
        fermion_links(U, smearing),
        action;
        seed=0x314159,
        sweep=1,
        subgroup=1,
    )
    sample_pseudofermions!(
        pseudofermion,
        fermion_links(U, smearing),
        action,
        noise,
    )
    momenta = gaussian_momenta(
        U;
        seed=UInt64(0x55667788),
        sweep=1,
    )
    old_links = map(similar, U)
    substitute_U!(old_links, U)
    workspace = conventional_workspace(U)
    initial_hamiltonian = conventional_hamiltonian(
        U,
        momenta,
        gauge_action,
        action,
        pseudofermion,
        smearing,
    )

    step_size = trajectory_length / steps
    for _ in 1:steps
        conventional_update_links!(U, momenta, step_size / 2, workspace)
        conventional_update_momenta!(
            momenta,
            U,
            step_size,
            gauge_action,
            action,
            pseudofermion,
            smearing,
            workspace,
        )
        conventional_update_links!(U, momenta, step_size / 2, workspace)
    end

    final_hamiltonian = conventional_hamiltonian(
        U,
        momenta,
        gauge_action,
        action,
        pseudofermion,
        smearing,
    )
    delta_hamiltonian = final_hamiltonian - initial_hamiltonian
    probability = exp(-max(0, delta_hamiltonian))
    accepted = accept_uniform < probability
    accepted || substitute_U!(U, old_links)
    return (; accepted, delta_hamiltonian)
end

function driver_hmc!(
    U,
    gauge_action,
    action,
    pseudofermion,
    noise;
    steps=1,
    trajectory_length=0.002,
    accept_uniform=0.5,
)
    fermion_md = PseudofermionMDAction(action, pseudofermion)
    refresh_pseudofermion!(
        fermion_md,
        U,
        noise;
        seed=0x314159,
        sweep=1,
        subgroup=1,
    )
    actions = MDActionSet(; gauge=gauge_action, fermion=fermion_md)
    driver = md_driver(
        U,
        actions;
        steps,
        trajectory_length,
        integrator=QPQ(),
    )
    momenta = gaussian_momenta(
        U;
        seed=UInt64(0x55667788),
        sweep=1,
    )
    old_links = map(similar, U)
    substitute_U!(old_links, U)
    diagnostics = md_trajectory!(U, momenta, driver)
    probability = exp(-max(0, diagnostics.delta_hamiltonian))
    accepted = accept_uniform < probability
    accepted || substitute_U!(U, old_links)
    return (; accepted, delta_hamiltonian=diagnostics.delta_hamiltonian)
end
```

#### Conventional HMC without stout smearing

Here the thin links are passed directly to the fermion action and force.

```julia
# README_V1_HMC_THIN_CONVENTIONAL
U, gauge_action, action, phi, noise, smearing = wilson_hmc_setup(stout=false)
thin_conventional = conventional_hmc!(
    U,
    gauge_action,
    action,
    phi,
    noise,
    smearing,
)
@assert isfinite(thin_conventional.delta_hamiltonian)
```

#### MD-driver HMC without stout smearing

The physical ingredients are unchanged; `PseudofermionMDAction` supplies the
fermion potential and force to `MDActionSet`.

```julia
# README_V1_HMC_THIN_DRIVER
U, gauge_action, action, phi, noise, _ = wilson_hmc_setup(stout=false)
thin_driver = driver_hmc!(U, gauge_action, action, phi, noise)
@assert thin_driver.accepted == thin_conventional.accepted
@assert isapprox(
    thin_driver.delta_hamiltonian,
    thin_conventional.delta_hamiltonian;
    rtol=2e-10,
    atol=2e-11,
)
```

#### Conventional HMC with stout smearing

This is the historical route. The fermion action is evaluated on `Uout`, and
the derivative is returned to the thin links by `back_prop!` before the
momentum update.

```julia
# README_V1_HMC_STOUT_CONVENTIONAL
U, gauge_action, action, phi, noise, smearing = wilson_hmc_setup(stout=true)
stout_conventional = conventional_hmc!(
    U,
    gauge_action,
    action,
    phi,
    noise,
    smearing,
)
@assert isfinite(stout_conventional.delta_hamiltonian)
```

#### MD-driver HMC with stout smearing

No smearing calls are needed in the application loop. The two-argument
`PseudofermionMDAction` constructor reads the smearing stored in
`FermiAction`; the driver performs the same forward smearing and thin-link
pullback internally.

```julia
# README_V1_HMC_STOUT_DRIVER
U, gauge_action, action, phi, noise, _ = wilson_hmc_setup(stout=true)
stout_driver = driver_hmc!(U, gauge_action, action, phi, noise)
@assert stout_driver.accepted == stout_conventional.accepted
@assert isapprox(
    stout_driver.delta_hamiltonian,
    stout_conventional.delta_hamiltonian;
    rtol=2e-9,
    atol=2e-10,
)
```

### HISQ

HISQ uses the staggered MPILattice field and requires a halo width of at least
three. Its standard fermion force is an analytic LatticeMatrices pullback:

```julia
# README_V1_HISQ
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

U = gauge_configuration(
    (4, 4, 4, 4);
    colors=3,
    halo=3,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)

x = Initialize_pseudofermion_fields(U[1], "staggered")
Random.seed!(102)
gauss_distribution_fermion!(x)

parameters = Dict{String,Any}(
    "Dirac_operator" => "HISQ",
    "mass" => 0.4,
    "naik_epsilon" => -0.083,
    "eps_CG" => 1e-8,
    "MaxCGstep" => 2000,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)

action = FermiAction(D, Dict("Nf" => 4))
action_value = evaluate_FermiAction(action, U, x)
force = calc_UdSfdU(action, U, x)

@assert isfinite(action_value)
@assert all(link -> isfinite(real(dot(link.U, link.U))), force)
```

The one-link staggered operator uses the same field with
`"Dirac_operator" => "staggered"` and a `"mass"` parameter.

### Domain-wall fermions

Shamir, Möbius, and generalized domain-wall operators share one
`DomainwallFermion_5D_MPILattice` storage type backed by a
`LatticeMatrix{5}`:

```julia
# README_V1_DOMAIN_WALL
import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

U = gauge_configuration(
    (2, 2, 2, 2);
    colors=3,
    halo=1,
    start=:cold,
    process_grid=(1, 1, 1, 1),
)

L5 = 2
x = Initialize_pseudofermion_fields(U[1], "Domainwall"; L5)
Random.seed!(103)
gauss_distribution_fermion!(x)

parameters = Dict{String,Any}(
    "Dirac_operator" => "Domainwall",
    "mass" => 0.1,
    "L5" => L5,
    "M" => -1.0,
    "eps_CG" => 1e-8,
    "MaxCGstep" => 1000,
    "verbose_level" => 0,
)

D = Dirac_operator(U, x, parameters)
y = similar(x)
mul!(y, D, x)

action = FermiAction(D, Dict())
action_value = evaluate_FermiAction(action, U, x)

@assert x isa DomainwallFermion_5D_MPILattice
@assert isfinite(real(dot(y, y)))
@assert isfinite(action_value)
```

Use `"MobiusDomainwall"` with scalar `b` and `c`, or
`"GeneralizedDomainwall"` with fifth-coordinate vectors `as`, `bs`,
and `cs`.

## Standard operators

| `Dirac_operator` value | Standard field | Force implementation |
|---|---|---|
| `"Wilson"` | `WilsonFermion_4D_MPILattice` | analytic |
| `"WilsonClover"` | `WilsonFermion_4D_MPILattice` | analytic |
| `"staggered"` | `StaggeredFermion_4D_MPILattice` | analytic |
| `"HISQ"` | `StaggeredFermion_4D_MPILattice` | analytic |
| `"Domainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |
| `"MobiusDomainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |
| `"GeneralizedDomainwall"` | `DomainwallFermion_5D_MPILattice` | analytic |

All standard fields store their data in LatticeMatrices. The MPILattice name
also covers a single process: use `process_grid=(1, 1, 1, 1)` for a normal
single-process run.

## User-defined Dirac operators

`GeneralFermionAction` accepts user-defined `apply_D!` and
`apply_Ddag!` callbacks. The examples progress from a minimal definition to
a four-direction stencil:

- [Quick start: define and use `apply_D!`](examples/GeneralFermion_Quickstart.jl)
- [Shift-based operator with automatic differentiation](examples/GeneralFermion_Shift_AD.jl)
- [Operator using `U1`, `U2`, `U3`, and `U4`](examples/GeneralFermion_AllDirections.jl)

The callback force requires Enzyme. Built-in standard operators should use
`Dirac_operator` and `FermiAction` directly unless a custom composition is
needed.

## MPI and GPU execution

The field and operator APIs above do not change with the execution backend.
Choose the JACC backend before constructing fields, and choose the domain
decomposition through `process_grid`. For example, a two-rank decomposition
along the first direction uses `process_grid=(2, 1, 1, 1)`.

Gauge initialization, rank-to-device mapping, and backend-specific setup are
documented by Gaugefields and LatticeMatrices:

- [Gaugefields four-dimensional and multi-GPU tutorial](https://github.com/akio-tomiya/Gaugefields.jl/blob/main/docs/src/tutorial4d.md)
- [Gaugefields randomness and reproducibility guide](https://github.com/akio-tomiya/Gaugefields.jl/blob/main/docs/src/randomness.md)

## Documentation

- [Documenter manual](https://akio-tomiya.github.io/LatticeDiracOperators.jl/dev/)
- [Quick start](docs/src/quickstart.md)
- [Wilson and Wilson--clover](docs/src/wilson.md)
- [Staggered and HISQ](docs/src/staggered_hisq.md)
- [Domain-wall fermions](docs/src/domainwall.md)
- [User-defined operators](docs/src/generalfermion.md)
- [Actions, forces, and solvers](docs/src/actions_forces.md)
- [MPI, GPU, and multi-GPU](docs/src/mpi_gpu.md)
- [High-level API parameters](docs/src/highlevelapi.md)
- [Public v1 API index](docs/src/publicapi.md)
- [Citing LDO](docs/src/references.md)
- [v1 API and compatibility boundary](docs/src/v1_api.md)
- [Historical API and HMC examples](docs/src/howtouse.md)
- [Wilson implementation notes](src/WilsonFermion/README.md)
- [Staggered and HISQ implementation notes](src/StaggeredFermion/README.md)
- [Domain-wall implementation notes](src/DomainwallFermion/README.md)
- [GeneralFermion callback guide](src/GeneralFermion/README.md)
- [v1 implementation audit](V1_AUDIT.md)

The compatibility files under `deprecated/` are still included in v1, so
historical function and concrete type names remain callable. They are not the
backend contract for new code.

## Questions

Please use this repository's issue tracker or the
[JuliaQCD discussion board](https://github.com/orgs/JuliaQCD/discussions).
Questions in Japanese are welcome.

## Citing LatticeDiracOperators

If this package contributes to a publication, please cite both papers:

1. Yuki Nagai and Akio Tomiya,
   [*JuliaQCD: Portable lattice QCD package in Julia language*](https://arxiv.org/abs/2409.03030),
   arXiv:2409.03030 (`hep-lat`, 2024).
2. Yuki Nagai, Akio Tomiya, and Hiroshi Ohno,
   [*Lattice Gauge Theory via LLVM-Level Automatic Differentiation*](https://arxiv.org/abs/2602.20516),
   arXiv:2602.20516 (`hep-lat`, 2026).

```bibtex
@article{Nagai:2024yaf,
    author = "Nagai, Yuki and Tomiya, Akio",
    title = "{JuliaQCD: Portable lattice QCD package in Julia language}",
    eprint = "2409.03030",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    year = "2024"
}

@article{Nagai:2026llvmad,
    author = "Nagai, Yuki and Tomiya, Akio and Ohno, Hiroshi",
    title = "{Lattice Gauge Theory via LLVM-Level Automatic Differentiation}",
    eprint = "2602.20516",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    year = "2026"
}
```
