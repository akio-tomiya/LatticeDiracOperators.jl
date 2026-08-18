# Actions, forces, and solvers

The high-level workflow is shared by all standard formulations:

1. create `U` with Gaugefields' `gauge_configuration`;
2. create a fermion field with `Initialize_pseudofermion_fields`;
3. construct `Dirac_operator` or `DdagD_operator`;
4. construct `FermiAction` when an action or force is needed.

## Operator application

LDO operators use the in-place `LinearAlgebra.mul!` interface:

```julia
y = similar(x)
mul!(y, D, x)
mul!(y, D', x)

DdagD = DdagD_operator(U, x, parameters)
mul!(y, DdagD, x)
```

Calling `D(Unew)` creates an operator with the same formulation and numerical
parameters for replacement gauge links. Actions use this mechanism so that
their argument `U` is authoritative.

## Solvers

```julia
diagnostics = solve_DinvX!(y, D, x)
diagnostics_dagdag = solve_DinvX!(y, DdagD, x)
```

`eps_CG`, `MaxCGstep`, `method_CG`, and `verbose_level` are stored in the
operator. Direct and adjoint Dirac solves normally use BiCG/BiCGStab;
positive `D†D` composites use CG. `SolverDiagnostics` records the iterative
residual, target, iteration limit, and convergence branch when returned by the
selected solver path.

For production convergence checks, recompute the true residual from the
returned solution rather than relying only on the recursively updated Krylov
residual.

## Pseudofermion actions

```julia
action = FermiAction(D, action_parameters)
value = evaluate_FermiAction(action, U, x)
```

Staggered and HISQ actions require `Dict("Nf" => Nf)`. Wilson and domain-wall
actions accept an empty dictionary; an `Nf` entry is harmless for the Wilson
constructor but is not used to change its action power.

To sample a pseudofermion from a Gaussian field:

```julia
noise = similar(x)
gauss_sampling_in_action!(noise, U, action)

phi = similar(x)
sample_pseudofermions!(phi, U, action, noise)
```

## Gauge forces

The allocating and preallocated forms are:

```julia
force = calc_UdSfdU(action, U, x)

force_preallocated = similar(U)
calc_UdSfdU!(force_preallocated, action, U, x)
```

The returned convention is LDO's existing
`Uμ (∂Sf/∂Uμ)†` matrix field. Gaugefields integrators perform the
traceless anti-Hermitian projection required by their momentum update.

| Action | Force route |
| --- | --- |
| Wilson | analytic |
| Wilson--clover | analytic LM link pullback |
| Staggered | analytic |
| HISQ | analytic LM thin-link pullback |
| Shamir/Möbius/generalized domain wall | analytic |
| GeneralFermion callbacks | automatic differentiation |

Enzyme is a weak dependency used by the `GeneralFermionAction` callback force
route. Standard Wilson--clover forces use the analytic LatticeMatrices
pullback.
