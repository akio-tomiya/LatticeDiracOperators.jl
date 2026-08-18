# High-level API parameters

The standard API keeps the existing string-keyed parameter dictionaries. This
page records the v1 MPILattice interpretation of those keys.

## Field initialization

```julia
x = Initialize_pseudofermion_fields(U[1], family; L5=2)
```

| `family` | Returned standard field | Notes |
| --- | --- | --- |
| `"Wilson"` | `WilsonFermion_4D_MPILattice` | also used by Wilson--clover |
| `"staggered"` | `StaggeredFermion_4D_MPILattice` | also used by HISQ |
| `"Domainwall"` | `DomainwallFermion_5D_MPILattice` | Shamir coefficients |
| `"MobiusDomainwall"` | `DomainwallFermion_5D_MPILattice` | scalar Möbius coefficients |
| `"GeneralizedDomainwall"` | `DomainwallFermion_5D_MPILattice` | slice-dependent coefficients |

`L5` is used only by domain-wall families. Geometry, halo width, process grid,
communicator, and precision are inherited from the Gaugefields v1 link.

## Common operator parameters

```julia
D = Dirac_operator(U, x, parameters)
DdagD = DdagD_operator(U, x, parameters)
```

| Key | Default | Meaning |
| --- | --- | --- |
| `"Dirac_operator"` | required | formulation selector |
| `"eps_CG"` | `1e-19` | iterative-solver tolerance; production code should normally set this explicitly |
| `"MaxCGstep"` | `3000` | maximum Krylov iterations |
| `"method_CG"` | `"bicg"` | direct Dirac solver method |
| `"verbose_level"` | `2` | solver diagnostic verbosity; use `0` for quiet runs |
| `"boundarycondition"` | `[1, 1, 1, -1]` | spatially periodic, temporally antiperiodic phases |
| `"numtempvec_CG"` | formulation dependent | number of preallocated Krylov fields |

For staggered and domain-wall fields, a supplied boundary condition must agree
with the phases stored in the fermion's LatticeMatrix. Prefer setting phases
when the field is initialized rather than changing them only in the operator
dictionary.

## Wilson and Wilson--clover

| Key | Wilson | Wilson--clover | Meaning |
| --- | --- | --- | --- |
| `"Dirac_operator"` | `"Wilson"` | `"WilsonClover"` | selector |
| `"κ"` | required | required | hopping parameter |
| `"cSW"` | unused | `1.5612` | clover coefficient |
| `"r"` | `1.0` | `1.0` | Wilson parameter; the LM fast path currently requires one |
| `"factor"` | `1` | `1` | overall application factor |

The MPILattice Wilson path always selects the LatticeMatrices wrapper.
Compatibility flags such as `"faster version"` and `"improved gpu"` should
not be used to choose a v1 backend.

## Staggered and HISQ

| Key | Staggered | HISQ | Meaning |
| --- | --- | --- | --- |
| `"Dirac_operator"` | `"staggered"` | `"HISQ"` | selector |
| `"mass"` | required | required | fermion mass |
| `"naik_epsilon"` | unused | `0.0` | species-dependent Naik correction |

HISQ currently requires `NC=3`. Operator application supports `halo=0` or
`halo>=3`; its dynamical force requires `halo>=3`.

## Domain wall

Common keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `"mass"` | required | physical mass |
| `"L5"` | required | fifth-coordinate extent; must match the field |
| `"M"` | `-1` | domain-wall height convention used by the LM operator |

Möbius parameters:

| Key | Default | Meaning |
| --- | --- | --- |
| `"b"` | `2` | scalar Möbius coefficient |
| `"c"` | `1` | scalar Möbius coefficient |

`"Domainwall"` overrides these to the Shamir preset `b=1`, `c=1`.

Generalized parameters:

| Key | Default | Meaning |
| --- | --- | --- |
| `"as"` | `ones(L5)` | left normalization by fifth slice |
| `"bs"` | `fill(1.5, L5)` | Wilson coefficient by fifth slice |
| `"cs"` | `fill(0.5, L5)` | shifted Wilson coefficient by fifth slice |

Each vector must have length `L5`.

## Action parameters

```julia
action = FermiAction(D, action_parameters)
```

| Formulation | Required action parameters |
| --- | --- |
| Wilson / Wilson--clover | none |
| Staggered / HISQ | `"Nf" => integer` |
| Domain-wall variants | none |

Staggered `Nf=4` and `Nf=8` have direct action paths. Other flavor counts use
the package's RHMC rational approximations.

## GeneralFermionAction keywords

```julia
action = GeneralFermionAction(
    U, x, apply_D!, apply_Ddag!;
    numcg=4,
    num=10,
    numg=10,
    eps_CG=1e-12,
    maxsteps=10000,
    verbose_level=2,
    numtemp=5,
)
```

`numtemp` is the number of callback work fields passed from the preallocated
pools. `num`, `numg`, and `numcg` size the fermion, gauge, and solver pools.
Increase them when a callback or nested action borrows more temporaries.
