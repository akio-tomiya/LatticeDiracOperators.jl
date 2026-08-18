# GeneralFermion in v1

`GeneralFermion` is the callback-based extension point for Dirac operators
that are not represented by one of LDO's standard fermion types.

The v1 boundary is:

- `GeneralFermion.field` uses `LatticeMatrices.LatticeMatrix` as its storage
  and MPI/accelerator backend.
- `GeneralFermionAction` receives explicit `apply_D` and `apply_Ddag`
  callbacks.
- A callback may wrap a high-level LM operator, but it does not have to.
  It may instead assemble `D` from `mul_AshiftB!`,
  `mul_shiftAshiftB!`, spin/flavor matrix products, and field additions.
- When Enzyme is loaded, `calc_UdSfdU!` differentiates the registered
  `apply_D` callback with respect to its four gauge-link arguments.
- Enzyme is an optional dependency. Forward applications of a
  `GeneralFermion` callback do not require it; the automatic force route does.

Start with
[`examples/GeneralFermion_Quickstart.jl`](../../examples/GeneralFermion_Quickstart.jl).
It defines a one-direction Hermitian `apply_D!` in a few field operations and
passes that function directly to `GeneralFermionAction`.

[`examples/GeneralFermion_AllDirections.jl`](../../examples/GeneralFermion_AllDirections.jl)
extends the same small operator to `(U1, U2, U3, U4)` by calling one hopping
helper four times. It is the simplest example whose AD force has four nonzero
components.

The advanced example is
[`examples/GeneralFermion_Shift_AD.jl`](../../examples/GeneralFermion_Shift_AD.jl).
It defines a Wilson-like operator entirely from explicit shifts, evaluates
`D' * D`, and calculates the pseudofermion force through Enzyme. Neither
example constructs `WilsonDiracOperator4D`; that operator appears only in the
regression test as an independent reference result.
