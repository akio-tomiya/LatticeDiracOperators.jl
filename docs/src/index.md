```@meta
CurrentModule = LatticeDiracOperators
```

# LatticeDiracOperators

Documentation for [LatticeDiracOperators](https://github.com/akio-tomiya/LatticeDiracOperators.jl).

# Abstract

This is a package for lattice QCD codes.
Treating pseudo-femrion fields with various lattice Dirac operators, fermion actions with MPI.

```@raw html
<img src="./LQCDjl_block.png" width=300>
```

This package will be used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl). 

# What this package can do:
- Constructing actions and its derivative for Staggered Fermion with 1-8 tastes (with the use of the rational HMC technique)
- Constructing actions and its derivative for Wilson Fermion
- Constructing actions and its derivative for Standard Domainwall Fermion (Experimental. not well tested)
- Hybrid Monte Carlo method with fermions.

With the use of the Gaugefields.jl, we can also do the HMC with STOUT smearing. 

This package will be used in LatticeQCD.jl. 
This package uses [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl). 
This package can be regarded as the additional package of the Gaugefields.jl to treat with Lattice fermions (pseudo- fermions). 

# Install

```
add LatticeDiracOperators.jl
```