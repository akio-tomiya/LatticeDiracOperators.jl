# LatticeDiracOperators

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cometscome.github.io/LatticeDiracOperators.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cometscome.github.io/LatticeDiracOperators.jl/dev)
[![Build Status](https://github.com/cometscome/LatticeDiracOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/LatticeDiracOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)

# This is the package for Lattice QCD codes. 

This is used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl)

# What this package can do:
- Constructing actions and its derivative for Staggered Fermion with 1-8 flavors (with the use of the rational HMC technique)
- Constructing actions and its derivative for Wilson Fermion
- Hybrid Monte Carlo method with fermions.

With the use of the Gaugefields.jl, we can also do the HMC with STOUT smearing. 

This package will be used in LatticeQCD.jl. 
This package uses [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl). 
This package can be regarded as the additional package of the Gaugefields.jl to treat with Lattice fermions (pseudo- fermions). 

# Install

```
add https://github.com/akio-tomiya/Wilsonloop.jl
add https://github.com/akio-tomiya/Gaugefields.jl
add https://github.com/akio-tomiya/LatticeDiracOperators.jl
```

# How to use

## Definition of the pseudo-fermion fields

The pseudo-fermin field is defined as 

```julia
using Gaugefields
using LatticeDiracOperators

NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
Dim = 4
NC = 3

U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
x = Initialize_pseudofermion_fields(U[1],"Wilson")
```

Now, x is a pseudo fermion fields for Wilson Dirac operator. 
The element of x is ```x[ic,ix,iy,iz,it,ialpha]```. ic is an index of the color. ialpha is the internal degree of the gamma matrix. 

Then, the Wilson Dirac operator can be defined as 

```julia
params = Dict()
params["Dirac_operator"] = "Wilson"
params["κ"] = 0.141139
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2

D = Dirac_operator(U,x,params)
```

If you want to get the Gaussiann distributed pseudo-fermions, just do

```julia
gauss_distribution_fermion!(x)
```

Then, you can apply the Dirac operator to the pseudo-fermion fields. 

```julia
using LinearAlgebra
y = similar(x)
mul!(y,D,x)
```

And you can solve the equation $D x = b$ like

```julia
solve_DinvX!(y,D,x)
println(y[1,1,1,1,1,1])
```
If you want to see the convergence of the CG method, you can change the "verbose_level" in the Dirac operator. 

```julia
params["verbose_level"] = 3
D = Dirac_operator(U,x,params)
gauss_distribution_fermion!(x)
solve_DinvX!(y,D,x)
println(y[1,1,1,1,1,1])
```

The output is like 

```
bicg method
1-th eps: 1742.5253056262081
2-th eps: 758.2899742222573
3-th eps: 378.7020470573924
4-th eps: 210.17029515182503
5-th eps: 118.00493128655506
6-th eps: 63.31719669150997
7-th eps: 36.18603541453448
8-th eps: 21.593691953496077
9-th eps: 16.02895509383768
10-th eps: 12.920647360667004
11-th eps: 9.532250164198402
12-th eps: 5.708202470516758
13-th eps: 3.1711913019834337
14-th eps: 0.9672090407947617
15-th eps: 0.14579004932559966
16-th eps: 0.02467506197970277
17-th eps: 0.005588563782732157
18-th eps: 0.002285284357387675
19-th eps: 5.147142014626153e-5
20-th eps: 3.5632092739322066e-10
Converged at 20-th step. eps: 3.5632092739322066e-10
```

## Other operators
You can use the adjoint of the Dirac operator 

```julia
gauss_distribution_fermion!(x)
solve_DinvX!(y,D',x)
println(y[1,1,1,1,1,1])
```

You can define the ```D^{\dagger} D``` operator. 

```julia
DdagD = DdagD_operator(U,x,params)
gauss_distribution_fermion!(x)
solve_DinvX!(y,DdagD,x) 
println(y[1,1,1,1,1,1])
```

# Staggared Fermions
The Dirac operator of the staggered fermions is defined as 

```julia
x = Initialize_pseudofermion_fields(U[1],"staggered")
gauss_distribution_fermion!(x)
params = Dict()
params["Dirac_operator"] = "staggered"
params["mass"] = 0.1
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2
D = Dirac_operator(U,x,params)

y = similar(x)
mul!(y,D,x)
println(y[1,1,1,1,1,1])

solve_DinvX!(y,D,x)
println(y[1,1,1,1,1,1])
```

The "tastes" of the Staggered Fermmion is defined in the action. 


# Fermion Action

## Wilson Fermion

The action for pseudo-fermion is defined as 

```julia

NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
Dim = 4
NC = 3

U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
x = Initialize_pseudofermion_fields(U[1],"Wilson")
gauss_distribution_fermion!(x)

params = Dict()
params["Dirac_operator"] = "Wilson"
params["κ"] = 0.141139
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2

D = Dirac_operator(U,x,params)

parameters_action = Dict()
fermi_action = FermiAction(D,parameters_action)


```

The fermion action with given pseudo-fermion fields is evaluated as 

```julia
Sfnew = evaluate_FermiAction(fermi_action,U,x)
println(Sfnew)
```

The derivative of the fermion action dSf/dU can be calculated as 

```julia
UdSfdUμ = calc_UdSfdU(fermi_action,U,x)
```
The function calc_UdSfdU calculates the ```U dSf/dU```,
You can also use ```calc_UdSfdU!(UdSfdUμ,fermi_action,U,x)```


## Staggered Fermion
In the case of the Staggered fermion, we can choose "taste". 
The action is defined as 

```julia
x = Initialize_pseudofermion_fields(U[1],"staggered")
gauss_distribution_fermion!(x)
params = Dict()
params["Dirac_operator"] = "staggered"
params["mass"] = 0.1
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2
D = Dirac_operator(U,x,params)

Nf = 2

println("Nf = $Nf")
parameters_action = Dict()
parameters_action["Nf"] = Nf
fermi_action = FermiAction(D,parameters_action)

Sfnew = evaluate_FermiAction(fermi_action,U,x)
println(Sfnew)

UdSfdUμ = calc_UdSfdU(fermi_action,U,x)
```

This package uses the RHMC techniques. 

# Hybrid Monte Carlo with fermions

## Wilson Fermion
We show the HMC code with this package. 

```julia
using Gaugefields
using LinearAlgebra
using InteractiveUtils
using Random

function MDtest!(gauge_action,U,Dim,fermi_action,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 10
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    Random.seed!(123)

    numtrj = 10
    for itrj = 1:numtrj
        @time accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
    Δτ = 1/MDsteps
    NC,_,NN... = size(U[1])
    
    gauss_distribution!(p)
    
    substitute_U!(Uold,U)
    gauss_sampling_in_action!(ξ,U,fermi_action)
    sample_pseudofermions!(η,U,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println("Sfold = $Sfold")

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,fermi_action,η)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Sfnew = evaluate_FermiAction(fermi_action,U,η)
    println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")

    accept = exp(Sold - Snew) >= rand()

    #ratio = min(1,exp(Snew-Sold))
    if accept != true #rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end

function P_update_fermion!(U,p,ϵ,Δτ,Dim,gauge_action,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    calc_UdSfdU!(UdSfdUμ,fermi_action,U,η)

    for μ=1:Dim
        Traceless_antihermitian_add!(p[μ],factor,UdSfdUμ[μ])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end
end

function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    x = Initialize_pseudofermion_fields(U[1],"Wilson")


    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.141139
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)


    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)

    y = similar(x)

    
    MDtest!(gauge_action,U,Dim,fermi_action,x,y)

end


test1()
```

## Staggered Fermion
if you want to use the Staggered fermions in HMC, the code is like: 

```julia
function test2()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    x = Initialize_pseudofermion_fields(U[1],"staggered")
    gauss_distribution_fermion!(x)
    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = 0.1
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)
    
    Nf = 2
    
    println("Nf = $Nf")
    parameters_action = Dict()
    parameters_action["Nf"] = Nf
    fermi_action = FermiAction(D,parameters_action)

    y = similar(x)

    
    MDtest!(gauge_action,U,Dim,fermi_action,x,y)

end

```