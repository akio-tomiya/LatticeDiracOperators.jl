# LatticeDiracOperators


[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://akio-tomiya.github.io/LatticeDiracOperators.jl/dev)
[![CI](https://github.com/akio-tomiya/LatticeDiracOperators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/LatticeDiracOperators.jl/actions/workflows/CI.yml)

# Abstract

This is a package for lattice QCD codes.
Treating pseudo-femrion fields with various lattice Dirac operators, fermion actions with MPI.

<img src="LQCDjl_block.png" width=300> 

This package is used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl)
and a code in a project [JuliaQCD](https://github.com/JuliaQCD/).


## Q&A/Issues
If you have questions and comments. Please use the issues section of this repository or use [Discussions in JuliaQCD](https://github.com/orgs/JuliaQCD/discussions/5).

[In Japanese] 質問やコメントを日本語でしたい方は[JuliaQCDのディスカッションボード](https://github.com/orgs/JuliaQCD/discussions/6)に書き込みをしてください。

# What this package can do:
- Constructing actions and its derivative for Staggered Fermion with 1-8 tastes (with the use of the rational HMC technique)
- Constructing actions and its derivative for Wilson Fermion
- (EXPERIMENTAL) Constructing actions and its derivative for Standard Domainwall Fermion 
- Hybrid Monte Carlo method with fermions.

With the use of the Gaugefields.jl, we can also do the HMC with STOUT smearing. 

This package will be used in LatticeQCD.jl. 
This package uses [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl). 
This package can be regarded as the additional package of the Gaugefields.jl to treat with Lattice fermions (pseudo- fermions). 

# Install

In the package mode, Julia REPL,
```
add LatticeDiracOperators
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

If you want to get the Gaussian distributed pseudo-fermions, just do

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

The "tastes" of the Staggered Fermion is defined in the action. 

# (EXPERIMENTAL) Domainwall Fermions
This package supports standard domainwall fermions. 
The Dirac operator of the domainwall fermion is defined as 

```julia
L5 = 4
x = Initialize_pseudofermion_fields(U[1],"Domainwall",L5=L5)
println("x ", x.w[1][1,1,1,1,1,1])
gauss_distribution_fermion!(x)

params = Dict()
params["Dirac_operator"] = "Domainwall"
params["eps_CG"] = 1.0e-16
params["MaxCGstep"] = 3000
params["verbose_level"] = 3
params["mass"] = 0.1
params["L5"] = L5
D = Dirac_operator(U,x,params)

println("x ", x[1,1,1,1,1,1,1])
y = similar(x)
solve_DinvX!(y,D,x)
println("y ", y[1,1,1,1,1,1,1])

z = similar(x)
mul!(z,D,y)
println("z ", z[1,1,1,1,1,1,1])

```

The domainwall fermion is defined in 5D space. The element of x is ```x[ic,ix,iy,iz,it,ialpha,iL]```, where iL is an index on the five dimensional axis. 

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

## (EXPERIMENTAL) Domainwall Fermions
In the case of the domainwall fermion, the action is defined as 

```julia
 L5 = 4
x = Initialize_pseudofermion_fields(U[1],"Domainwall",L5 = L5)
gauss_distribution_fermion!(x)

params = Dict()
params["Dirac_operator"] = "Domainwall"
params["mass"] = 0.1
params["L5"] = L5
params["eps_CG"] = 1.0e-19
params["verbose_level"] = 2
params["method_CG"] = "bicg"
D = Dirac_operator(U,x,params)

parameters_action = Dict()
fermi_action = FermiAction(D,parameters_action)

Sfnew = evaluate_FermiAction(fermi_action,U,x)
println(Sfnew)

UdSfdUμ = calc_UdSfdU(fermi_action,U,x)
```


# Hybrid Monte Carlo with fermions

## Wilson Fermion
We show the HMC code with this package. 

```julia
using Gaugefields
using LatticeDiracOperators
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

# HMC with fermions with stout smearing
We show the code of HMC with Wilson fermions with stout smearing. 

```julia
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using LatticeDiracOperators

function MDtest!(gauge_action,U,Dim,nn,fermi_action,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)
    
    substitute_U!(Uold,U)
    MDsteps = 10
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ)
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


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ)
    

    Δτ = 1/MDsteps
    gauss_distribution!(p)

    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    #Sold = calc_action(gauge_action,Uout,p)

    substitute_U!(Uold,U)

    gauss_sampling_in_action!(ξ,Uout,fermi_action)
    sample_pseudofermions!(η,Uout,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println("Sfold = $Sfold")

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)


    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end

    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    #Snew = calc_action(gauge_action,Uout,p)

    Sfnew = evaluate_FermiAction(fermi_action,Uout,η)
    println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
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


function P_update_fermion!(U,p,ϵ,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    Uout,Uout_multi,_ = calc_smearedU(U,nn)

    for μ=1:Dim
        calc_UdSfdU!(UdSfdUμ,fermi_action,Uout,η)
        mul!(dSdU[μ],Uout[μ]',UdSfdUμ[μ])
    end

    dSdUbare = back_prop(dSdU,nn,Uout_multi,U) 
    

    for μ=1:Dim
        mul!(temps[1],U[μ],dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
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

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(gauge_action,β,plaqloop)

    show(gauge_action)

    L = [NX,NY,NZ,NT]
    nn = CovNeuralnet(U)
    ρ = [0.1]
    layername = ["plaquette"]
    #st = STOUT_Layer(layername,ρ,L)
    st = STOUT_Layer(layername, ρ, U)
    push!(nn,st)
    #push!(nn,st)

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
    

    MDtest!(gauge_action,U,Dim,nn,fermi_action,x,y)

end


test1()
```

# HMC with `GeneralFermion` and Enzyme-based AD (experimental)

`GeneralFermion` is a flexible fermion field type defined in `LatticeDiracOperators.jl`.
It is useful for writing HMC codes where you define the Dirac operator application explicitly.

The built-in `LatticeMatrices` operator registration examples in this section,
including cached clover/HISQ differentiation and the HMC example, require
**LatticeMatrices v1.0.0 or later**. LDO's compatibility with
LatticeMatrices 0.3 is retained for the older API, but the v1 integration
below is enabled only when the v1 operator API is available.

`GeneralFermion`, the built-in `LatticeMatrices` operators, and the
`GeneralFermionAction`/Enzyme HMC workflow support MPI process grids. Set
`PEs` to a process grid whose product is the number of MPI processes.

In this example, **both gauge and fermion forces are computed via automatic differentiation (AD) using `Enzyme.jl`:**

- **Gauge force:** AD of the gauge action `calc_action(...)` w.r.t. link variables.
- **Fermion force:** AD-based force computed by `calc_UdSfdU!` (internally using Enzyme).

## 1. Define `GeneralFermion`

```julia
using Gaugefields
using LatticeDiracOperators
import JACC
using Enzyme
JACC.@init_backend

NX, NY, NZ, NT = 4, 4, 4, 4
NC = 3
NG = 4
gsize = (NX, NY, NZ, NT)

# Use any process grid supported by the selected LatticeMatrices operator
PEs = (1,1,1,1)

x  = GeneralFermion(NC, NG, gsize, PEs; nw=1, elementtype=ComplexF64)
η  = similar(x)
ξ  = similar(x)
```

## 2. Define a fermion action from custom Dirac operators (Wilson example)
```julia
κ = 0.141139
params = (κ=κ,)

apply_D!(y, U1, U2, U3, U4, x, phitemp, temp) =
    apply_wilson!(y, U1, U2, U3, U4, x, params, phitemp, temp)

apply_Ddag!(y, U1, U2, U3, U4, x, phitemp, temp) =
    apply_wilson_dag!(y, U1, U2, U3, U4, x, params, phitemp, temp)

fermi_action = GeneralFermionAction(U, x, apply_D!, apply_Ddag!; numtemp=8, eps_CG=1e-16)
```

## 3. Register built-in LatticeMatrices operators in `GeneralFermionAction`

`GeneralFermion.field` is a `LatticeMatrices.LatticeMatrix`. Gauge fields
created with `isMPILattice=true` also store each link in a `LatticeMatrix`.
An HMC action is constructed by registering `apply_D` and `apply_Ddag` in
`GeneralFermionAction`. Each callback must take the current `U1`, ..., `U4`
arguments explicitly. This is important: the solver uses these callbacks for
`D' * D`, and Enzyme differentiates the same `apply_D` call with respect to
the dynamical links when calculating the fermion force.

The callback signature is
`(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)`. The last two arguments
can be ignored by a `LatticeMatrices` operator because it manages its own
workspace. The following common setup works with one or more MPI processes.
Gauge links are periodic, while the fermions are antiperiodic in time.

```julia
using MPI
using LinearAlgebra
using LatticeMatrices

MPI.Initialized() || MPI.Init()
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

NC = 3
gsize = (4 * nprocs, 4, 4, 4)
PEs = (nprocs, 1, 1, 1)
fermion_phases = (1, 1, 1, -1)

U_lattice = Initialize_Gaugefields(
    NC, 1, gsize...;
    condition="cold",
    isMPILattice=true,
    PEs,
    verbose_level=0,
)
links = [U_mu.U for U_mu in U_lattice]
```

### Wilson fermion

Wilson spinors have four spin components per site (`NG=4`).

```julia
psi_wilson = GeneralFermion(
    NC, 4, gsize, PEs; nw=1, phases=fermion_phases)

function apply_D_wilson!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    D = WilsonDiracOperator4D([U1.U, U2.U, U3.U, U4.U], 0.12)
    mul!(y.field, D, x.field)
    return y
end

function apply_Ddag_wilson!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    D = WilsonDiracOperator4D([U1.U, U2.U, U3.U, U4.U], 0.12)
    mul!(y.field, adjoint(D), x.field)
    return y
end

wilson_action = GeneralFermionAction(
    U_lattice, psi_wilson, apply_D_wilson!, apply_Ddag_wilson!;
    numtemp=8, eps_CG=1e-12)
```

### Wilson--clover fermion

```julia
const D_clover = WilsonDiracCloverOperator4D(links, 0.12, 1.0)

function apply_D_clover!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    mul_cached_clover!(
        y.field, D_clover, U1.U, U2.U, U3.U, U4.U, x.field)
    return y
end

function apply_Ddag_clover!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    mul_cached_clover_adjoint!(
        y.field, D_clover, U1.U, U2.U, U3.U, U4.U, x.field)
    return y
end

clover_action = GeneralFermionAction(
    U_lattice, psi_wilson, apply_D_clover!, apply_Ddag_clover!;
    numtemp=8, eps_CG=1e-12)
```

The last constructor argument is `cSW`. The explicit-link cached entry points
refresh the clover field after a link changes and expose the current links to
the Enzyme pullback.

### One-link staggered fermion

Staggered spinors have one component per site (`NG=1`).

```julia
psi_staggered = GeneralFermion(
    NC, 1, gsize, PEs; nw=1, phases=fermion_phases)

function apply_D_staggered!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    D = StaggeredDiracOperator4D([U1.U, U2.U, U3.U, U4.U], 0.01)
    mul!(y.field, D, x.field)
    return y
end

function apply_Ddag_staggered!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    D = StaggeredDiracOperator4D([U1.U, U2.U, U3.U, U4.U], 0.01)
    mul!(y.field, adjoint(D), x.field)
    return y
end

staggered_action = GeneralFermionAction(
    U_lattice, psi_staggered, apply_D_staggered!, apply_Ddag_staggered!;
    numtemp=8, eps_CG=1e-12)
```

### HISQ fermion

The complete HISQ smearing and Naik stencil require `nw>=3`, so both its thin
links and staggered field use a separate halo width of three.

```julia
U_hisq = Initialize_Gaugefields(
    NC, 3, gsize...;
    condition="cold",
    isMPILattice=true,
    PEs,
    verbose_level=0,
)
links_hisq = [U_mu.U for U_mu in U_hisq]
psi_hisq = GeneralFermion(
    NC, 1, gsize, PEs; nw=3, phases=fermion_phases)

naik_epsilon = -0.083
const hisq_cache = HISQDiracCache4D(
    links_hisq, 0.01; naik_epsilon)

function apply_D_hisq!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    mul_cached_hisq!(
        y.field, hisq_cache, U1.U, U2.U, U3.U, U4.U, x.field)
    return y
end

function apply_Ddag_hisq!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    mul_cached_hisq_adjoint!(
        y.field, hisq_cache, U1.U, U2.U, U3.U, U4.U, x.field)
    return y
end

hisq_action = GeneralFermionAction(
    U_hisq, psi_hisq, apply_D_hisq!, apply_Ddag_hisq!;
    numtemp=8, num=12, numcg=12, numg=18, eps_CG=1e-12)
```

The cached HISQ pullback differentiates through level-1 smearing,
reunitarization, level-2 smearing, and the Naik links to the thin links.

#### Complete `4^4` HISQ HMC example

The following is a complete HMC driver. In particular, `apply_D` and
`apply_Ddag` below are the callable objects registered in
`GeneralFermionAction`; `calc_UdSfdU!` differentiates that same `apply_D`
through the LatticeMatrices v1 HISQ cache to the thin links.

```julia
import JACC
JACC.@init_backend

using Enzyme
using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using MPI
using Random

const HMC_DIM = 4
const HMC_NC = 3
const HMC_SIZE = (4, 4, 4, 4)
const HMC_PES = (1, 1, 1, 1)

struct ApplyHISQForHMC{Adjoint,C}
    cache::C
end

ApplyHISQForHMC{Adjoint}(cache::C) where {Adjoint,C} =
    ApplyHISQForHMC{Adjoint,C}(cache)

function (apply::ApplyHISQForHMC{false})(
    y, U1, U2, U3, U4, x, fermion_temps, gauge_temps,
)
    mul_cached_hisq!(
        y.field, apply.cache, U1.U, U2.U, U3.U, U4.U, x.field)
    return y
end

function (apply::ApplyHISQForHMC{true})(
    y, U1, U2, U3, U4, x, fermion_temps, gauge_temps,
)
    mul_cached_hisq_adjoint!(
        y.field, apply.cache, U1.U, U2.U, U3.U, U4.U, x.field)
    return y
end

function make_hmc_gauge_action(U, beta)
    action = GaugeAction(U)
    plaquette_loops = make_loops_fromname("plaquette")
    append!(plaquette_loops, plaquette_loops')
    push!(action, beta / 2, plaquette_loops)
    return action
end

function hmc_hamiltonian(gauge, U, momentum, fermion_action_value)
    Sg = -evaluate_GaugeAction(gauge, U) / U[1].NC
    Sp = momentum * momentum / 2
    return real(Sg + Sp + fermion_action_value)
end

function hmc_update_links!(U, momentum, step, gauge)
    temps = get_temporary_gaugefields(gauge)
    temp1, temp2, exponential, work = temps[1:4]
    for mu in 1:HMC_DIM
        exptU!(exponential, step, momentum[mu], [temp1, temp2])
        mul!(work, exponential, U[mu])
        substitute_U!(U[mu], work)
    end
    return U
end

function hmc_update_gauge_momentum!(momentum, U, step, gauge)
    temps = get_temporary_gaugefields(gauge)
    derivative = temps[end]
    factor = -step / U[1].NC
    for mu in 1:HMC_DIM
        calc_dSdUμ!(derivative, gauge, mu, U)
        mul!(temps[1], U[mu], derivative)
        Traceless_antihermitian_add!(momentum[mu], factor, temps[1])
    end
    return momentum
end

function hmc_update_fermion_momentum!(
    momentum, U, step, gauge, fermion, pseudofermion,
)
    UdSfdU = get_temporary_gaugefields(gauge)[1:HMC_DIM]
    calc_UdSfdU!(UdSfdU, fermion, U, pseudofermion)
    for mu in 1:HMC_DIM
        Traceless_antihermitian_add!(momentum[mu], -step, UdSfdU[mu])
    end
    return momentum
end

function hisq_hmc_trajectory!(
    U, momentum, old_U, pseudofermion, gaussian_fermion, gauge, fermion;
    mdsteps, trajectory_length,
)
    step = trajectory_length / mdsteps

    gauss_distribution!(momentum)
    gauss_sampling_in_action!(gaussian_fermion, U, fermion)
    sample_pseudofermions!(pseudofermion, U, fermion, gaussian_fermion)
    old_Sf = real(dot(gaussian_fermion, gaussian_fermion))

    substitute_U!(old_U, U)
    old_H = hmc_hamiltonian(gauge, U, momentum, old_Sf)

    for _ in 1:mdsteps
        hmc_update_links!(U, momentum, step / 2, gauge)
        hmc_update_gauge_momentum!(momentum, U, step, gauge)
        hmc_update_fermion_momentum!(
            momentum, U, step, gauge, fermion, pseudofermion)
        hmc_update_links!(U, momentum, step / 2, gauge)
    end

    new_Sf = evaluate_FermiAction(fermion, U, pseudofermion)
    new_H = hmc_hamiltonian(gauge, U, momentum, new_Sf)
    delta_H = new_H - old_H
    accepted = log(rand()) < -delta_H
    accepted || substitute_U!(U, old_U)
    return (; accepted, delta_H)
end

function run_hisq_hmc_4x4(;
    trajectories=1,
    mdsteps=2,
    trajectory_length=0.02,
    beta=6.0,
    mass=0.1,
    naik_epsilon=0.0,
    eps_CG=1e-10,
    seed=1234,
)
    MPI.Initialized() || MPI.Init()
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error(
        "The fixed 4^4, nw=3 example must run on one MPI rank")
    Random.seed!(seed)

    U = Initialize_Gaugefields(
        HMC_NC, 3, HMC_SIZE...;
        condition="cold",
        isMPILattice=true,
        PEs=HMC_PES,
        verbose_level=0,
    )
    gauge = make_hmc_gauge_action(U, beta)

    template = GeneralFermion(
        HMC_NC, 1, HMC_SIZE, HMC_PES;
        nw=3,
        phases=(1, 1, 1, -1),
        elementtype=ComplexF64,
    )
    cache = HISQDiracCache4D([link.U for link in U], mass; naik_epsilon)

    # These are the D and Ddag callbacks used by both the solver and HMC force.
    apply_D = ApplyHISQForHMC{false}(cache)
    apply_Ddag = ApplyHISQForHMC{true}(cache)
    numtemp = 8
    fermion = GeneralFermionAction(
        U, template, apply_D, apply_Ddag;
        numtemp,
        num=12,
        numcg=12,
        numg=2numtemp + 2,
        eps_CG,
        maxsteps=10_000,
        verbose_level=0,
    )

    pseudofermion = template
    gaussian_fermion = similar(template)
    momentum = initialize_TA_Gaugefields(U)
    old_U = similar(U)
    substitute_U!(old_U, U)

    accepted_count = 0
    for trajectory in 1:trajectories
        result = hisq_hmc_trajectory!(
            U, momentum, old_U, pseudofermion, gaussian_fermion,
            gauge, fermion;
            mdsteps,
            trajectory_length,
        )
        accepted_count += result.accepted
        println(
            "trajectory $trajectory: accepted=$(result.accepted) ",
            "deltaH=$(result.delta_H)",
        )
    end
    println("acceptance rate = ", accepted_count / trajectories)
    return U
end

run_hisq_hmc_4x4()
```

The maintained executable version of this code is
`examples/HISQ_HMC_4x4.jl`.

Run the example from this package directory after activating its Julia
environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("examples/HISQ_HMC_4x4.jl")
HISQHMC4x4Example.main()
```

The fixed `4^4` example runs on one MPI rank for a geometric reason: HISQ
needs `nw=3`, while decomposing any length-four direction would make its local
extent smaller than three. This is not a restriction of
`GeneralFermionAction`. MPI HISQ HMC uses the same callbacks on a larger
lattice, with every local extent at least three.

### Standard (Shamir) domain-wall fermion

The fermion is five-dimensional, but its gauge links remain four-dimensional.
The fifth direction must not be MPI-decomposed. In the LatticeMatrices
parameter convention, `b=c=1` selects the standard Shamir kernel.

```julia
const L5 = 8
gsize5 = (gsize..., L5)
PEs5 = (PEs..., 1)
phases5 = (fermion_phases..., 1)

psi_domainwall = GeneralFermion(
    NC, 4, gsize5, PEs5; nw=1, phases=phases5)

function apply_D_domainwall!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    D = D5DW_MobiusDomainwallOperator5D(
        [U1.U, U2.U, U3.U, U4.U], L5, 0.01, -1.0, 1.0, 1.0)
    mul!(y.field, D, x.field)
    return y
end

function apply_Ddag_domainwall!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    D = D5DW_MobiusDomainwallOperator5D(
        [U1.U, U2.U, U3.U, U4.U], L5, 0.01, -1.0, 1.0, 1.0)
    mul!(y.field, adjoint(D), x.field)
    return y
end

domainwall_action = GeneralFermionAction(
    U_lattice, psi_domainwall, apply_D_domainwall!, apply_Ddag_domainwall!;
    numtemp=8, eps_CG=1e-12)
```

### Generalized domain-wall fermion

The generalized operator accepts independent real `a_s`, `b_s`, and `c_s`
coefficients for every fifth-dimensional slice.

```julia
const a5 = 1 .+ 0.05 .* sin.(2pi .* (0:L5-1) ./ L5)
const b5 = 1.5 .+ 0.10 .* cos.(2pi .* (0:L5-1) ./ L5)
const c5 = 0.5 .+ 0.08 .* sin.(2pi .* (0:L5-1) ./ L5)

function apply_D_generalized!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    D = D5DW_GeneralizedDomainwallOperator5D(
        [U1.U, U2.U, U3.U, U4.U], L5, 0.01, -1.0, a5, b5, c5)
    mul!(y.field, D, x.field)
    return y
end

function apply_Ddag_generalized!(y, U1, U2, U3, U4, x, fermion_temps, gauge_temps)
    D = D5DW_GeneralizedDomainwallOperator5D(
        [U1.U, U2.U, U3.U, U4.U], L5, 0.01, -1.0, a5, b5, c5)
    mul!(y.field, adjoint(D), x.field)
    return y
end

generalized_action = GeneralFermionAction(
    U_lattice, psi_domainwall,
    apply_D_generalized!, apply_Ddag_generalized!;
    numtemp=8, eps_CG=1e-12)
```

`GeneralFermionAction` represents the pseudofermion action
`phi' * (D' * D)^(-1) * phi`. Fractional determinants, such as rooted
staggered/HISQ physical-flavor simulations, require an RHMC rational power.

The regression test `test/latticematrices_backend.jl` registers and executes
`apply_D` and `apply_Ddag` for all six cases on the active `LatticeMatrices`
backend.

## 4. AD gauge force (Enzyme)
```julia
set_wing_U!(U)
Enzyme_derivative!(
    calc_action,
    U1, U2, U3, U4,
    dSdU[1], dSdU[2], dSdU[3], dSdU[4],
    nodiff(β), nodiff(NC);
    temp,
    dtemp
)
```

## 5. AD fermion force (Enzyme)
The fermion force is obtained by
```julia
calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)
```
where calc_UdSfdU! computes U dSf/dU using **Enzyme-based AD**.
Full working example:
examples/HMC_AD.jl


# SLHMC

```julia
using LinearAlgebra
using Optimisers
using Wilsonloop
using Gaugefields
using LatticeDiracOperators
import Gaugefields.Abstractsmearing_module: get_parameters, zero_grad!, set_parameters!

# For debug
import InteractiveUtils
versionstring = """
$(InteractiveUtils.versioninfo())
LatticeDiracOperators $(pkgversion(LatticeDiracOperators))
Gaugefields $(pkgversion(Gaugefields))
Wilsonloop $(pkgversion(Wilsonloop))
"""
println(versionstring)
#---

const NX = 4
const NY = 4
const NZ = 4
const NT = 4

const Ncin = 2
const β0 = 2.45
const mass = 0.3
const mass_eff = 0.4

const Nf = 4
const filename = "mass005eff01_stout_2_1109.txt"
const paramfilename = "param_" * filename
const actionfilename = "diff_" * filename

const MDsteps = 20
const numtrj = 100
const numbatch = 1
const eta = 1e-4 #parameter for ADAM 
const optimiser = Optimisers.Adam(eta)
const trainable = true

function MDtest!(gauge_action, U, Dim, nn, fermi_action, η, ξ, fermi_action_eff, numtrj, isbare)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)

    substitute_U!(Uold, U)
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0
    fp = open(filename, "w")
    fp2 = open(paramfilename, "w")
    fp3 = open(actionfilename, "w")

    θ = get_parameters(nn)
    state = Optimisers.setup(optimiser, θ)
    dLdθ = zero(θ)


    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, fermi_action, η, ξ, fermi_action_eff, state, θ, dLdθ, fp3, isbare)

        if itrj % numbatch == 0 && !isbare
            println("θ_before = $θ        ")
            if trainable
                println("dLdθ ")
                display(dLdθ)
                Optimisers.update!(state, θ, dLdθ / numbatch)
            end
            println("θ = $θ        ")
            set_parameters!(nn, θ)
            dLdθ .= 0
        end


        for p in θ
            print(fp2, p, "\t")
        end
        println(fp2, "\t")
        flush(fp2)
        flush(fp3)
        numaccepted += ifelse(accepted, 1, 0)

        plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        println("$itrj plaq_t = $plaq_t")
        #println(fp,"$itrj $plaq_t")
        println("acceptance ratio ", numaccepted / itrj)
        println(fp, "$itrj $plaq_t $(numaccepted / itrj) #itrj plaq_t acceptanceratio")
        flush(fp)

    end
    close(fp)
    close(fp2)
    close(fp3)
end

function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, fermi_action, η, ξ,
    fermi_action_eff, state, θ, dLdθ, fp3, isbare)

    Δτ = 1 / MDsteps
    gauss_distribution!(p)


    #Uout, Uout_multi, _ = calc_smearedU(U, nn)
    substitute_U!(Uold, U)

    gauss_sampling_in_action!(ξ, U, fermi_action)
    sample_pseudofermions!(η, U, fermi_action, ξ)
    Sfold = real(dot(ξ, ξ))
    println("Sfold = $Sfold")

    Sgold = calc_action(gauge_action, U, p)
    println("Sgold = $Sgold")
    Sold = Sgold + Sfold
    println("Sold = ", Sold)


    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action)
        #P_update_fermion!(U, p, 1.0, Δτ, Dim, gauge_action, dSdU, nn, fermi_action, η)
        P_update_fermion!(U, p, 1.0, Δτ, Dim, gauge_action, dSdU, nn, fermi_action_eff, η, isbare)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
    end

    Sfnew = evaluate_FermiAction(fermi_action, U, η)
    println("Sfnew = $Sfnew")
    if !isbare
        Uout, Uout_multi, _ = calc_smearedU(U, nn)
        Sfnew_eff = evaluate_FermiAction(fermi_action_eff, Uout, η)
        println(fp3, "$Sfnew \t $(Sfnew_eff) #Sf, Sf_eff")
        zero_grad!(nn)
    end




    if !isbare
        temps = get_temporary_gaugefields(gauge_action)
        UdSfdUμ = temps[1:Dim]
        for μ = 1:Dim
            calc_UdSfdU!(UdSfdUμ, fermi_action_eff, Uout, η)
            mul!(dSdU[μ], Uout[μ]', UdSfdUμ[μ])
        end

        dSdUbare = back_prop(dSdU, nn, Uout_multi, U)
        dSdw = deepcopy(get_parameter_derivatives(nn) * -1)

        loss = (Sfnew - Sfnew_eff)^2
        dLdw = (-2) * dSdw * (Sfnew - Sfnew_eff)


        dLdθ .+= dLdw
        println(dSdw)
        println(dLdw)
        println("loss = $loss")
    end


    Sgnew = calc_action(gauge_action, U, p)
    Snew = Sgnew + Sfnew
    println("Sgnew = $Sgnew")
    println("Sg: Sgnew Sgold Sgnew-Sgold: $Sgnew $Sgold $(Sgnew-Sgold)")
    println("Sf: Sfnew Sfold Sfnew-Sfold: $Sfnew $Sfold $(Sfnew-Sfold)")

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-(Snew - Sold)))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end


function U_update!(U, p, ϵ, Δτ, Dim, gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]
    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)

    end
end

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temps[1], U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temps[1])
    end
end


function P_update_fermion!(U, p, ϵ, Δτ, Dim, gauge_action, dSdU,
    nn, fermi_action, η, isbare)  # p -> p +factor*U*dSdUμ
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor = -ϵ * Δτ

    if !isbare
        Uout, Uout_multi, _ = calc_smearedU(U, nn)
        for μ = 1:Dim
            calc_UdSfdU!(UdSfdUμ, fermi_action, Uout, η)
            mul!(dSdU[μ], Uout[μ]', UdSfdUμ[μ])
        end
        dSdUbare = back_prop(dSdU, nn, Uout_multi, U)

        for μ = 1:Dim
            mul!(temps[1], U[μ], dSdUbare[μ]) # U*dSdUμ
            Traceless_antihermitian_add!(p[μ], factor, temps[1])
        end

    else
        for μ = 1:Dim
            calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)
            mul!(dSdU[μ], U[μ]', UdSfdUμ[μ])
        end

        for μ = 1:Dim
            mul!(temps[1], U[μ], dSdU[μ]) # U*dSdUμ
            Traceless_antihermitian_add!(p[μ], factor, temps[1])
        end
    end


end

function test1()
    Nwing = 0
    Dim = 4
    NC = Ncin

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = β0 / 2
    push!(gauge_action, β, plaqloop)

    show(gauge_action)

    L = [NX, NY, NZ, NT]

    nn = CovNeuralnet(U)
    layername = ["plaquette", "polyakov_x", "polyakov_y", "polyakov_z", "polyakov_t"]
    ρ = (2 * rand(length(layername)) .- 1) * 1e-3
    st = STOUT_Layer(layername, ρ, U)
    push!(nn, st)
    ρ = (2 * rand(length(layername)) .- 1) * 1e-3
    st2 = STOUT_Layer(layername, ρ, U)
    push!(nn, st2)

    x = Initialize_pseudofermion_fields(U[1], "staggered")
    gauss_distribution_fermion!(x)
    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = mass
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U, x, params)

    parameters_action = Dict()
    parameters_action["Nf"] = Nf
    fermi_action = FermiAction(D, parameters_action)

    y = similar(x)

    isbare = true
    MDtest!(gauge_action, U, Dim, nn, fermi_action, x, y, fermi_action, 10, isbare)

    x_eff = Initialize_pseudofermion_fields(U[1], "staggered")
    params_eff = Dict()
    params_eff["Dirac_operator"] = "staggered"
    params_eff["mass"] = mass_eff

    params_eff["eps_CG"] = 1.0e-8
    params_eff["verbose_level"] = 2
    D_eff = Dirac_operator(U, x_eff, params_eff)
    parameters_action_eff = Dict()
    parameters_action_eff["Nf"] = Nf
    fermi_action_eff = FermiAction(D_eff, parameters_action_eff)

    isbare = false
    MDtest!(gauge_action, U, Dim, nn, fermi_action, x, y, fermi_action_eff, numtrj, isbare)

end

test1()
```

# Acknowledgment
If you write a paper using this package, please refer this code.

BibTeX citation is following
```
@article{Nagai:2024yaf,
    author = "Nagai, Yuki and Tomiya, Akio",
    title = "{JuliaQCD: Portable lattice QCD package in Julia language}",
    eprint = "2409.03030",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    month = "9",
    year = "2024"
}
```
and the paper is [arXiv:2409.03030](https://arxiv.org/abs/2409.03030).
