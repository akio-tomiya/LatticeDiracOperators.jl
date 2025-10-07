using CUDA
using Gaugefields
using LinearAlgebra
using InteractiveUtils
using Random
using LatticeDiracOperators
import Gaugefields: Initialize_4DGaugefields
import LatticeDiracOperators.SSmodule: shiftedbicg_inSS, shiftedbicgstab_inSS, shiftedbicgstab,
    shiftedbicg_Frommer2003, shiftedbicg_Frommer2003_seed, shiftedbicg_Frommer2003_G_seed
import JACC
JACC.@init_backend


function main()
    NX = 24
    NY = 24
    NZ = 24
    NT = 24
    Nwing = 1
    Dim = 4
    NC = 3
    singleprecision = false

    #U = Initialize_4DGaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold")
    Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
#    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold";
#        isMPILattice=true,singleprecision)
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold"; isMPILattice=true)
    Ucuda = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold"; accelerator="cuda")

    x = Initialize_pseudofermion_fields(U[1], "Wilson")
    xcuda = Initialize_pseudofermion_fields(Ucuda[1], "Wilson")
    xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson")

    substitute_U!(U, Ucpu)
    substitute_U!(Ucuda, Ucpu)



        params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.141139 / 2
    params["eps_CG"] = 1.0e-18
    #params["eps_CG"] = 1.0e-1
    #params["verbose_level"] = 3
    #params["method_CG"] = "preconditiond_bicgstab"
    #params["method_CG"] = "bicgstab"
    params["method_CG"] = "bicg"
    params["faster version"] = true
    D = Dirac_operator(U, x, params)

    Dcpu = Dirac_operator(Ucpu, xcpu, params)
    Dcuda = Dirac_operator(Ucuda, xcuda, params)


    parameters_action = Dict()
    fermi_action = FermiAction(Dcpu, parameters_action)
    gauss_sampling_in_action!(xcpu, Ucpu, fermi_action)



    substitute_fermion!(x,xcpu)
    substitute_fermion!(xcuda,xcpu)


    y = similar(x)
    ycpu = similar(xcpu)
    ycuda = similar(xcuda)
    @time mul!(y, D, x)
    println(dot(y,y))
    @time mul!(ycpu, Dcpu, xcpu)
    println(dot(ycpu,ycpu))
    @time mul!(ycuda, Dcuda, xcuda)
    println(dot(ycuda,ycuda))

    t =0.0
    tcpu = 0.0
    tcuda = 0.0
    n = 10

    for i=1:n
        gauss_sampling_in_action!(xcpu, Ucpu, fermi_action)
        substitute_fermion!(x,xcpu)
        substitute_fermion!(xcuda,xcpu)

        t += @elapsed mul!(y, D, x)
        tcpu += @elapsed mul!(ycpu, Dcpu, xcpu)
        tcuda += @elapsed mul!(ycuda, Dcuda, xcuda)

        println(" i = ", i, " JACC time = ", t/i, " CPU time = ", tcpu/i, " CUDA time = ", tcuda/i)
        println(" |y|^2 = ", dot(y,y), " |ycpu|^2 = ", dot(ycpu,ycpu), " |ycuda|^2 = ", dot(ycuda,ycuda))
    end

    println(" Average JACC time = ", t/n, " Average CPU time = ", tcpu/n, " Average CUDA time = ", tcuda/n)






end

main()