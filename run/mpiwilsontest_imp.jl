using InteractiveUtils
#versioninfo()
using Gaugefields
using MPI
using LinearAlgebra
using Random
using LatticeDiracOperators

import Gaugefields.AbstractGaugefields_module:getvalue,setvalue!


#=
if length(ARGS)<7
    error("USAGE:\njulia exe.jl nPEx nPEy nPEz nPEt mpi-true/false Ntrj nMDsteps")
end
=#
if length(ARGS)<5
    error("USAGE:\njulia exe.jl nPEx nPEy nPEz nPEt mpi-true/false")
end

println("Test for 1/D for Wilson MPI")

versioninfo()

const pes = Tuple(parse.(Int64,ARGS[1:4]))
const mpi = parse(Bool,ARGS[5])
const numtrj_in = 10 # parse(Int64, ARGS[6])
const MDsteps_in = 0 # parse(Int64, ARGS[7])

const βin = 6.0
const NXin = 16
const NYin = 16
const NZin = 16
const NTin = 32
const κin = 0.141139

header = """
Parameters
const pes = $pes
const mpi = $mpi
const numtrj_in = $numtrj_in
const MDsteps_in = $MDsteps_in

const βin = $βin
const NXin = $NXin
const NYin = $NYin
const NZin = $NZin
const NTin = $NTin
const κin = $κin
"""


function MDtest!(snet,U,Dim,mpi=false)
    p = initialize_TA_Gaugefields(U)
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = MDsteps_in
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    plaq_t = calculate_Plaquette(U,temp1,temp2)*factor

    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    if get_myrank(U) == 0
        println("0 $plaq_t # plaquette")
        println("0 $(real(poly)) $(imag(poly)) # polyakovloop")
    end


    numtrj = numtrj_in
    for itrj = 1:numtrj
        accepted, runtime = @timed MDstep!(snet,U,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        poly = calculate_Polyakov_loop(U,temp1,temp2) 
        
        if get_myrank(U) == 0
            #println("$itrj plaq_t = $plaq_t")
            println("$itrj $(numaccepted/itrj) #acceptance")
            println("$itrj $runtime # runtime")
            #println("polyakov loop = $(real(poly)) $(imag(poly))")
            println("$itrj $plaq_t # plaquette")
            println("$itrj $(real(poly)) $(imag(poly)) # polyakovloop")
        end
    end
end

function calc_action(snet,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(snet,U)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(snet,U,p,MDsteps,Dim,Uold)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(snet,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,snet)
        #println(getvalue(U[1],1,1,1,1,1,1))

        P_update!(U,p,1.0,Δτ,Dim,snet)
        #if get_myrank(U) == 0
        #    println(getvalue(U[1],1,1,1,1,1,1))
        #    println("p = ",p[1][1,1,1,1,1])
        #    if isnan(p[1][1,1,1,1,1])
        #        error("p")
        #    end
        #end


        U_update!(U,p,0.5,Δτ,Dim,snet)
        #error("dd")
    end
    #error("end")
    
    Snew = calc_action(snet,U,p)
    if get_myrank(U) == 0
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1,exp(Snew-Sold))
    r = rand()
    if mpi
        r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    end
    #println(r,"\t",ratio)

    if r > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,snet)
    temps = get_temporary_gaugefields(snet)
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

function P_update!(U,p,ϵ,Δτ,Dim,snet) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(snet)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,snet,μ,U)
        #println("dSdU = ",getvalue(dSdUμ,1,1,1,1,1,1))
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end



function test1()
    NX = NXin
    NY = NYin
    NZ = NZin
    NT = NTin
    Nwing = 0
    Dim = 4
    NC = 3


    if mpi
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",mpi=true,PEs = PEs,mpiinit = false)
        
    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    end

    Random.seed!(123+get_myrank(U[1]))    

    if get_myrank(U) == 0
        println(header) # print header
    end

    snet = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = βin/2
    push!(snet,β,plaqloop)
    
    #show(snet)

    nowing = true
    Random.seed!(123)
    x = Initialize_pseudofermion_fields(U[1],"Wilson",nowing = nowing)

    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = κin
    params["faster version"] = true
    params["eps_CG"] = 1.0e-16
    params["verbose_level"] = 3 # to show CG counts

    D = Dirac_operator(U,x,params)

    if mpi
        if x.myrank == 0
            setvalue!(x,10.0,1,1,1,1,1,1)
            setvalue!(x,im,1,2,1,1,1,1)
            setvalue!(x,2im,1,3,1,1,1,1)
            setvalue!(x,3im,1,4,1,1,1,1)
            setvalue!(x,10.0,1,1,2,1,1,1)
            setvalue!(x,im,1,2,1,1,4,1)
            setvalue!(x,2im,1,3,1,4,1,1)
            setvalue!(x,3im,1,4,1,1,1,1)
        end
        
    else
        setvalue_fermion!(x,10.0,1,1,1,1,1,1)
        setvalue_fermion!(x,im,1,2,1,1,1,1)
        setvalue_fermion!(x,2im,1,3,1,1,1,1)
        setvalue_fermion!(x,3im,1,4,1,1,1,1)
        setvalue_fermion!(x,10.0,1,1,2,1,1,1)
        setvalue_fermion!(x,im,1,2,1,1,4,1)
        setvalue_fermion!(x,2im,1,3,1,4,1,1)
        setvalue_fermion!(x,3im,1,4,1,1,1,1)
        
    end
    set_wing_fermion!(x)

    

    if mpi
        if x.myrank == 0
            println(getvalue(x,1,1,1,1,1,1))
            println(getvalue(x,1,2,1,1,1,1))
        end
    else
        println(x[1,1,1,1,1,1])
        println(x[1,2,1,1,1,1])
    end
    #Random.seed!(123)
    #gauss_distribution_fermion!(x)
    

    b = similar(x)
    y = similar(x)
    @time solve_DinvX!(b,D,x)
    @time mul!(b,D,x)
    @time mul!(b,D,x)

    #return
    println_verbose_level1(U[1],"Benchmark!!")
    @time solve_DinvX!(y,D,b)

    println_verbose_level1(U[1],"We check whether y = D^-1 (Dx) = x")
    if mpi 
        if x.myrank == 0
            println(getvalue(y,1,1,1,1,1,1))
            println(getvalue(y,1,2,1,1,1,1))
        end
    else
        println(y[1,1,1,1,1,1])
        println(y[1,2,1,1,1,1])
    end

    println_verbose_level1(U[1],"several times!!!")
    elptime = 0.0
    tmp = 0.0
    for itrj=1:numtrj_in 
        println_verbose_level1(U[1],"itrj = $itrj")
        gauss_distribution_fermion!(x)
        tmp = @elapsed solve_DinvX!(y,D,x)
        elptime+=tmp
    end
    elptime/=numtrj_in
    println_verbose_level1(U[1],"$eltime #Dtime")


    #MDtest!(snet,U,Dim,mpi)

end


test1()
if mpi
    MPI.Finalize()
end