
using Gaugefields
using CUDA
using LinearAlgebra
using Random
using LatticeDiracOperators
using InteractiveUtils
import Gaugefields.AbstractGaugefields_module:getvalue,setvalue!



if length(ARGS)<5
    error("USAGE:\njulia exe.jl blocks[1] blocks[2] blocks[3] blocks[4] cuda-true/false")
end

println("Test for 1/D for Wilson CUDA")

versioninfo()

const pes = Tuple(parse.(Int64,ARGS[1:4]))
const cuda = parse(Bool,ARGS[5])
const numtrj_in = 10 # parse(Int64, ARGS[6])
const MDsteps_in = 0 # parse(Int64, ARGS[7])

const βin = 6.0
const NXin = 16*2
const NYin = 16*2
const NZin = 16*2
const NTin = 32
const κin = 0.141139

header = """
Parameters
const blocks = $pes
const cuda = $cuda
const numtrj_in = $numtrj_in
const MDsteps_in = $MDsteps_in

const βin = $βin
const NXin = $NXin
const NYin = $NYin
const NZin = $NZin
const NTin = $NTin
const κin = $κin
"""



function test1()
    NX = NXin
    NY = NYin
    NZ = NZin
    NT = NTin
    Nwing = 0
    Dim = 4
    NC = 3


    if cuda
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",cuda=true,blocks=PEs)
        
    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    end
    Ucpu = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    substitute_U!(Ucpu,U)

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

    Random.seed!(123)
    x = Initialize_pseudofermion_fields(U[1],"Wilson")
    xcpu = Initialize_pseudofermion_fields(Ucpu[1],"Wilson",nowing=true)

    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = κin
    params["faster version"] = true
    params["eps_CG"] = 1.0e-16
    params["verbose_level"] = 3 # to show CG counts

    D = Dirac_operator(U,x,params)


    setvalue_fermion!(xcpu,10.0,1,1,1,1,1,1)
    setvalue_fermion!(xcpu,im,1,2,1,1,1,1)
    setvalue_fermion!(xcpu,2im,1,3,1,1,1,1)
    setvalue_fermion!(xcpu,3im,1,4,1,1,1,1)
    setvalue_fermion!(xcpu,10.0,1,1,2,1,1,1)
    setvalue_fermion!(xcpu,im,1,2,1,1,4,1)
    setvalue_fermion!(xcpu,2im,1,3,1,4,1,1)
    setvalue_fermion!(xcpu,3im,1,4,1,1,1,1)

    substitute_fermion!(x,xcpu)


    b = similar(x)
    y = similar(x)
    @time solve_DinvX!(b,D,x)
    @time mul!(b,D,x)
    @time mul!(b,D,x)

    println_verbose_level1(U[1],"Benchmark!!")
    @time solve_DinvX!(y,D,b)


    println_verbose_level1(U[1],"We check whether y = D^-1 (Dx) = x")
    ycpu = similar(xcpu)
    substitute_fermion!(ycpu,y)

    println(ycpu[1,1,1,1,1,1])
    println(ycpu[1,2,1,1,1,1])

    println_verbose_level1(U[1],"several times!!!")
    elptime = 0.0
    tmp = 0.0
    for itrj=1:numtrj_in 
        println_verbose_level1(U[1],"itrj = $itrj")
        gauss_distribution_fermion!(x)
        tmp = @elapsed solve_DinvX!(y,D,x)
        println("# CG solved $tmp sec")
        elptime+=tmp
    end
    elptime/=numtrj_in
    println_verbose_level1(U[1],"$elptime #Dtime")


    

    return
end
test1()