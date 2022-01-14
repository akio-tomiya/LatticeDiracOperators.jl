
using LinearAlgebra
using LatticeDiracOperators

function test_staggered()
    NX=4
    NY=4
    NZ=4
    NT=4
    Nwing = 1
    NC=3

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    U2 = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    x = Initialize_pseudofermion_fields(U[1],"staggered")
    

    gauss_distribution_fermion!(x)
    #x[1,1,1,1,1,1] = 4
    println(x[1,1,1,1,1,1])

    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = 0.1
    params["verbose_level"] = 3
    D = Dirac_operator(U,x,params)
    D2 = D(U2)

    for Nf in [8,4,2]
        println("Nf = $Nf")
        parameters_action = Dict()
        parameters_action["Nf"] = Nf
        fermi_action = FermiAction(D,parameters_action)
        gauss_sampling_in_action!(x,U,fermi_action)
        println("Sfold = ", dot(x,x))
        y = similar(x)
        sample_pseudofermions!(y,U,fermi_action,x)

        UdSfdU = calc_UdSfdU(fermi_action,U,y)

        Sf = evaluate_FermiAction(fermi_action,U,y)
        println("Sfnew = ", Sf)

    end



    y = similar(x)
    mul!(y,D,x)
    mul!(y,D2,x)
    mul!(y,D(U2),x)

    println("BICG method")
    @time solve_DinvX!(y,D,x)
    #@time bicg(y,D,x,verbose = Verbose_3())

    DdagD = DdagD_operator(U,x,params)
    mul!(y,DdagD,x)

    @time solve_DinvX!(y,DdagD,x)

    #@time cg(y,DdagD,x,verbose = Verbose_3())



end

function test_wilson()
    NX=4
    NY=4
    NZ=4
    NT=4
    Nwing = 1
    NC=3

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    x = Initialize_pseudofermion_fields(U[1],"Wilson")
    gauss_distribution_fermion!(x)
    println(x[1,1,1,1,1,1])

    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.1
    params["verbose_level"] = 3
    D = Dirac_operator(U,x,params)
    D2 = D(U)

    y = similar(x)

    for i=1:3
        y = similar(x)
        y1 = deepcopy(y)
        y2 = deepcopy(y)

        println("BICG")
        @time bicg(y1,D,x, verbose = Verbose_print(3))
        println(dot(y1,y1))
        println("BICGstab")
        @time bicgstab(y2,D,x,verbose = Verbose_print(3))
        println(dot(y2,y2))
    end

    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)
    gauss_sampling_in_action!(x,U,fermi_action)
    println("Sfold = ", dot(x,x))
    y = similar(x)
    sample_pseudofermions!(y,U,fermi_action,x)

    UdSfdU = calc_UdSfdU(fermi_action,U,y)

    Sf = evaluate_FermiAction(fermi_action,U,y)
    println("Sfnew = ", Sf)

    #=

    y = similar(x)
    mul!(y,D,x)

    println("BICG method")
    @time solve_DinvX!(y,D,x)

    #@time bicg(y,D,x,verbose = Verbose_3())

    DdagD = DdagD_operator(U,x,params)
    mul!(y,DdagD,x)
    @time solve_DinvX!(y,DdagD,x)
    #@time cg(y,DdagD,x,verbose = Verbose_3())
    =#


    return 


end
#test_staggered()
test_wilson()