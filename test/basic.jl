
import Gaugefields:Verbose_3
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
    D = Dirac_operator(U,x,params)
    D2 = D(U2)
    

    y = similar(x)
    mul!(y,D,x)
    mul!(y,D2,x)
    mul!(y,D(U2),x)

    println("BICG method")
    bicg(y,D,x,verbose = Verbose_3())

    DdagD = DdagD_operator(U,x,params)
    mul!(y,DdagD,x)

    cg(y,DdagD,x,verbose = Verbose_3())



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
    D = Dirac_operator(U,x,params)
    D2 = D(U)

    y = similar(x)
    mul!(y,D,x)

    println("BICG method")
    bicg(y,D,x,verbose = Verbose_3())

    DdagD = DdagD_operator(U,x,params)
    mul!(y,DdagD,x)
    cg(y,DdagD,x,verbose = Verbose_3())


    return 


end
#test_staggered()
test_wilson()