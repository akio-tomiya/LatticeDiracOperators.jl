
import Gaugefields:Verbose_3
function test()
    NX=4
    NY=4
    NZ=4
    NT=4
    Nwing = 1
    NC=3

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    x = Initialize_pseudofermion_fields(U[1],"staggered")

    gauss_distribution_fermion!(x)
    #x[1,1,1,1,1,1] = 4
    println(x[1,1,1,1,1,1])

    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = 0.1
    D = Dirac_operator(U,x,params)
    y = similar(x)
    mul!(y,D,x)

    println("BICG method")
    bicg(y,D,x,verbose = Verbose_3())
    


end
test()