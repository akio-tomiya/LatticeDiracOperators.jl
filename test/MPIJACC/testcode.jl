using Gaugefields
using LatticeDiracOperators
using JACC

JACC.@init_backend

function test()

    U = Initialize_Gaugefields(3, 1, 4, 4, 4, 4, condition="cold", isMPILattice=true)
    x = Initialize_pseudofermion_fields(U[1], "Wilson")

    return nothing

end

test()