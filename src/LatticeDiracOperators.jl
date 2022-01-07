module LatticeDiracOperators
    using Wilsonloop
    using Gaugefields
    using Requires
    using AlgRemez_jll
    # Write your package code here.
    include("./rhmc/AlgRemez.jl")
    include("./rhmc/rhmc.jl")
    


    include("./cgmethods.jl")
    include("Diracoperators.jl")

    export Initialize_pseudofermion_fields,Dirac_operator,gauss_distribution_fermion!,cg,bicg
    export DdagD_operator,solve_DinvX!,FermiAction
    export sample_pseudofermions!,gauss_sampling_in_action!
end
