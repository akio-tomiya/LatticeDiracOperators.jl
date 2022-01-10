module LatticeDiracOperators
    using Wilsonloop
    using Gaugefields
    using Requires
    using AlgRemez_jll

    import Gaugefields:add_U!
    import Gaugefields:Abstractfields,clear_U!

    # Write your package code here.
    include("./rhmc/AlgRemez.jl")
    include("./rhmc/rhmc.jl")
    


    
    include("Diracoperators.jl")

    import .Dirac_operators:Initialize_pseudofermion_fields,Dirac_operator,gauss_distribution_fermion!,
            Initialize_WilsonFermion,Initialize_4DWilsonFermion,
            DdagD_operator,solve_DinvX!,FermiAction,
            shift_fermion,
            sample_pseudofermions!,gauss_sampling_in_action!,evaluate_FermiAction,calc_UdSfdU,calc_UdSfdU!,bicgstab,
            Wilson_Dirac_operator_evenodd,calc_p_UdSfdU!

    export Initialize_pseudofermion_fields,Dirac_operator,gauss_distribution_fermion!,cg,bicg
    export Initialize_WilsonFermion,Initialize_4DWilsonFermion
    export DdagD_operator,solve_DinvX!,FermiAction
    export shift_fermion
    export WilsonFermion_4D_wing
    export sample_pseudofermions!,gauss_sampling_in_action!,evaluate_FermiAction,calc_UdSfdU,calc_UdSfdU!,bicgstab,calc_p_UdSfdU!
    export Wilson_Dirac_operator_evenodd
end
