function dw()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0
    Dim = 4
    NC = 3

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold")

    L5 = 4
    x = Initialize_pseudofermion_fields(U[1], "Domainwall", L5=L5)
    gauss_distribution_fermion!(x)

    params = Dict()
    params["Dirac_operator"] = "Domainwall"
    params["mass"] = 0.1
    params["L5"] = L5
    params["eps_CG"] = 1.0e-19
    params["verbose_level"] = 2
    params["method_CG"] = "bicg"
    D = Dirac_operator(U, x, params)

    parameters_action = Dict()
    fermi_action = FermiAction(D, parameters_action)

    Sfnew = evaluate_FermiAction(fermi_action, U, x)
    println(Sfnew)

    UdSfdUμ = calc_UdSfdU(fermi_action, U, x)
end

println("4D DW ")
dw()