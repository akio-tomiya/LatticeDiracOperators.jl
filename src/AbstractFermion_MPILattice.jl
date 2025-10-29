function Initialize_pseudofermion_fields(
    u::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW},
    Dirac_operator::String;
    L5=2,
    nowing=true, kwargs...
) where {NC,NX,NY,NZ,NT,T,AT,NDW}

    #=
        if Dirac_operator == "staggered"
            error(
                "Dirac_operator  = $Dirac_operator witn nowing = $nowing is not supported",
            )
        elseif Dirac_operator == "Wilson"
            x = Initialize_WilsonFermion(u)
        else
            error("Dirac_operator  = $Dirac_operator is not supported")
        end
        return x
        =#
    Dim = 4
    if Dim == 4
        if Dirac_operator == "staggered"
            x = Initialize_StaggeredFermion(u, nowing=nowing)
        elseif Dirac_operator == "Wilson"
            x = Initialize_WilsonFermion(u, nowing=nowing)
        elseif Dirac_operator == "Domainwall"
            #@warn "Domainwall fermion is not well tested!!"
            x = Initialize_DomainwallFermion(u, L5, nowing=nowing)
        elseif Dirac_operator == "MobiusDomainwall"
            #@warn "MobiusDomainwall fermion is not well tested!!"
            x = Initialize_MobiusDomainwallFermion(u, L5; nowing, kwargs...)
        elseif Dirac_operator == "GeneralizedDomainwall"
            #@warn "GeneralizedDomainwall fermion is not well tested!!"
            x = Initialize_GeneralizedDomainwallFermion(u, L5, nowing=nowing)

        else
            error("Dirac_operator = $Dirac_operator is not supported")
        end
    elseif Dim == 2
        if Dirac_operator == "staggered"
            x = Initialize_StaggeredFermion(u)
        elseif Dirac_operator == "Wilson"
            x = Initialize_WilsonFermion(u)
        elseif Dirac_operator == "Domainwall"
            #@warn "Domainwall fermion is not well tested!!"
            x = Initialize_DomainwallFermion(u, L5)
        elseif Dirac_operator == "MobiusDomainwall"
            @warn "MobiusDomainwall fermion is not well tested!!"
            x = Initialize_MobiusDomainwallFermion(u, L5)
        elseif Dirac_operator == "GeneralizedDomainwall"
            @warn "GeneralizedDomainwall fermion is not tested!!"
            x = Initialize_GeneralizedDomainwallFermion(u, L5, nowing=nowing)
        else
            error("Dirac_operator = $Dirac_operator is not supported")
        end
    else
        error("Dim = $Dim is not supported")
    end
    return x

end