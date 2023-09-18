
struct GeneralDomainwall_Dirac_operator{Dim,T,fermion,L5} <:
    Dirac_operator{Dim} where {T<:AbstractGaugefields}
    b5::Vector{Float64} 
    c5::Vector{Float64}
end

function GeneralDomainwall_Dirac_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    @assert haskey(parameters, "mass") "parameters should have the keyword mass"
    mass = parameters["mass"]
    if haskey(parameters, "b5")
        b5 = parameters["b5"]
    else
        b5 = ones(x.L5)
    end
    if haskey(parameters, "c5")
        c5 = parameters["c5"]
    else
        c5 = ones(x.L5)
    end

    error("GeneralDomainwall is not implemented yet!")
end

function (D::GeneralDomainwall_Dirac_operator{Dim,T,fermion,L5})(
    U,
) where {Dim,T,fermion,L5}
    error("D operator for GeneralDomainwall is not implemented yet!")
end
