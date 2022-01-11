using Wilsonloop

struct General_Dirac_line{Dim}
    coefficient::ComplexF64
    matrix_for_spinors::Matrix{ComplexF64}
    lines_for_gaugefields::Wilsonline{Dim}


end

struct Wilson_GeneralDirac_operator{Dim,T,fermion} <: Dirac_operator{Dim}  where T <: AbstractGaugefields
    U::Array{T,1}
    boundarycondition::Vector{Int8}
    _temporary_fermi::Vector{fermion}
    
    
end