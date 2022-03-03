
abstract type WilsonFermion_2D{NC} <: AbstractFermionfields_2D{NC}
end

function Wx!(xout::T,U::Array{G,1},x::T,A)  where  {T <: WilsonFermion_2D,G <: AbstractGaugefields}
    Wx!(xout,U,x,A,2) 
    return
end

function Dx!(xout::T1,U::Array{G,1},x::T2,A) where  {T1 <: WilsonFermion_2D,T2 <: WilsonFermion_2D,G <: AbstractGaugefields}
    Dx!(xout,U,x,A,2)
    return
end

function Ddagx!(xout::T1,U::Array{G,1},x::T2,A) where  {T1 <: WilsonFermion_2D,T2 <: WilsonFermion_2D,G <: AbstractGaugefields}
    Ddagx!(xout,U,x,A,2)
    return
end


function Tx!(xout::T,U::Array{G,1},x::T,A)  where  {T <: WilsonFermion_2D,G <: AbstractGaugefields} # Tx, W = (1 - T)x
    Tx!(xout,U,x,A,2) 
    return
end



function Wdagx!(xout::T,U::Array{G,1},
    x::T,A) where  {T <: WilsonFermion_2D,G <: AbstractGaugefields}
    #,temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_2D,G <: AbstractGaugefields}
    Wdagx!(xout,U,x,A,2)
    return
end
