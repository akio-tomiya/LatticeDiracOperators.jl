struct Fermionvector{fermion} <: AbstractVector{ComplexF64}
    data::fermion
    function Fermionvector(x::T) where T
        return new{T}(x)
    end
end

Base.eltype(x::Fermionvector{fermion}) where fermion = ComplexF64

function Base.size(x::Fermionvector{fermion}) where fermion
    NN = size(x.data)
    return (prod(NN),)
end

function Base.getindex(x::Fermionvector{fermion},i::Int) where fermion
    return x.data[i]
end

function Base.setindex!(x::Fermionvector{fermion}, v, i) where {fermion}
    x.data[i] = v
end

function Base.similar(x::Fermionvector{fermion}) where fermion
    return similar(x.data)
end

using LinearAlgebra
function LinearAlgebra.zero(x::Fermionvector{fermion}) where fermion
    return Fermionvector(similar(x))
end

function LinearAlgebra.dot(a::Fermionvector{fermion},b::Fermionvector{fermion}) where fermion
    return dot(a.data,b.data)
end

function LinearAlgebra.norm(a::Fermionvector{fermion}) where fermion
    return sqrt(real(dot(a.data,a.data)))
end

import .Dirac_operators:Dirac_operator,get_temporaryvectors_forCG,γ5D_operator

function Base.:*(A::Dirac_operator,x::Fermionvector{fermion}) where fermion
    y = zero(x)
    mul!(y.data,A,x.data)
    return y
end

function Base.:*(A::γ5D_operator,x::Fermionvector{fermion}) where fermion
    y = zero(x)
    mul!(y.data,A,x.data)
    return y
end





