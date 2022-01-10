struct WilsonClover
    clover_coefficient::Float64
    internal_flags::Array{Bool,1}
    inn_table::Array{Int64,3}
    _ftmp_vectors::Array{Array{ComplexF64,3},1}
    _is1::Array{Int64,1}
    _is2::Array{Int64,1}

    function WilsonClover(U::Array{<: AbstractGaugefields{NC,Dim},1},x,clover_coefficient) where {NC,Dim}
        _,_,NN... = size(U[1])
        NV = prod(NN)
        inn_table= zeros(Int64,NV,4,2)
        internal_flags = zeros(Bool,2)
        _ftmp_vectors = Array{Array{ComplexF64,3},1}(undef,6)
        for i=1:6
            _ftmp_vectors[i] = zeros(ComplexF64,NC,NV,4)
        end

        _is1 = zeros(Int64,NV)
        _is2 = zeros(Int64,NV)

        return new(clover_coefficient,internal_flags,inn_table,_ftmp_vectors,_is1,_is2)
    end
end