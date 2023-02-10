const Clover_coefficient  = 1.5612

struct Wilsonclover_data{Dim,gauge} 
    Clover_coefficient::Float64
    internal_flags::Array{Bool,1}
    inn_table::Array{Int64,3}
    _is1::Array{Int64,1}
    _is2::Array{Int64,1}
    Clover_loopterms::Vector{gauge}

    function Wilsonclover_data(D::Dirac_operator{Dim},NV;Clover_coefficient=Clover_coefficient) where Dim
        @assert Dim==4 "Dim /= 4 is not supported!!"

        inn_table= zeros(Int64,NV,4,2)
        internal_flags = zeros(Bool,2)
        _ftmp_vectors = Array{Array{ComplexF64,3},1}(undef,6)
        _is1 = zeros(Int64,NV)
        _is2 = zeros(Int64,NV)

        Clover_loopterms = Array{eltype(D.U)}(undef,6)
        for μν=1:6
            Clover_loopterms[μν] = similar(D.U[1])
        end

        return new{Dim,eltype(D.U)}(Clover_coefficient,internal_flags,inn_table,_is1,_is2,Clover_loopterms)

    end
end