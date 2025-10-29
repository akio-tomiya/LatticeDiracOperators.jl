function LinearAlgebra.dot(A::LatticeMatrix{5,T1,AT1,NC1,NG,nw}, B::LatticeMatrix{5,T2,AT2,NC1,NG,nw}) where {NG,T1,T2,AT1,AT2,NC1,nw}
    s = JACC.parallel_reduce(prod(A.PN), +, kernel_dot_5D,
        A.A, B.A, A.indexer, Val(NC1), Val(NG), Val(nw); init=zero(eltype(A.A)))
end

@inline function kernel_dot_5D(i, A, B, dindexer, ::Val{NC1}, ::Val{NG}, ::Val{nw}) where {NC1,nw,NG}
    indices = delinearize(dindexer, i, nw)
    s = zero(eltype(A))

    @inbounds for ialpha = 1:NG
        for ic = 1:NC1
            s += conj(A[ic, ialpha, indices...]) * B[ic, ialpha, indices...]
        end
    end
    return s
end

