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
#C = C+ α*A + β*B
function add_matrix!(C::LatticeMatrix{5,T,AT,NC1,NC2,nw}, A::LatticeMatrix{5,T1,AT1,NC1,NC2,nw},
    B::LatticeMatrix{5,T1,AT1,NC1,NC2,nw},
    α::S1=1, β::S2=1) where {T,T1,AT,AT1,NC1,NC2,nw,S1<:Number,S2<:Number}
    JACC.parallel_for(prod(C.PN), kernel_add_5D!, C.A, A.A, B.A, C.indexer, Val(NC1), Val(NC2), α, β, Val(nw))
    #set_halo!(C)
end

@inline function kernel_add_5D!(i, u, v, v2, dindexer, ::Val{NC1}, ::Val{NC2}, α, β, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, indices...] += α * v[ic, jc, indices...] + β * v2[ic, jc, indices...]
        end
    end
end


#=
@inline @inbounds function kernel_dot_5D(i, A, B, dindexer, ::Val{3}, ::Val{4}, ::Val{nw}) where {nw}
    ix,iy,iz,it,ik = delinearize(dindexer, i, nw)
    #
    s = zero(eltype(A))
    s = muladd(conj(A[1, 1, ix,iy,iz,it,ik]), B[1, 1, ix,iy,iz,it,ik], s)
    s = muladd(conj(A[2, 1, ix,iy,iz,it,ik]), B[2, 1, ix,iy,iz,it,ik], s)
    s = muladd(conj(A[3, 1, ix,iy,iz,it,ik]), B[3, 1, ix,iy,iz,it,ik], s)

    s = muladd(conj(A[1, 2, ix,iy,iz,it,ik]), B[1, 2, ix,iy,iz,it,ik], s)
    s = muladd(conj(A[2, 2, ix,iy,iz,it,ik]), B[2, 2, ix,iy,iz,it,ik], s)
    s = muladd(conj(A[3, 2, ix,iy,iz,it,ik]), B[3, 2, ix,iy,iz,it,ik], s)

    s = muladd(conj(A[1, 3, ix,iy,iz,it,ik]), B[1, 3, ix,iy,iz,it,ik], s)
    s = muladd(conj(A[2, 3, ix,iy,iz,it,ik]), B[2, 3, ix,iy,iz,it,ik], s)
    s = muladd(conj(A[3, 3, ix,iy,iz,it,ik]), B[3, 3, ix,iy,iz,it,ik], s)

    s = muladd(conj(A[1, 1, ix,iy,iz,it,ik]), B[1, 4, ix,iy,iz,it,ik], s)
    s = muladd(conj(A[2, 1, ix,iy,iz,it,ik]), B[2, 4, ix,iy,iz,it,ik], s)
    s = muladd(conj(A[3, 1, ix,iy,iz,it,ik]), B[3, 4, ix,iy,iz,it,ik], s)

    return s
end
=#


#C = A B 
function LinearAlgebra.mul!(C::LatticeMatrix{5,T1,AT1,NC1,NC2,nw,DIC},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw,DIA}, 
    B::LatticeMatrix{5,T3,AT3,NC3,NC2,nw,DIB}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DIC,DIA,DIB}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_545!, C.A, A.A, B.A, Val(NC1), Val(NC2),Val(NC3),Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_545!(i, C, A, B, ::Val{NC1}, ::Val{NC2},::Val{NC3},::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    ix,iy,iz,it,i5 = indices
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC3
            b = B[kc, jc, indices...]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, ix,iy,iz,it] * b# B[kc, jc, indices...]
            end
        end
    end
end

#=

@inline function kernel_Dmatrix_mul_545!(i, C, A, B, ::Val{3}, ::Val{4},::Val{3},::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    ix,iy,iz,it,i5 = indices
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, ix,iy,iz,it]
        a21 = A[2, 1, ix,iy,iz,it]
        a31 = A[3, 1, ix,iy,iz,it]
        a12 = A[1, 2, ix,iy,iz,it]
        a22 = A[2, 2, ix,iy,iz,it]
        a32 = A[3, 2, ix,iy,iz,it]
        a13 = A[1, 3, ix,iy,iz,it]
        a23 = A[2, 3, ix,iy,iz,it]
        a33 = A[3, 3, ix,iy,iz,it]

        b11 = B[1, 1, indices...]
        b21 = B[2, 1, indices...]
        b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]
        b22 = B[2, 2, indices...]
        b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]
        b23 = B[2, 3, indices...]
        b33 = B[3, 3, indices...]
        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

=#

#C = A shiftedB 
function LinearAlgebra.mul!(C::TC,
    A::TA, B::TB) where {
        D,T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DI5,DI4,
        L<:LatticeMatrix{5,T3,AT3,NC3,NC2,nw,DI5},
        TC<:LatticeMatrix{5,T1,AT1,NC1,NC2,nw,DI5},
        TA<:LatticeMatrix{4,T2,AT2,NC1,NC3,nw,DI4},
        TB<:Shifted_Lattice{L,D},
        }

    shift = get_shift(B)
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_54shift5!, C.A, A.A, B.data.A, Val(NC1), Val(NC2),Val(NC3),Val(nw), C.indexer,shift
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_54shift5!(i, C, A, B, ::Val{NC1},  ::Val{NC2}, ::Val{NC3},::Val{nw}, dindexer,shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)

    ix,iy,iz,it,i5 = indices
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC3
            b = B[kc, jc, indices_p...]
            for ic = 1:NC1
                C[ic, jc, indices...] += A[ic, kc, ix,iy,iz,it] * b# B[kc, jc, indices...]
            end
        end
    end
end

#=

@inline function kernel_Dmatrix_mul_54shift5!(i, C, A, B, ::Val{3},  ::Val{4}, ::Val{3},::Val{nw}, dindexer,shift) where {nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    ix,iy,iz,it,i5 = indices
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, ix,iy,iz,it]
        a21 = A[2, 1, ix,iy,iz,it]
        a31 = A[3, 1, ix,iy,iz,it]
        a12 = A[1, 2, ix,iy,iz,it]
        a22 = A[2, 2, ix,iy,iz,it]
        a32 = A[3, 2, ix,iy,iz,it]
        a13 = A[1, 3, ix,iy,iz,it]
        a23 = A[2, 3, ix,iy,iz,it]
        a33 = A[3, 3, ix,iy,iz,it]

        b11 = B[1, 1, indices_p...]
        b21 = B[2, 1, indices_p...]
        b31 = B[3, 1, indices_p...]
        b12 = B[1, 2, indices_p...]
        b22 = B[2, 2, indices_p...]
        b32 = B[3, 2, indices_p...]
        b13 = B[1, 3, indices_p...]
        b23 = B[2, 3, indices_p...]
        b33 = B[3, 3, indices_p...]
        C[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

=#

function LinearAlgebra.mul!(C::LatticeMatrix{5,T1,AT1,NC1,4,nw},
    A::TA) where {T1,AT1,NC1,nw,TA<:AbstractMatrix}
    At = JACC.array(A)
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mulA!, C.A, At, Val(NC1), Val(nw), C.indexer
    )

end


#=
#C = A Bdag 
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw,DIC},
    A::LatticeMatrix{5,T2,AT2,NC1,NC3,nw,DIA}, 
    B::Adjoint_Lattice{L}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DIC,DIA,DIB,L<:LatticeMatrix{5,T3,AT3,NC2,NC3,nw,DIB}}

    clear_matrix!(C)

    JACC.parallel_for(
        prod(A.PN), kernel_Dmatrix_mul_455dag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2),Val(NC3),Val(nw), A.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_455dag!(i, C, A, B, ::Val{NC1}, ::Val{NC2},::Val{NC3},::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    ix,iy,iz,it,i5 = indices
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = conj(B[jc, kc, ix,iy,iz,it,i5])
            for ic = 1:NC1
                C[ic, jc, ix,iy,iz,it] += A[ic, kc, ix,iy,iz,it,i5] * b# B[kc, jc, indices...]
            end
        end
    end
end

=#

#=

@inline function kernel_Dmatrix_mul_455dag!(i, C, A, B, ::Val{3}, ::Val{4},::Val{3},::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    ix,iy,iz,it,i5 = indices
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, ix,iy,iz,it,i5]
        a21 = A[2, 1, ix,iy,iz,it,i5]
        a31 = A[3, 1, ix,iy,iz,it,i5]
        a12 = A[1, 2, ix,iy,iz,it,i5]
        a22 = A[2, 2, ix,iy,iz,it,i5]
        a32 = A[3, 2, ix,iy,iz,it,i5]
        a13 = A[1, 3, ix,iy,iz,it,i5]
        a23 = A[2, 3, ix,iy,iz,it,i5]
        a33 = A[3, 3, ix,iy,iz,it,i5]

        b11 = conj(B[1, 1, indices...])
        b21 = conj(B[1, 2, indices...])
        b31 = conj(B[1, 3, indices...])
        b12 = conj(B[2, 1, indices...])
        b22 = conj(B[2, 2, indices...])
        b32 = conj(B[2, 3, indices...])
        b13 = conj(B[3, 1, indices...])
        b23 = conj(B[3, 2, indices...])
        b33 = conj(B[3, 3, indices...])
        C[1, 1,  ix,iy,iz,it] += a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1,  ix,iy,iz,it] += a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1,  ix,iy,iz,it] += a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2,  ix,iy,iz,it] += a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2,  ix,iy,iz,it] += a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2,  ix,iy,iz,it] += a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3,  ix,iy,iz,it] += a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3,  ix,iy,iz,it] += a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3,  ix,iy,iz,it] += a31 * b13 + a32 * b23 + a33 * b33
    end
end

=#



#C = Adag Bdag 
function LinearAlgebra.mul!(C::LatticeMatrix{5,T1,AT1,NC1,NC2,nw,DIC},
    A::Adjoint_Lattice{L1}, 
    B::Adjoint_Lattice{L2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DIC,DIA,DIB,
    L1<:LatticeMatrix{5,T2,AT2,NC3,NC2,nw,DIA},L2<:LatticeMatrix{4,T3,AT3,NC1,NC3,nw,DIB}}

    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_55dag4dag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2),Val(NC3),Val(nw), C.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_55dag4dag!(i, C, A, B, ::Val{NC1}, ::Val{NC2},::Val{NC3},::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    ix,iy,iz,it,i5 = indices

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC3
            for ic = 1:NC1
                C[ic,jc,ix,iy,iz,it,i5] +=  conj(A[kc,jc,ix,iy,iz,it,i5])*conj(B[ic, kc, ix,iy,iz,it])
            end
        end

    end

end

#=

@inline function kernel_Dmatrix_mul_55dag4dag!(i, C, A, B, ::Val{3}, ::Val{4},::Val{3},::Val{nw}, dindexer) where {nw}
    indices = delinearize(dindexer, i, nw)
    ix,iy,iz,it,i5 = indices
    #    ix += nw
    #    iy += nw
    #    iz += nw
    #    it += nw

    @inbounds begin
        a11 = A[1, 1, ix,iy,iz,it,i5]
        a21 = A[2, 1, ix,iy,iz,it,i5]
        a31 = A[3, 1, ix,iy,iz,it,i5]
        a12 = A[1, 2, ix,iy,iz,it,i5]
        a22 = A[2, 2, ix,iy,iz,it,i5]
        a32 = A[3, 2, ix,iy,iz,it,i5]
        a13 = A[1, 3, ix,iy,iz,it,i5]
        a23 = A[2, 3, ix,iy,iz,it,i5]
        a33 = A[3, 3, ix,iy,iz,it,i5]

        b11 = conj(B[1, 1, indices...])
        b21 = conj(B[1, 2, indices...])
        b31 = conj(B[1, 3, indices...])
        b12 = conj(B[2, 1, indices...])
        b22 = conj(B[2, 2, indices...])
        b32 = conj(B[2, 3, indices...])
        b13 = conj(B[3, 1, indices...])
        b23 = conj(B[3, 2, indices...])
        b33 = conj(B[3, 3, indices...])
        C[1, 1,  ix,iy,iz,it] += a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1,  ix,iy,iz,it] += a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1,  ix,iy,iz,it] += a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2,  ix,iy,iz,it] += a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2,  ix,iy,iz,it] += a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2,  ix,iy,iz,it] += a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3,  ix,iy,iz,it] += a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3,  ix,iy,iz,it] += a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3,  ix,iy,iz,it] += a31 * b13 + a32 * b23 + a33 * b33
    end
end

=#

#C = Adagshift Bdag 
function LinearAlgebra.mul!(C::LatticeMatrix{5,T1,AT1,NC1,NC2,nw,DIC},
    A::Adjoint_Lattice{Shifted_Lattice{L1,5}}, 
    B::Adjoint_Lattice{L2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DIC,DIA,DIB,
    L1<:LatticeMatrix{5,T2,AT2,NC3,NC2,nw,DIA},L2<:LatticeMatrix{4,T3,AT3,NC1,NC3,nw,DIB}}

    shift = get_shift(A)
    JACC.parallel_for(
        prod(C.PN), kernel_Dmatrix_mul_55dag4dag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2),Val(NC3),Val(nw), 
        C.indexer,shift
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_55dag4dag!(i, C, A, B, ::Val{NC1}, ::Val{NC2},::Val{NC3},::Val{nw},
     dindexer,shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    ix,iy,iz,it,i5 = indices

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, indices...] = zero(eltype(C))
        end

        for kc = 1:NC3
            for ic = 1:NC1
                C[ic,jc,ix,iy,iz,it,i5] +=  conj(A[kc,jc,indices_p...])*conj(B[ic, kc, ix,iy,iz,it])
            end
        end

    end

end

#C = A B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw,DIC},
    A::LatticeMatrix{5,T2,AT2,NC1,NC3,nw,DIA}, 
    B::L) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,DIC,DIA,DIB,L<:LatticeMatrix{5,T3,AT3,NC2,NC3,nw,DIB}}

    clear_matrix!(C)

    JACC.parallel_for(
        prod(A.PN), kernel_Dmatrix_mul_455!, C.A, A.A, B.A, Val(NC1), Val(NC2),Val(NC3),Val(nw), A.indexer
    )
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mul_455!(i, C, A, B, ::Val{NC1}, ::Val{NC2},::Val{NC3},::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    ix,iy,iz,it,i5 = indices
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            b = B[jc, kc, ix,iy,iz,it,i5]
            for ic = 1:NC1
                C[ic, jc, ix,iy,iz,it] += A[ic, kc, ix,iy,iz,it,i5] * b# B[kc, jc, indices...]
            end
        end
    end
end
