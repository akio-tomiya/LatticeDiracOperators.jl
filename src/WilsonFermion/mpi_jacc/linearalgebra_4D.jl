import Gaugefields.MPILattice: add_matrix!, kernel_add_4D!


#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    a::Number,
    X::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    b::Number,
    Y::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
) where {T1,AT1,NC1,NC2,nw}

    JACC.parallel_for(
        prod(Y.PN), kernel_4D_axpby!, a, X.A, b, Y.A, Val(NC1), Val(NC2), Val(nw), Y.PN
    )
end

@inline function kernel_4D_axpby!(i, a, X, b, Y, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            Y[ic, jc, ix, iy, iz, it] = a * X[ic, jc, ix, iy, iz, it] + b * Y[ic, jc, ix, iy, iz, it]
        end
    end
end

#C = A*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,4,nw},
    A::TA) where {T1,AT1,NC1,nw,TA<:AbstractMatrix}
    #_, NC3 = size(A)
    #println("mul")
    #display(C.A[:, :, 2, 2, 2, 2])
    #println("before")
    #n1, n2 = size(A)
    #At = zeros(eltype(A), n1, n2)
    #At .= A
    At = JACC.array(A)
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mulA!, C.A, At, Val(NC1), Val(nw), C.PN
    )
    #display(C.A[:, :, 2, 2, 2, 2])
    #set_halo!(C)
end

function kernel_4Dmatrix_mulA!(i, C, A, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        e1 = C[ic, 1, ix, iy, iz, it]
        e2 = C[ic, 2, ix, iy, iz, it]
        e3 = C[ic, 3, ix, iy, iz, it]
        e4 = C[ic, 4, ix, iy, iz, it]

        C[ic, 1, ix, iy, iz, it] =
            A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
        C[ic, 2, ix, iy, iz, it] =
            A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
        C[ic, 3, ix, iy, iz, it] =
            A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
        C[ic, 4, ix, iy, iz, it] =
            A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4


        #row = ntuple(kc -> C[ic, kc, ix, iy, iz, it], 4)
        #for jc = 1:4
        #    acc = zero(eltype(C))
        #    @inbounds for kc = 1:4
        ##        acc += row[kc] * A[jc, kc]   # = dot(C[ic,:], A[jc,:])
        #    end
        #    C[ic, jc, ix, iy, iz, it] = acc
        #end
    end
    return

end



#C = x*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,4,nw},
    A::TA, x::LatticeMatrix{4,T1,AT1,NC1,4,nw}) where {T1,AT1,NC1,nw,TA<:AbstractMatrix}


    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mulxAT!, C.A, A, x.A, Val(NC1), Val(nw), C.PN
    )
    #set_halo!(C)
end

function kernel_4Dmatrix_mulxAT!(i, C, A, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        row = ntuple(kc -> x[ic, kc, ix, iy, iz, it], 4)
        for jc = 1:4
            acc = zero(eltype(C))
            @inbounds for kc = 1:4
                acc += row[kc] * A[jc, kc]   # = dot(C[ic,:], A[jc,:])
            end
            C[ic, jc, ix, iy, iz, it] = acc
        end
    end
    return
end


function kernel_4Dmatrix_mulxAT!(i, C, A, x, ::Val{3}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:3
        row = ntuple(kc -> x[ic, kc, ix, iy, iz, it], 4)
        for jc = 1:4
            acc = zero(eltype(C))
            @inbounds for kc = 1:4
                acc += row[kc] * A[jc, kc]   # = dot(C[ic,:], A[jc,:])
            end
            C[ic, jc, ix, iy, iz, it] = acc
        end
    end
    return
end

#C = C+ α*A + β*B
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    B::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    α::S1=1, β::S2=1) where {T,T1,AT,AT1,NC1,NC2,nw,S1<:Number,S2<:Number}
    JACC.parallel_for(prod(C.PN), kernel_add_4D!, C.A, A.A, B.A, C.PN, Val(NC1), Val(NC2), α, β, Val(nw))
    #set_halo!(C)
end

@inline function kernel_add_4D!(i, u, v, v2, PN, ::Val{NC1}, ::Val{NC2}, α, β, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] += α * v[ic, jc, ix, iy, iz, it] + β * v2[ic, jc, ix, iy, iz, it]
        end
    end
end

function LinearAlgebra.dot(A::LatticeMatrix{4,T1,AT1,NC1,NG,nw}, B::LatticeMatrix{4,T2,AT2,NC1,NG,nw}) where {NG,T1,T2,AT1,AT2,NC1,nw}
    s = JACC.parallel_reduce(prod(A.PN), +, kernel_dot_4D,
        A.A, B.A, A.PN, Val(NC1), Val(NG), Val(nw); init=zero(eltype(A.A)))
end

@inline function kernel_dot_4D(i, A, B, PN, ::Val{NC1}, ::Val{NG}, ::Val{nw}) where {NC1,nw,NG}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    s = zero(eltype(A))

    @inbounds for ialpha = 1:NG
        for ic = 1:NC1
            s += conj(A[ic, ialpha, ix, iy, iz, it]) * B[ic, ialpha, ix, iy, iz, it]
        end
    end
    return s
end

Base.any(isnan, f::LatticeMatrix) = any(isnan, f.A)


#C = a*x
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NG,nw},
    a::TA, x::LatticeMatrix{4,T1,AT1,NC1,NG,nw}) where {T1,AT1,NC1,nw,NG,TA<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mulsx!, C.A, a, x.A, Val(NC1), Val(NG), Val(nw), C.PN
    )
    #set_halo!(C)
end

function kernel_4Dmatrix_mulsx!(i, C, a, x, ::Val{NC1}, ::Val{NG}, ::Val{nw}, PN) where {NC1,NG,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    for ig = 1:NG
        for ic = 1:NC1
            C[ic, ig, ix, iy, iz, it] = a * x[ic, ig, ix, iy, iz, it]
        end
    end
    return
end

struct Oneγμ{sign,μ} <: AbstractMatrix{Complex{Int64}}
end

JACC.array(A::Oneγμ{sign,μ}) where {sign,μ} = A

Base.size(::Oneγμ{sign,μ}) where {sign,μ} = (4, 4)


function kernel_4Dmatrix_mulxAT!(i, C, ::Oneγμ{:plus,1}, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ix, iy, iz, it]
        x2 = x[ic, 2, ix, iy, iz, it]
        x3 = x[ic, 3, ix, iy, iz, it]
        x4 = x[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 - im * x4
        C[ic, 2, ix, iy, iz, it] = x2 - im * x3
        C[ic, 3, ix, iy, iz, it] = x3 + im * x2
        C[ic, 4, ix, iy, iz, it] = x4 + im * x1
    end

    return
end

function kernel_4Dmatrix_mulxAT!(i, C, ::Oneγμ{:minus,1}, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ix, iy, iz, it]
        x2 = x[ic, 2, ix, iy, iz, it]
        x3 = x[ic, 3, ix, iy, iz, it]
        x4 = x[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 + im * x4
        C[ic, 2, ix, iy, iz, it] = x2 + im * x3
        C[ic, 3, ix, iy, iz, it] = x3 - im * x2
        C[ic, 4, ix, iy, iz, it] = x4 - im * x1
    end

    return
end

function kernel_4Dmatrix_mulxAT!(i, C, ::Oneγμ{:plus,2}, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ix, iy, iz, it]
        x2 = x[ic, 2, ix, iy, iz, it]
        x3 = x[ic, 3, ix, iy, iz, it]
        x4 = x[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 - x4
        C[ic, 2, ix, iy, iz, it] = x2 + x3
        C[ic, 3, ix, iy, iz, it] = x3 + x2
        C[ic, 4, ix, iy, iz, it] = x4 - x1
    end

    return
end

function kernel_4Dmatrix_mulxAT!(i, C, ::Oneγμ{:minus,2}, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ix, iy, iz, it]
        x2 = x[ic, 2, ix, iy, iz, it]
        x3 = x[ic, 3, ix, iy, iz, it]
        x4 = x[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 + x4
        C[ic, 2, ix, iy, iz, it] = x2 - x3
        C[ic, 3, ix, iy, iz, it] = x3 - x2
        C[ic, 4, ix, iy, iz, it] = x4 + x1
    end

    return
end


function kernel_4Dmatrix_mulxAT!(i, C, ::Oneγμ{:plus,3}, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ix, iy, iz, it]
        x2 = x[ic, 2, ix, iy, iz, it]
        x3 = x[ic, 3, ix, iy, iz, it]
        x4 = x[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 - im * x3
        C[ic, 2, ix, iy, iz, it] = x2 + im * x4
        C[ic, 3, ix, iy, iz, it] = x3 + im * x1
        C[ic, 4, ix, iy, iz, it] = x4 - im * x2
    end

    return
end

function kernel_4Dmatrix_mulxAT!(i, C, ::Oneγμ{:minus,3}, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ix, iy, iz, it]
        x2 = x[ic, 2, ix, iy, iz, it]
        x3 = x[ic, 3, ix, iy, iz, it]
        x4 = x[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 + im * x3
        C[ic, 2, ix, iy, iz, it] = x2 - im * x4
        C[ic, 3, ix, iy, iz, it] = x3 - im * x1
        C[ic, 4, ix, iy, iz, it] = x4 + im * x2
    end

    return
end

function kernel_4Dmatrix_mulxAT!(i, C, ::Oneγμ{:plus,4}, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ix, iy, iz, it]
        x2 = x[ic, 2, ix, iy, iz, it]
        x3 = x[ic, 3, ix, iy, iz, it]
        x4 = x[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 - x3
        C[ic, 2, ix, iy, iz, it] = x2 - x4
        C[ic, 3, ix, iy, iz, it] = x3 - x1
        C[ic, 4, ix, iy, iz, it] = x4 - x2
    end

    return
end

function kernel_4Dmatrix_mulxAT!(i, C, ::Oneγμ{:minus,4}, x, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ix, iy, iz, it]
        x2 = x[ic, 2, ix, iy, iz, it]
        x3 = x[ic, 3, ix, iy, iz, it]
        x4 = x[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 + x3
        C[ic, 2, ix, iy, iz, it] = x2 + x4
        C[ic, 3, ix, iy, iz, it] = x3 + x1
        C[ic, 4, ix, iy, iz, it] = x4 + x2
    end

    return
end


#U = U + x*y
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    x::LatticeMatrix{4,T1,AT1,NC1,4,nw}, y::LatticeMatrix{4,T1,AT1,NC2,4,nw}) where {T1,AT1,NC1,NC2,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_muluxy!, C.A, x.A, y.A, Val(NC1), Val(NC2), Val(nw), C.PN
    )
    #set_halo!(C)
end

function kernel_4Dmatrix_muluxy!(i, C, x, y, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds for jc = 1:NC2
        for ic = 1:NC2
            for ik = 1:4
                C[ic, jc, ix, iy, iz, it] += x[ic, ik, ix, iy, iz, it] * y[jc, ik, ix, iy, iz, it]
            end
        end
    end

end

#U = U + x'*y
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    xdag::Adjoint_Lattice{LatticeMatrix{4,T1,AT1,NC1,4,nw}}, y::LatticeMatrix{4,T1,AT1,NC2,4,nw}) where {T1,AT1,NC1,NC2,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_muluxdagy!, C.A, xdag.data.A, y.A, Val(NC1), Val(NC2), Val(nw), C.PN
    )
    #set_halo!(C)
end


function kernel_4Dmatrix_muluxdagy!(i, C, x, y, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds for ik = 1:4
        for jc = 1:NC2
            for ic = 1:NC2

                C[ic, jc, ix, iy, iz, it] += conj(x[ic, ik, ix, iy, iz, it]) * y[jc, ik, ix, iy, iz, it]
            end
        end
    end

end

#U = U + x*y'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    x::LatticeMatrix{4,T1,AT1,NC1,4,nw}, ydag::Adjoint_Lattice{LatticeMatrix{4,T1,AT1,NC2,4,nw}}) where {T1,AT1,NC1,NC2,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_muluxydag!, C.A, x.A, ydag.data.A, Val(NC1), Val(NC2), Val(nw), C.PN
    )
    #set_halo!(C)
end


function kernel_4Dmatrix_muluxydag!(i, C, x, y, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds for ik = 1:4
        for jc = 1:NC2
            for ic = 1:NC2

                C[ic, jc, ix, iy, iz, it] += x[ic, ik, ix, iy, iz, it] * conj(y[jc, ik, ix, iy, iz, it])

            end
        end
    end

end

#y = x*U
function LinearAlgebra.mul!(y::LatticeMatrix{4,T1,AT1,NC1,4,nw},
    x::LatticeMatrix{4,T1,AT1,NC2,4,nw}, U::LatticeMatrix{4,T1,AT1,NC2,NC1,nw}) where {T1,AT1,NC1,NC2,nw}

    JACC.parallel_for(
        prod(y.PN), kernel_4Dmatrix_mulyxU!, y.A, x.A, U.A, Val(NC1), Val(NC2), Val(nw), y.PN
    )
    #set_halo!(C)
end

function kernel_4Dmatrix_mulyxU!(i, y, x, U, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    for ik = 1:4
        for ic = 1:NC1
            y[ic, ik, ix, iy, iz, it] = zero(eltype(y))
            for jc = 1:NC2
                y[ic, ik, ix, iy, iz, it] += x[jc, ik, ix, iy, iz, it] * U[jc, ic, ix, iy, iz, it]
            end
        end
    end

end

#y = xdagshifted*Udag
function LinearAlgebra.mul!(y::LatticeMatrix{4,T1,AT1,NC1,4,nw},
    x::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T1,AT1,NC2,4,nw},shift}},
    U::Adjoint_Lattice{LatticeMatrix{4,T1,AT1,NC2,NC1,nw}}) where {T1,AT1,NC1,NC2,nw,shift}

    JACC.parallel_for(
        prod(y.PN), kernel_4Dmatrix_mulyxdagshiftedUdag!, y.A, x.data.data.A, U.data.A, Val(NC1), Val(NC2), Val(nw), y.PN, shift
    )
    #set_halo!(C)
end

function kernel_4Dmatrix_mulyxdagshiftedUdag!(i, y, x, U, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN, shift) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for ik = 1:4
        for ic = 1:NC1
            y[ic, ik, ix, iy, iz, it] = zero(eltype(y))
            for jc = 1:NC2
                y[ic, ik, ix, iy, iz, it] += conj(x[jc, ik, ixp, iyp, izp, itp]) * conj(U[ic, jc, ix, iy, iz, it])
            end
        end
    end

end





#C = xshifted*A^T
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,4,nw},
    A::TA, x::Shifted_Lattice{LatticeMatrix{4,T1,AT1,NC1,4,nw},shift}) where {T1,AT1,NC1,nw,TA<:AbstractMatrix,shift}


    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mulxshiftedAT!, C.A, A, x.data.A, Val(NC1), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end



function kernel_4Dmatrix_mulxshiftedAT!(i, C, ::Oneγμ{:plus,1}, x, ::Val{NC1}, ::Val{nw}, PN, shift) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ixp, iyp, izp, itp]
        x2 = x[ic, 2, ixp, iyp, izp, itp]
        x3 = x[ic, 3, ixp, iyp, izp, itp]
        x4 = x[ic, 4, ixp, iyp, izp, itp]
        C[ic, 1, ix, iy, iz, it] = x1 - im * x4
        C[ic, 2, ix, iy, iz, it] = x2 - im * x3
        C[ic, 3, ix, iy, iz, it] = x3 + im * x2
        C[ic, 4, ix, iy, iz, it] = x4 + im * x1
    end

    return
end

function kernel_4Dmatrix_mulxshiftedAT!(i, C, ::Oneγμ{:minus,1}, x, ::Val{NC1}, ::Val{nw}, PN, shift) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ixp, iyp, izp, itp]
        x2 = x[ic, 2, ixp, iyp, izp, itp]
        x3 = x[ic, 3, ixp, iyp, izp, itp]
        x4 = x[ic, 4, ixp, iyp, izp, itp]
        C[ic, 1, ix, iy, iz, it] = x1 + im * x4
        C[ic, 2, ix, iy, iz, it] = x2 + im * x3
        C[ic, 3, ix, iy, iz, it] = x3 - im * x2
        C[ic, 4, ix, iy, iz, it] = x4 - im * x1
    end

    return
end

function kernel_4Dmatrix_mulxshiftedAT!(i, C, ::Oneγμ{:plus,2}, x, ::Val{NC1}, ::Val{nw}, PN, shift) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ixp, iyp, izp, itp]
        x2 = x[ic, 2, ixp, iyp, izp, itp]
        x3 = x[ic, 3, ixp, iyp, izp, itp]
        x4 = x[ic, 4, ixp, iyp, izp, itp]
        C[ic, 1, ix, iy, iz, it] = x1 - x4
        C[ic, 2, ix, iy, iz, it] = x2 + x3
        C[ic, 3, ix, iy, iz, it] = x3 + x2
        C[ic, 4, ix, iy, iz, it] = x4 - x1
    end

    return
end

function kernel_4Dmatrix_mulxshiftedAT!(i, C, ::Oneγμ{:minus,2}, x, ::Val{NC1}, ::Val{nw}, PN, shift) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ixp, iyp, izp, itp]
        x2 = x[ic, 2, ixp, iyp, izp, itp]
        x3 = x[ic, 3, ixp, iyp, izp, itp]
        x4 = x[ic, 4, ixp, iyp, izp, itp]
        C[ic, 1, ix, iy, iz, it] = x1 + x4
        C[ic, 2, ix, iy, iz, it] = x2 - x3
        C[ic, 3, ix, iy, iz, it] = x3 - x2
        C[ic, 4, ix, iy, iz, it] = x4 + x1
    end

    return
end


function kernel_4Dmatrix_mulxshiftedAT!(i, C, ::Oneγμ{:plus,3}, x, ::Val{NC1}, ::Val{nw}, PN, shift) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ixp, iyp, izp, itp]
        x2 = x[ic, 2, ixp, iyp, izp, itp]
        x3 = x[ic, 3, ixp, iyp, izp, itp]
        x4 = x[ic, 4, ixp, iyp, izp, itp]
        C[ic, 1, ix, iy, iz, it] = x1 - im * x3
        C[ic, 2, ix, iy, iz, it] = x2 + im * x4
        C[ic, 3, ix, iy, iz, it] = x3 + im * x1
        C[ic, 4, ix, iy, iz, it] = x4 - im * x2
    end

    return
end

function kernel_4Dmatrix_mulxshiftedAT!(i, C, ::Oneγμ{:minus,3}, x, ::Val{NC1}, ::Val{nw}, PN, shift) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ixp, iyp, izp, itp]
        x2 = x[ic, 2, ixp, iyp, izp, itp]
        x3 = x[ic, 3, ixp, iyp, izp, itp]
        x4 = x[ic, 4, ixp, iyp, izp, itp]
        C[ic, 1, ix, iy, iz, it] = x1 + im * x3
        C[ic, 2, ix, iy, iz, it] = x2 - im * x4
        C[ic, 3, ix, iy, iz, it] = x3 - im * x1
        C[ic, 4, ix, iy, iz, it] = x4 + im * x2
    end

    return
end

function kernel_4Dmatrix_mulxshiftedAT!(i, C, ::Oneγμ{:plus,4}, x, ::Val{NC1}, ::Val{nw}, PN, shift) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ixp, iyp, izp, itp]
        x2 = x[ic, 2, ixp, iyp, izp, itp]
        x3 = x[ic, 3, ixp, iyp, izp, itp]
        x4 = x[ic, 4, ixp, iyp, izp, itp]
        C[ic, 1, ix, iy, iz, it] = x1 - x3
        C[ic, 2, ix, iy, iz, it] = x2 - x4
        C[ic, 3, ix, iy, iz, it] = x3 - x1
        C[ic, 4, ix, iy, iz, it] = x4 - x2
    end

    return
end

function kernel_4Dmatrix_mulxshiftedAT!(i, C, ::Oneγμ{:minus,4}, x, ::Val{NC1}, ::Val{nw}, PN, shift) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for ic = 1:NC1
        x1 = x[ic, 1, ixp, iyp, izp, itp]
        x2 = x[ic, 2, ixp, iyp, izp, itp]
        x3 = x[ic, 3, ixp, iyp, izp, itp]
        x4 = x[ic, 4, ixp, iyp, izp, itp]
        C[ic, 1, ix, iy, iz, it] = x1 + x3
        C[ic, 2, ix, iy, iz, it] = x2 + x4
        C[ic, 3, ix, iy, iz, it] = x3 + x1
        C[ic, 4, ix, iy, iz, it] = x4 + x2
    end

    return
end



function kernel_4Dmatrix_mulA!(i, C, ::Oneγμ{:plus,1}, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = C[ic, 1, ix, iy, iz, it]
        x2 = C[ic, 2, ix, iy, iz, it]
        x3 = C[ic, 3, ix, iy, iz, it]
        x4 = C[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 - im * x4
        C[ic, 2, ix, iy, iz, it] = x2 - im * x3
        C[ic, 3, ix, iy, iz, it] = x3 + im * x2
        C[ic, 4, ix, iy, iz, it] = x4 + im * x1
    end

    return
end

function kernel_4Dmatrix_mulA!(i, C, ::Oneγμ{:minus,1}, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = C[ic, 1, ix, iy, iz, it]
        x2 = C[ic, 2, ix, iy, iz, it]
        x3 = C[ic, 3, ix, iy, iz, it]
        x4 = C[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 + im * x4
        C[ic, 2, ix, iy, iz, it] = x2 + im * x3
        C[ic, 3, ix, iy, iz, it] = x3 - im * x2
        C[ic, 4, ix, iy, iz, it] = x4 - im * x1
    end

    return
end

function kernel_4Dmatrix_mulA!(i, C, ::Oneγμ{:plus,2}, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = C[ic, 1, ix, iy, iz, it]
        x2 = C[ic, 2, ix, iy, iz, it]
        x3 = C[ic, 3, ix, iy, iz, it]
        x4 = C[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 - x4
        C[ic, 2, ix, iy, iz, it] = x2 + x3
        C[ic, 3, ix, iy, iz, it] = x3 + x2
        C[ic, 4, ix, iy, iz, it] = x4 - x1
    end

    return
end

function kernel_4Dmatrix_mulA!(i, C, ::Oneγμ{:minus,2}, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = C[ic, 1, ix, iy, iz, it]
        x2 = C[ic, 2, ix, iy, iz, it]
        x3 = C[ic, 3, ix, iy, iz, it]
        x4 = C[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 + x4
        C[ic, 2, ix, iy, iz, it] = x2 - x3
        C[ic, 3, ix, iy, iz, it] = x3 - x2
        C[ic, 4, ix, iy, iz, it] = x4 + x1
    end

    return
end


function kernel_4Dmatrix_mulA!(i, C, ::Oneγμ{:plus,3}, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = C[ic, 1, ix, iy, iz, it]
        x2 = C[ic, 2, ix, iy, iz, it]
        x3 = C[ic, 3, ix, iy, iz, it]
        x4 = C[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 - im * x3
        C[ic, 2, ix, iy, iz, it] = x2 + im * x4
        C[ic, 3, ix, iy, iz, it] = x3 + im * x1
        C[ic, 4, ix, iy, iz, it] = x4 - im * x2
    end

    return
end

function kernel_4Dmatrix_mulA!(i, C, ::Oneγμ{:minus,3}, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = C[ic, 1, ix, iy, iz, it]
        x2 = C[ic, 2, ix, iy, iz, it]
        x3 = C[ic, 3, ix, iy, iz, it]
        x4 = C[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 + im * x3
        C[ic, 2, ix, iy, iz, it] = x2 - im * x4
        C[ic, 3, ix, iy, iz, it] = x3 - im * x1
        C[ic, 4, ix, iy, iz, it] = x4 + im * x2
    end

    return
end

function kernel_4Dmatrix_mulA!(i, C, ::Oneγμ{:plus,4}, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = C[ic, 1, ix, iy, iz, it]
        x2 = C[ic, 2, ix, iy, iz, it]
        x3 = C[ic, 3, ix, iy, iz, it]
        x4 = C[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 - x3
        C[ic, 2, ix, iy, iz, it] = x2 - x4
        C[ic, 3, ix, iy, iz, it] = x3 - x1
        C[ic, 4, ix, iy, iz, it] = x4 - x2
    end

    return
end

function kernel_4Dmatrix_mulA!(i, C, ::Oneγμ{:minus,4}, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds for ic = 1:NC1
        x1 = C[ic, 1, ix, iy, iz, it]
        x2 = C[ic, 2, ix, iy, iz, it]
        x3 = C[ic, 3, ix, iy, iz, it]
        x4 = C[ic, 4, ix, iy, iz, it]
        C[ic, 1, ix, iy, iz, it] = x1 + x3
        C[ic, 2, ix, iy, iz, it] = x2 + x4
        C[ic, 3, ix, iy, iz, it] = x3 + x1
        C[ic, 4, ix, iy, iz, it] = x4 + x2
    end

    return
end
