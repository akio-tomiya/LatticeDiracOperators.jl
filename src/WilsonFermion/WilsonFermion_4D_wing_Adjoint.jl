import Base

"""
Struct for WilsonFermion
"""
struct WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis} <: WilsonFermion_4D{NC} #AbstractFermionfields_4D{NC}

    f::Array{ComplexF64,6}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    NumofBasis::Int64
    #BoundaryCondition::Vector{Int8}


    function WilsonFermion_4D_wing_Adjoint(
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
    ) where {T<:Integer}
        NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
        NG = 4
        NDW = 1
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64, NumofBasis, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW, NG)
        Dirac_operator = "Wilson"
        return new{NC,NDW,NumofBasis}(
            f,
            NC,
            NX,
            NY,
            NZ,
            NT,
            NG,
            NDW,
            Dirac_operator,
            NumofBasis,
        )
    end

    function WilsonFermion_4D_wing_Adjoint{NC}(
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
    ) where {T<:Integer,NC}
        NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
        NG = 4
        NDW = 1
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64, NumofBasis, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW, NG)
        Dirac_operator = "Wilson"
        return new{NC,NDW,NumofBasis}(
            f,
            NC,
            NX,
            NY,
            NZ,
            NT,
            NG,
            NDW,
            Dirac_operator,
            NumofBasis,
        )
    end


end

function initialize_Adjoint_fermion(x::WilsonFermion_4D_wing{NC,NDW}) where {NC,NDW}
    xadj = WilsonFermion_4D_wing_Adjoint(x.NC, x.NX, x.NY, x.NZ, x.NT)
end

function Base.length(
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
) where {NC,NDW,NumofBasis}
    return NC * x.NX * x.NY * x.NZ * x.NT * x.NG
end

function Base.setindex!(
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC,NDW,NumofBasis}
    @inbounds x.f[i1, i2+NDW, i3+NDW, i4+NDW, i5+NDW, i6] = v
end

function Base.getindex(
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC,NDW,NumofBasis}
    @inbounds return x.f[i1, i2.+NDW, i3.+NDW, i4.+NDW, i5.+NDW, i6]
end

function Base.getindex(
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    i1::N,
    i2::N,
    i3::N,
    i4::N,
    i5::N,
    i6::N,
) where {NC,NDW,NumofBasis,N<:Integer}
    @inbounds return x.f[i1, i2+NDW, i3+NDW, i4+NDW, i5+NDW, i6]
end




function Base.similar(x::T) where {T<:WilsonFermion_4D_wing_Adjoint}
    return WilsonFermion_4D_wing_Adjoint(x.NC, x.NX, x.NY, x.NZ, x.NT)
end



function set_wing_fermion!(
    a::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    boundarycondition,
) where {NC,NDW,NumofBasis}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX

    #!  X-direction
    for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for k = 1:NumofBasis
                        a[k, 0, iy, iz, it, ialpha] =
                            boundarycondition[1] * a[k, NX, iy, iz, it, ialpha]
                    end
                end
            end
        end
    end

    for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for k = 1:NumofBasis
                        a[k, NX+1, iy, iz, it, ialpha] =
                            boundarycondition[1] * a[k, 1, iy, iz, it, ialpha]
                    end
                end
            end
        end
    end

    #Y-direction
    for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for ix = 1:NX
                    @simd for k = 1:NumofBasis
                        a[k, ix, 0, iz, it, ialpha] =
                            boundarycondition[2] * a[k, ix, NY, iz, it, ialpha]
                    end
                end
            end
        end
    end

    for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for ix = 1:NX
                    @simd for k = 1:NumofBasis
                        a[k, ix, NY+1, iz, it, ialpha] =
                            boundarycondition[2] * a[k, ix, 1, iz, it, ialpha]
                    end
                end
            end
        end
    end


    for ialpha = 1:4
        # Z-direction
        for it = 1:NT
            for iy = 1:NY
                for ix = 1:NX
                    @simd for k = 1:NumofBasis
                        a[k, ix, iy, 0, it, ialpha] =
                            boundarycondition[3] * a[k, ix, iy, NZ, it, ialpha]
                        a[k, ix, iy, NZ+1, it, ialpha] =
                            boundarycondition[3] * a[k, ix, iy, 1, it, ialpha]

                    end
                end
            end
        end

        #T-direction
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    @simd for k = 1:NumofBasis
                        a[k, ix, iy, iz, 0, ialpha] =
                            boundarycondition[4] * a[k, ix, iy, iz, NT, ialpha]
                        a[k, ix, iy, iz, NT+1, ialpha] =
                            boundarycondition[4] * a[k, ix, iy, iz, 1, ialpha]
                    end
                end
            end
        end

    end

end

function set_wing_fermion!(
    a::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    boundarycondition,
    iseven::Bool,
) where {NC,NDW,NumofBasis}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX

    #!  X-direction
    @inbounds for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    ix = NX
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        @simd for k = 1:NumofBasis
                            a[k, 0, iy, iz, it, ialpha] =
                                boundarycondition[1] * a[k, NX, iy, iz, it, ialpha]
                        end
                    end
                end
            end
        end
    end

    @inbounds for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    ix = 1
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        @simd for k = 1:NumofBasis
                            a[k, NX+1, iy, iz, it, ialpha] =
                                boundarycondition[1] * a[k, 1, iy, iz, it, ialpha]
                        end
                    end
                end
            end
        end
    end

    #Y-direction
    @inbounds for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for ix = 1:NX
                    iy = NY
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven

                        @simd for k = 1:NumofBasis
                            a[k, ix, 0, iz, it, ialpha] =
                                boundarycondition[2] * a[k, ix, NY, iz, it, ialpha]
                        end
                    end
                end

            end
        end
    end

    @inbounds for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for ix = 1:NX
                    iy = 1
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven

                        @simd for k = 1:NumofBasis
                            a[k, ix, NY+1, iz, it, ialpha] =
                                boundarycondition[2] * a[k, ix, 1, iz, it, ialpha]
                        end
                    end
                end
            end
        end
    end


    @inbounds for ialpha = 1:4
        # Z-direction
        for it = 1:NT
            for iy = 1:NY
                for ix = 1:NX
                    iz = NX
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        @simd for k = 1:NumofBasis
                            a[k, ix, iy, 0, it, ialpha] =
                                boundarycondition[3] * a[k, ix, iy, NZ, it, ialpha]
                        end
                    end

                    iz = 1
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        @simd for k = 1:NumofBasis
                            a[k, ix, iy, NZ+1, it, ialpha] =
                                boundarycondition[3] * a[k, ix, iy, 1, it, ialpha]
                        end
                    end
                end
            end
        end

        #T-direction
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    it = NT
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        @simd for k = 1:NumofBasis
                            a[k, ix, iy, iz, 0, ialpha] =
                                boundarycondition[4] * a[k, ix, iy, iz, NT, ialpha]
                        end
                    end

                    it = 1
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        @simd for k = 1:NumofBasis
                            a[k, ix, iy, iz, NT+1, ialpha] =
                                boundarycondition[4] * a[k, ix, iy, iz, 1, ialpha]
                        end
                    end

                end
            end
        end

    end

end




function add_fermion!(
    c::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    α::Number,
    a::T1,
    β::Number,
    b::T2,
    iseven,
) where {NC,NDW,NumofBasis,T1<:Abstractfermion,T2<:Abstractfermion}#c += alpha*a + beta*b
    n1, n2, n3, n4, n5, n6 = size(c.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            it = i5 - NDW
            for i4 = 1:n4
                iz = i4 - NDW
                for i3 = 1:n3
                    iy = i3 - NDW
                    for i2 = 1:n2
                        ix = i2 - NDW
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            @simd for i1 = 1:NumofBasis
                                #println(a.f[i1,i2,i3,i4,i5,i6],"\t",b.f[i1,i2,i3,i4,i5,i6] )
                                c.f[i1, i2, i3, i4, i5, i6] +=
                                    α * a.f[i1, i2, i3, i4, i5, i6] +
                                    β * b.f[i1, i2, i3, i4, i5, i6]
                            end
                        end
                    end
                end
            end
        end
    end
    return
end

function add_fermion!(
    c::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    α::Number,
    a::T1,
    iseven::Bool,
) where {NC,NDW,NumofBasis,T1<:Abstractfermion}#c += alpha*a + beta*b
    n1, n2, n3, n4, n5, n6 = size(c.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            it = i5 - NDW
            for i4 = 1:n4
                iz = i4 - NDW
                for i3 = 1:n3
                    iy = i3 - NDW
                    for i2 = 1:n2
                        ix = i2 - NDW
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            @simd for i1 = 1:NumofBasis
                                #println(a.f[i1,i2,i3,i4,i5,i6],"\t",b.f[i1,i2,i3,i4,i5,i6] )
                                c.f[i1, i2, i3, i4, i5, i6] +=
                                    α * a.f[i1, i2, i3, i4, i5, i6]
                            end
                        end
                    end
                end
            end
        end
    end
    return
end



function clear_fermion!(
    a::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    iseven,
) where {NC,NDW,NumofBasis}
    n1, n2, n3, n4, n5, n6 = size(a.f)
    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            it = i5 - NDW
            for i4 = 1:n4
                iz = i4 - NDW
                for i3 = 1:n3
                    iy = i3 - NDW
                    for i2 = 1:n2
                        ix = i2 - NDW
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            @simd for i1 = 1:NumofBasis
                                a.f[i1, i2, i3, i4, i5, i6] = 0
                            end
                        end
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    A::TA,
) where {TA<:AbstractMatrix,NC,NDW,NumofBasis}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        e1 = x[ic, ix, iy, iz, it, 1]
                        e2 = x[ic, ix, iy, iz, it, 2]
                        e3 = x[ic, ix, iy, iz, it, 3]
                        e4 = x[ic, ix, iy, iz, it, 4]

                        x[ic, ix, iy, iz, it, 1] =
                            A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
                        x[ic, ix, iy, iz, it, 2] =
                            A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
                        x[ic, ix, iy, iz, it, 3] =
                            A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
                        x[ic, ix, iy, iz, it, 4] =
                            A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4

                    end
                end
            end
        end
    end

end

function LinearAlgebra.mul!(
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    A::TA,
    iseven,
) where {TA<:AbstractMatrix,NC,NDW,NumofBasis}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            e1 = x[ic, ix, iy, iz, it, 1]
                            e2 = x[ic, ix, iy, iz, it, 2]
                            e3 = x[ic, ix, iy, iz, it, 3]
                            e4 = x[ic, ix, iy, iz, it, 4]

                            x[ic, ix, iy, iz, it, 1] =
                                A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
                            x[ic, ix, iy, iz, it, 2] =
                                A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
                            x[ic, ix, iy, iz, it, 3] =
                                A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
                            x[ic, ix, iy, iz, it, 4] =
                                A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
                        end
                    end
                end
            end
        end
    end

end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    A::TA,
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
) where {TA<:AbstractMatrix,NC,NDW,NumofBasis}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        e1 = x[ic, ix, iy, iz, it, 1]
                        e2 = x[ic, ix, iy, iz, it, 2]
                        e3 = x[ic, ix, iy, iz, it, 3]
                        e4 = x[ic, ix, iy, iz, it, 4]

                        xout[ic, ix, iy, iz, it, 1] =
                            A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
                        xout[ic, ix, iy, iz, it, 2] =
                            A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
                        xout[ic, ix, iy, iz, it, 3] =
                            A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
                        xout[ic, ix, iy, iz, it, 4] =
                            A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4

                    end
                end
            end
        end
    end

end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    A::TA,
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    iseven,
) where {TA<:AbstractMatrix,NC,NDW,NumofBasis}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            e1 = x[ic, ix, iy, iz, it, 1]
                            e2 = x[ic, ix, iy, iz, it, 2]
                            e3 = x[ic, ix, iy, iz, it, 3]
                            e4 = x[ic, ix, iy, iz, it, 4]

                            xout[ic, ix, iy, iz, it, 1] =
                                A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
                            xout[ic, ix, iy, iz, it, 2] =
                                A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
                            xout[ic, ix, iy, iz, it, 3] =
                                A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
                            xout[ic, ix, iy, iz, it, 4] =
                                A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
                        end

                    end
                end
            end
        end
    end

end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    A::TA,
) where {TA<:AbstractMatrix,NC,NDW,NumofBasis}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        e1 = x[ic, ix, iy, iz, it, 1]
                        e2 = x[ic, ix, iy, iz, it, 2]
                        e3 = x[ic, ix, iy, iz, it, 3]
                        e4 = x[ic, ix, iy, iz, it, 4]

                        xout[ic, ix, iy, iz, it, 1] =
                            A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3 + A[4, 1] * e4
                        xout[ic, ix, iy, iz, it, 2] =
                            A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3 + A[4, 2] * e4
                        xout[ic, ix, iy, iz, it, 3] =
                            A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3 + A[4, 3] * e4
                        xout[ic, ix, iy, iz, it, 4] =
                            A[1, 4] * e1 + A[2, 4] * e2 + A[3, 4] * e3 + A[4, 4] * e4

                    end
                end
            end
        end
    end

end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    A::TA,
    iseven,
) where {TA<:AbstractMatrix,NC,NDW,NumofBasis}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            e1 = x[ic, ix, iy, iz, it, 1]
                            e2 = x[ic, ix, iy, iz, it, 2]
                            e3 = x[ic, ix, iy, iz, it, 3]
                            e4 = x[ic, ix, iy, iz, it, 4]

                            xout[ic, ix, iy, iz, it, 1] =
                                A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3 + A[4, 1] * e4
                            xout[ic, ix, iy, iz, it, 2] =
                                A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3 + A[4, 2] * e4
                            xout[ic, ix, iy, iz, it, 3] =
                                A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3 + A[4, 3] * e4
                            xout[ic, ix, iy, iz, it, 4] =
                                A[1, 4] * e1 + A[2, 4] * e2 + A[3, 4] * e3 + A[4, 4] * e4
                        end
                    end
                end
            end
        end
    end

end

function LinearAlgebra.dot(
    a::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    b::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    iseven::Bool,
) where {NC,NDW,NumofBasis}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX
    NG = a.NG

    c = 0.0im
    @inbounds for α = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            @simd for ic = 1:NumofBasis
                                c +=
                                    conj(a[ic, ix, iy, iz, it, α]) *
                                    b[ic, ix, iy, iz, it, α]
                            end
                        end
                    end
                end
            end
        end
    end
    return c
end


#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    a::Number,
    X::T,
    b::Number,
    Y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    iseven::Bool,
) where {NC,NDW,NumofBasis,T<:AbstractFermionfields_4D}
    n1, n2, n3, n4, n5, n6 = size(Y.f)
    #println("axpby")

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            it = i5 + NDW
            for i4 = 1:n4
                iz = i4 + NDW
                for i3 = 1:n3
                    iy = i3 + NDW
                    for i2 = 1:n2
                        ix = i2 + NDW
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            @simd for i1 = 1:NumofBasis
                                Y.f[i1, i2, i3, i4, i5, i6] =
                                    a * X.f[i1, i2, i3, i4, i5, i6] +
                                    b * Y.f[i1, i2, i3, i4, i5, i6]
                            end
                        end
                    end
                end
            end
        end
    end
end




"""
c--------------------------------------------------------------------------c
c     y = gamma_5 * x
c     here
c                  ( -1       )
c        GAMMA5 =  (   -1     )
c                  (     +1   )
c                  (       +1 )
c--------------------------------------------------------------------------c
    """
function mul_γ5x!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
) where {NC,NDW,NumofBasis}
    n1, n2, n3, n4, n5, n6 = size(x.f)
    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            #it = i5+NDW
            for i4 = 1:n4
                #iz = i4+NDW
                for i3 = 1:n3
                    #iy = i3+NDW
                    for i2 = 1:n2
                        #ix = i2+NDW
                        @simd for ic = 1:NumofBasis
                            y.f[ic, i2, i3, i4, i5, i6] =
                                x.f[ic, i2, i3, i4, i5, i6] * ifelse(i6 <= 2, -1, 1)
                        end
                    end
                end
            end
        end
    end

end

function apply_γ5!(
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
) where {NC,NDW,NumofBasis}
    n1, n2, n3, n4, n5, n6 = size(x.f)
    #println("axpby")

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            #it = i5+NDW
            for i4 = 1:n4
                #iz = i4+NDW
                for i3 = 1:n3
                    #iy = i3+NDW
                    for i2 = 1:n2
                        #ix = i2+NDW
                        @simd for ic = 1:NumofBasis
                            x.f[ic, i2, i3, i4, i5, i6] =
                                x.f[ic, i2, i3, i4, i5, i6] * ifelse(i6 <= 2, -1, 1)
                        end
                    end
                end
            end
        end
    end

end


function mul_1plusγ5x!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
) where {NC,NDW,NumofBasis}#(1+gamma_5)/2
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #NC = x.NC
    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        y[ic, ix, iy, iz, it, 1] = 0#-1*x[ic,ix,iy,iz,it,1]
                        y[ic, ix, iy, iz, it, 2] = 0#-1*x[ic,ix,iy,iz,it,2]
                        y[ic, ix, iy, iz, it, 3] = x[ic, ix, iy, iz, it, 3]
                        y[ic, ix, iy, iz, it, 4] = x[ic, ix, iy, iz, it, 4]
                    end
                end
            end
        end
    end
end

function mul_1plusγ5x_add!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    factor,
) where {NDW,NumofBasis,NC}#x = x +(1+gamma_5)/2
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #NC = x.NC
    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        y[ic, ix, iy, iz, it, 3] += factor * x[ic, ix, iy, iz, it, 3]
                        y[ic, ix, iy, iz, it, 4] += factor * x[ic, ix, iy, iz, it, 4]
                    end
                end
            end
        end
    end
end

function mul_1minusγ5x!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
) where {NC,NDW,NumofBasis}#(1-gamma_5)/2
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #NC = x.NC
    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        y[ic, ix, iy, iz, it, 1] = x[ic, ix, iy, iz, it, 1]
                        y[ic, ix, iy, iz, it, 2] = x[ic, ix, iy, iz, it, 2]
                        y[ic, ix, iy, iz, it, 3] = 0#x[ic,ix,iy,iz,it,3]
                        y[ic, ix, iy, iz, it, 4] = 0#x[ic,ix,iy,iz,it,4]
                    end
                end
            end
        end
    end
end

function mul_1minusγ5x_add!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    factor,
) where {NC,NDW,NumofBasis}#+(1-gamma_5)/2
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #NC = x.NC
    for ic = 1:NumofBasis
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for ix = 1:NX
                        y[ic, ix, iy, iz, it, 1] += factor * x[ic, ix, iy, iz, it, 1]
                        y[ic, ix, iy, iz, it, 2] += factor * x[ic, ix, iy, iz, it, 2]
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    A::T,
    x::T3,
) where {NC,NDW,NumofBasis,T<:Abstractfields,T3<:Abstractfermion}
    #@assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    #@assert NC != 3 "NC should not be 3"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for k1 = 1:NumofBasis
                            y[k1, ix, iy, iz, it, ialpha] = 0
                            @simd for k2 = 1:NumofBasis
                                y[k1, ix, iy, iz, it, ialpha] +=
                                    A[k1, k2, ix, iy, iz, it] *
                                    x[k2, ix, iy, iz, it, ialpha]
                            end
                        end
                    end
                end
            end
        end
    end
end


#=
function LinearAlgebra.mul!(y::WilsonFermion_4D_wing_Adjoint{3,NDW,NumofBasis},A::T,x::T3) where {NDW,NumofBasis,T<:Abstractfields,T3 <:Abstractfermion}
    mul!(y,A,x)
end

function LinearAlgebra.mul!(y::WilsonFermion_4D_wing_Adjoint{3,NDW,NumofBasis},A::T,x::T3,iseven::Bool) where {NDW,NumofBasis,T<:Abstractfields,T3 <:Abstractfermion}
    mul!(y,A,x,iseven)
end

function LinearAlgebra.mul!(y::WilsonFermion_4D_wing_Adjoint{2,NDW,NumofBasis},A::T,x::T3) where {NDW,NumofBasis,T<:Abstractfields,T3 <:Abstractfermion}
    mul!(y,A,x)
end

function LinearAlgebra.mul!(y::WilsonFermion_4D_wing_Adjoint{2,NDW,NumofBasis},A::T,x::T3,iseven::Bool) where {NDW,NumofBasis,T<:Abstractfields,T3 <:Abstractfermion}
    mul!(y,A,x,iseven)
end
=#

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::T3,
    A::T,
) where {NC,NDW,NumofBasis,T<:Abstractfields,T3<:Abstractfermion}
    #@assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for k1 = 1:NumofBasis
                            y[k1, ix, iy, iz, it, ialpha] = 0
                            @simd for k2 = 1:NumofBasis
                                y[k1, ix, iy, iz, it, ialpha] +=
                                    x[k1, ix, iy, iz, it, ialpha] *
                                    A[k1, k2, ix, iy, iz, it]
                            end
                        end
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    x::T3,
    A::T,
    iseven::Bool,
) where {NC,NDW,NumofBasis,T<:Abstractfields,T3<:Abstractfermion}
    #@assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven

                            for k1 = 1:NumofBasis
                                y[k1, ix, iy, iz, it, ialpha] = 0
                                @simd for k2 = 1:NumofBasis
                                    y[k1, ix, iy, iz, it, ialpha] +=
                                        x[k1, ix, iy, iz, it, ialpha] *
                                        A[k1, k2, ix, iy, iz, it]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_wing_Adjoint{3,NDW,NumofBasis},
    x::T3,
    A::T,
) where {NDW,NumofBasis,T<:Abstractfields,T3<:Abstractfermion}
    mul!(y, x, A)
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_wing_Adjoint{2,NDW,NumofBasis},
    x::T3,
    A::T,
) where {NDW,NumofBasis,T<:Abstractfields,T3<:Abstractfermion}
    mul!(y, x, A)
end



function LinearAlgebra.mul!(
    y::WilsonFermion_4D_wing_Adjoint{NC,NDW,NumofBasis},
    A::T,
    x::T3,
) where {NC,NDW,NumofBasis,T<:Number,T3<:Abstractfermion}
    @assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for k1 = 1:NumofBasis
                            y[k1, ix, iy, iz, it, ialpha] =
                                A * x[k1, ix, iy, iz, it, ialpha]
                        end
                    end
                end
            end
        end
    end
end
