import Base

"""
Struct for WilsonFermion
"""
struct WilsonFermion_4D_wing{NC,NDW} <: WilsonFermion_4D{NC} #AbstractFermionfields_4D{NC}

    f::Array{ComplexF64,6}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    #BoundaryCondition::Vector{Int8}


    function WilsonFermion_4D_wing(NC::T, NX::T, NY::T, NZ::T, NT::T) where {T<:Integer}
        NG = 4
        NDW = 1
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW, NG)
        Dirac_operator = "Wilson"
        return new{NC,NDW}(f, NC, NX, NY, NZ, NT, NG, NDW, Dirac_operator)
    end

    function WilsonFermion_4D_wing{NC}(NX::T, NY::T, NZ::T, NT::T) where {T<:Integer,NC}
        NG = 4
        NDW = 1
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW, NG)
        Dirac_operator = "Wilson"
        return new{NC,NDW}(f, NC, NX, NY, NZ, NT, NG, NDW, Dirac_operator)
    end


end



function Base.length(x::WilsonFermion_4D_wing{NC,NDW}) where {NC,NDW}
    return NC * x.NX * x.NY * x.NZ * x.NT * x.NG
end

function Base.setindex!(
    x::WilsonFermion_4D_wing{NC,NDW},
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC,NDW}
    @inbounds x.f[i1, i2+NDW, i3+NDW, i4+NDW, i5+NDW, i6] = v
end

function Base.getindex(
    x::WilsonFermion_4D_wing{NC,NDW},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC,NDW}
    @inbounds return x.f[i1, i2.+NDW, i3.+NDW, i4.+NDW, i5.+NDW, i6]
end

function Base.getindex(
    x::WilsonFermion_4D_wing{NC,NDW},
    i1::N,
    i2::N,
    i3::N,
    i4::N,
    i5::N,
    i6::N,
) where {NC,NDW,N<:Integer}
    @inbounds return x.f[i1, i2+NDW, i3+NDW, i4+NDW, i5+NDW, i6]
end

function setindex_global!() end


#=
function Base.getindex(F::Shifted_fermionfields_4D{NC,WilsonFermion_4D_wing{NC,NDW}},i1::N,i2::N,i3::N,i4::N,i5::N,i6::N)  where {NC,NDW,N <: Integer}

    @inbounds begin
    si2 = i2 + NDW  + F.shift[1]
    si3 = i3 + NDW  + F.shift[2]
    si4 = i4 + NDW  + F.shift[3]
    si5 = i5 + NDW  + F.shift[4]
    end
    #v = F.parent.f
    #return  v[i1,si2,si3,si4,si5,i6]

   return  @inbounds F.parent.f[i1,si2,si3,si4,si5,i6]
end
=#

#=

function Base.getindex(x::WilsonFermion_4D_wing{NC,NDW},i1,i2,i3,i4,i5,i6) where {NC,NDW}
    #=
    i2new = i2 .+ x.NDW
    i3new = i3 .+ x.NDW
    i4new = i4 .+ x.NDW
    i5new = i5 .+ x.NDW
    @inbounds return x.f[i1,i2new,i3new,i4new,i5new,i6]
    =#
    #return @inbounds Base.getindex(x.f,i1,i2 .+ NDW,i3 .+ NDW,i4 .+ NDW,i5 .+ NDW,i6)
    @inbounds return x.f[i1,i2 .+ NDW,i3 .+ NDW,i4 .+ NDW,i5 .+ NDW,i6]
end

=#

function Base.similar(x::T) where {T<:WilsonFermion_4D_wing}
    return WilsonFermion_4D_wing(x.NC, x.NX, x.NY, x.NZ, x.NT)
end

#=
function Base.getindex(x::T,i1,i2,i3,i4,i5,i6) where T <: WilsonFermion_4D_wing{NC} where NC
    i2new = i2 + x.NDW
    i3new = i3 + x.NDW
    i4new = i4 + x.NDW
    i5new = i5 + x.NDW
    f1 = x.f
    #println((i1,i2new,i3new,i4new,i5new,i6))
    @inbounds v = f1[i1,i2new,i3new,i4new,i5new,i6]
    #x.f[i1,i2new,i3new,i4new,i5new,i6]
    return v
    @inbounds return x.f[i1,i2new,i3new,i4new,i5new,i6]
    #@inbounds return x.f[i1,i2 .+ x.NDW,i3 .+ x.NDW,i4 .+ x.NDW,i5 .+ x.NDW,i6]
end
=#

function set_wing_fermion!(
    a::WilsonFermion_4D_wing{NC,NDW},
    boundarycondition,
) where {NC,NDW}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX

    #!  X-direction
    for ialpha = 1:4
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    @simd for k = 1:NC
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
                    @simd for k = 1:NC
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
                    @simd for k = 1:NC
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
                    @simd for k = 1:NC
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
                    @simd for k = 1:NC
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
                    @simd for k = 1:NC
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
    a::WilsonFermion_4D_wing{NC,NDW},
    boundarycondition,
    iseven::Bool,
) where {NC,NDW}
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
                        @simd for k = 1:NC
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
                        @simd for k = 1:NC
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

                        @simd for k = 1:NC
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

                        @simd for k = 1:NC
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
                        @simd for k = 1:NC
                            a[k, ix, iy, 0, it, ialpha] =
                                boundarycondition[3] * a[k, ix, iy, NZ, it, ialpha]
                        end
                    end

                    iz = 1
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        @simd for k = 1:NC
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
                        @simd for k = 1:NC
                            a[k, ix, iy, iz, 0, ialpha] =
                                boundarycondition[4] * a[k, ix, iy, iz, NT, ialpha]
                        end
                    end

                    it = 1
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        @simd for k = 1:NC
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
    c::WilsonFermion_4D_wing{NC,NDW},
    α::Number,
    a::T1,
    β::Number,
    b::T2,
    iseven,
) where {NC,NDW,T1<:Abstractfermion,T2<:Abstractfermion}#c += alpha*a + beta*b
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
                            @simd for i1 = 1:NC
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

"""
    add_fermion!(
    c::WilsonFermion_4D_wing{NC,NDW},
    α::Number,
    a::T1,
    iseven::Bool,
) where {NC,NDW,T1<:Abstractfermion}

TBW
"""
function add_fermion!(
    c::WilsonFermion_4D_wing{NC,NDW},
    α::Number,
    a::T1,
    iseven::Bool,
) where {NC,NDW,T1<:Abstractfermion}#c += alpha*a + beta*b
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
                            @simd for i1 = 1:NC
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

function WWx!(
    xout::T,
    U::Array{G,1},
    x::T,
    A,
) where {T<:WilsonFermion_4D_wing,G<:AbstractGaugefields} #(1 - K^2 Teo Toe) xe
    iseven = true
    isodd = false
    temp = A._temporary_fermi[7]#temps[4]
    temp2 = A._temporary_fermi[6]#temps[4]

    #println("Wx")
    #@time Wx!(xout,U,x,A) 
    clear_fermion!(xout, iseven)


    #Tx!(temp,U,x,A) 
    Toex!(temp, U, x, A, iseven) #Toe
    #Tx!(temp2,U,temp,A) 
    Toex!(temp2, U, temp, A, isodd) #Teo

    #set_wing_fermion!(temp,A.boundarycondition)
    #add_fermion!(xout,1,x,-1,temp2)
    add_fermion!(xout, 1, x, -1, temp2, iseven)
    set_wing_fermion!(xout, A.boundarycondition, iseven)



    #Wx!(xout,U,x,A) 
    return

    #Tx!(temp2,U,x,A) #Toe

    #Toex!(temp2,U,x,A,iseven) #Toe

    #Toex!(temp,U,temp2,A,isodd) #Teo
    #Tx!(temp,U,temp2,A) #Toe

    Tx!(temp, U, x, A) #Toe

    add_fermion!(xout, 1, x, -1, temp)
    #add_fermion!(xout,1,x,-1,temp,iseven)

    iseven = true
    set_wing_fermion!(xout, A.boundarycondition)

    return
end

function WWdagx!(
    xout::T,
    U::Array{G,1},
    x::T,
    A,
) where {T<:WilsonFermion_4D_wing,G<:AbstractGaugefields} #(1 - K^2 Teo Toe) xe
    iseven = true
    isodd = false
    temp = A._temporary_fermi[7]#temps[4]
    temp2 = A._temporary_fermi[6]#temps[4]

    clear_fermion!(xout)

    #Tx!(temp,U,x,A) 
    Tdagoex!(temp, U, x, A, iseven) #Toe
    #Tx!(temp2,U,temp,A) 
    Tdagoex!(temp2, U, temp, A, isodd) #Teo

    #set_wing_fermion!(temp,A.boundarycondition)
    #add_fermion!(xout,1,x,-1,temp2)
    add_fermion!(xout, 1, x, -1, temp2, iseven)
    set_wing_fermion!(xout, A.boundarycondition)


    return
end

function clear_fermion!(a::WilsonFermion_4D_wing{NC,NDW}, iseven) where {NC,NDW}
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
                            @simd for i1 = 1:NC
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
    x::WilsonFermion_4D_wing{NC,NDW},
    A::TA,
) where {TA<:AbstractMatrix,NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NC
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
    x::WilsonFermion_4D_wing{NC,NDW},
    A::TA,
    iseven::Bool,
) where {TA<:AbstractMatrix,NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NC
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

#function LinearAlgebra.mul!(xout::WilsonFermion_4D_wing{NC,NDW},A::TA,x::WilsonFermion_4D_wing{NC}) where {TA <: AbstractMatrix, NC,NDW}
function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_wing{NC,NDW},
    A::TA,
    x::Abstractfermion,
) where {TA<:AbstractMatrix,NC,NDW}

    NX = xout.NX
    NY = xout.NY
    NZ = xout.NZ
    NT = xout.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NC
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
    xout::WilsonFermion_4D_wing{NC,NDW},
    A::TA,
    x::WilsonFermion_4D_wing{NC},
    iseven::Bool,
) where {TA<:AbstractMatrix,NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NC
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
    xout::WilsonFermion_4D_wing{NC,NDW},
    x::WilsonFermion_4D_wing{NC},
    A::TA,
) where {TA<:AbstractMatrix,NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NC
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
    xout::WilsonFermion_4D_wing{NC,NDW},
    x::AbstractFermionfields_4D{NC},
    A::TA,
) where {TA<:AbstractMatrix,NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NC
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
    xout::WilsonFermion_4D_wing{NC,NDW},
    x::WilsonFermion_4D_wing{NC},
    A::TA,
    iseven::Bool,
) where {TA<:AbstractMatrix,NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic = 1:NC
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
    a::WilsonFermion_4D_wing{NC,NDW},
    b::WilsonFermion_4D_wing{NC,NDW},
    iseven::Bool,
) where {NC,NDW}
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
                            @simd for ic = 1:NC
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
    Y::WilsonFermion_4D_wing{NC,NDW},
    iseven::Bool,
) where {NC,NDW,T<:AbstractFermionfields_4D}
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
                            @simd for i1 = 1:NC
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



#=
function set_wing_fermion!(a::WilsonFermion_4D_wing{NC},boundarycondition) where NC 
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX

    #!  X-direction
    for ialpha=1:4
        for it=1:NT
            for iz = 1:NZ
                for iy=1:NY
                    @simd for k=1:NC
                        a[k,0,iy,iz,it,ialpha] = boundarycondition[1]*a[k,NX,iy,iz,it,ialpha]
                    end
                end
            end
        end
    end

    for ialpha=1:4
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for k=1:NC
                        a[k,NX+1,iy,iz,it,ialpha] =boundarycondition[1]*a[k,1,iy,iz,it,ialpha]
                    end
                end
            end
        end
    end

    #Y-direction
    for ialpha = 1:4
        for it=1:NT
            for iz=1:NZ
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,0,iz,it,ialpha] =boundarycondition[2]*a[k,ix,NY,iz,it,ialpha]
                    end
                end
            end
        end
    end

    for ialpha=1:4
        for it=1:NT
            for iz=1:NZ
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,NY+1,iz,it,ialpha] = boundarycondition[2]*a[k,ix,1,iz,it,ialpha]
                    end
                end
            end
        end
    end


    for ialpha=1:4
        # Z-direction
        for it=1:NT
            for iy=1:NY
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,iy,0,it,ialpha] = boundarycondition[3]*a[k,ix,iy,NZ,it,ialpha]
                        a[k,ix,iy,NZ+1,it,ialpha] = boundarycondition[3]*a[k,ix,iy,1,it,ialpha]

                    end
                end
            end
        end

        #T-direction
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,iy,iz,0,ialpha] = boundarycondition[4]*a[k,ix,iy,iz,NT,ialpha]
                        a[k,ix,iy,iz,NT+1,ialpha] =boundarycondition[4]*a[k,ix,iy,iz,1,ialpha]
                    end
                end
            end
        end

    end

end

=#

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
function mul_γ5x!(y::WilsonFermion_4D_wing{NC}, x::WilsonFermion_4D_wing{NC}) where {NC}
    n1, n2, n3, n4, n5, n6 = size(x.f)

    @inbounds for i6 = 1:n6
        s = ifelse(1 <= i6 <= 2, -1, 1)
        for i5 = 1:n5
            #it = i5+NDW
            for i4 = 1:n4
                #iz = i4+NDW
                for i3 = 1:n3
                    #iy = i3+NDW
                    for i2 = 1:n2
                        #ix = i2+NDW
                        @simd for i1 = 1:NC
                            y.f[i1, i2, i3, i4, i5, i6] =
                                x.f[i1, i2, i3, i4, i5, i6] * s #* ifelse(i6 <= 2, -1, 1)
                        end
                    end
                end
            end
        end
    end
    println(y.f)
    error("y")
    println("yy ", sum(abs.(y.f)), "\t", sum(y.f))


    return
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    for ig = 1:4
        for ic = 1:NC
            for it = 1:NT
                for iz = 1:NZ
                    for iy = 1:NY
                        for ix = 1:NX
                            @simd for ic = 1:NC
                                y[ic, ix, iy, iz, it, ig] = x[ic, ix, iy, iz, it, ig] * ifelse(ig <= 2, -1, 1)
                            end
                        end
                    end
                end
            end
        end
    end

    println("xx ", sum(abs.(x.f)), "\t", sum(x.f))
    println("yy ", sum(abs.(y.f)), "\t", sum(y.f))



end

function apply_γ5!(x::WilsonFermion_4D_wing{NC}) where {NC}
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
                        @simd for ic = 1:NC
                            x.f[i1, i2, i3, i4, i5, i6] =
                                x.f[i1, i2, i3, i4, i5, i6] * ifelse(i6 <= 2, -1, 1)
                        end
                    end
                end
            end
        end
    end

end


function mul_1plusγ5x!(
    y::WilsonFermion_4D_wing{NC},
    x::WilsonFermion_4D_wing{NC},
) where {NC}#(1+gamma_5)/2
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #NC = x.NC
    for ic = 1:NC
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
    y::WilsonFermion_4D_wing{NC},
    x::WilsonFermion_4D_wing{NC},
    factor,
) where {NC}#x = x +(1+gamma_5)/2
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #NC = x.NC
    for ic = 1:NC
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
    y::WilsonFermion_4D_wing{NC},
    x::WilsonFermion_4D_wing{NC},
) where {NC}#(1-gamma_5)/2
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #NC = x.NC
    for ic = 1:NC
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
    y::WilsonFermion_4D_wing{NC},
    x::WilsonFermion_4D_wing{NC},
    factor,
) where {NC}#+(1-gamma_5)/2
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #NC = x.NC
    for ic = 1:NC
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

"""
           (       -i )              (       -1 )
 GAMMA1 =  (     -i   )     GAMMA2 = (     +1   )
           (   +i     )              (   +1     )
           ( +i       )              ( -1       )
"""

function mul_1minusγ1x!(y::WilsonFermion_4D_wing{NC}, x) where {NC}#(1-gamma_5)/2
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    for ic = 1:NC
                        v1 = x[ic, ix, iy, iz, it, 1] + im * x[ic, ix, iy, iz, it, 4]
                        v2 = x[ic, ix, iy, iz, it, 2] + im * x[ic, ix, iy, iz, it, 3]
                        v3 = x[ic, ix, iy, iz, it, 3] - im * x[ic, ix, iy, iz, it, 2]
                        v4 = x[ic, ix, iy, iz, it, 4] - im * x[ic, ix, iy, iz, it, 1]
                        y[ic, ix, iy, iz, it, 1] = v1
                        y[ic, ix, iy, iz, it, 2] = v2
                        y[ic, ix, iy, iz, it, 3] = v3
                        y[ic, ix, iy, iz, it, 4] = v4
                    end
                end
            end
        end

    end
end

function mul_1plusγ1x!(y::WilsonFermion_4D_wing{NC}, x) where {NC}#(1-gamma_5)/2
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    for ic = 1:NC
                        v1 = x[ic, ix, iy, iz, it, 1] - im * x[ic, ix, iy, iz, it, 4]
                        v2 = x[ic, ix, iy, iz, it, 2] - im * x[ic, ix, iy, iz, it, 3]
                        v3 = x[ic, ix, iy, iz, it, 3] + im * x[ic, ix, iy, iz, it, 2]
                        v4 = x[ic, ix, iy, iz, it, 4] + im * x[ic, ix, iy, iz, it, 1]
                        y[ic, ix, iy, iz, it, 1] = v1
                        y[ic, ix, iy, iz, it, 2] = v2
                        y[ic, ix, iy, iz, it, 3] = v3
                        y[ic, ix, iy, iz, it, 4] = v4
                    end
                end
            end
        end
    end
end

function cloverterm!(vec::WilsonFermion_4D_wing{NC}, cloverterm, x::WilsonFermion_4D_wing{NC}) where {NC}
    NT = x.NT
    NZ = x.NZ
    NY = x.NY
    NX = x.NX
    CloverFμν = cloverterm.CloverFμν

    i = 0
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    i += 1
                    for k1 = 1:NC
                        for k2 = 1:NC

                            c1 = x[k2, ix, iy, iz, it, 1]
                            c2 = x[k2, ix, iy, iz, it, 2]
                            c3 = x[k2, ix, iy, iz, it, 3]
                            c4 = x[k2, ix, iy, iz, it, 4]

                            vec[k1, ix, iy, iz, it, 1] += CloverFμν[1][k1, k2, i] * (-c1) +
                                                          +CloverFμν[2][k1, k2, i] * (-im * c2) +
                                                          +CloverFμν[3][k1, k2, i] * (-c2) +
                                                          +CloverFμν[4][k1, k2, i] * (-c2) +
                                                          +CloverFμν[5][k1, k2, i] * (im * c2) +
                                                          +CloverFμν[6][k1, k2, i] * (-c1)



                            vec[k1, ix, iy, iz, it, 2] += CloverFμν[1][k1, k2, i] * (c2) +
                                                          +CloverFμν[2][k1, k2, i] * (im * c1) +
                                                          +CloverFμν[3][k1, k2, i] * (-c1) +
                                                          +CloverFμν[4][k1, k2, i] * (-c1) +
                                                          +CloverFμν[5][k1, k2, i] * (-im * c1) +
                                                          +CloverFμν[6][k1, k2, i] * (c2)

                            vec[k1, ix, iy, iz, it, 3] += CloverFμν[1][k1, k2, i] * (-c3) +
                                                          +CloverFμν[2][k1, k2, i] * (-im * c4) +
                                                          +CloverFμν[3][k1, k2, i] * (c4) +
                                                          +CloverFμν[4][k1, k2, i] * (-c4) +
                                                          +CloverFμν[5][k1, k2, i] * (-im * c4) +
                                                          +CloverFμν[6][k1, k2, i] * (c3)

                            vec[k1, ix, iy, iz, it, 4] += CloverFμν[1][k1, k2, i] * (c4) +
                                                          +CloverFμν[2][k1, k2, i] * (im * c3) +
                                                          +CloverFμν[3][k1, k2, i] * (c3) +
                                                          +CloverFμν[4][k1, k2, i] * (-c3) +
                                                          +CloverFμν[5][k1, k2, i] * (im * c3) +
                                                          +CloverFμν[6][k1, k2, i] * (-c4)


                        end
                    end
                end
            end

        end
    end

    #println("vec = ",vec*vec)

end

