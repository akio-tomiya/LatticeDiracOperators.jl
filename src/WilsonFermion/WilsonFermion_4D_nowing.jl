import Base

"""
Struct for WilsonFermion
"""
struct WilsonFermion_4D_nowing{NC} <: WilsonFermion_4D{NC} #AbstractFermionfields_4D{NC}
    f::Array{ComplexF64,6}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    fshifted::Array{ComplexF64,6}
    #BoundaryCondition::Vector{Int8}


    function WilsonFermion_4D_nowing(NC::T, NX::T, NY::T, NZ::T, NT::T) where {T<:Integer}
        NG = 4
        NDW = 0
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW, NG)
        fshifted = zero(f)
        Dirac_operator = "Wilson"

        return new{NC}(f, NC, NX, NY, NZ, NT, NG, NDW, Dirac_operator, fshifted)
    end

    function WilsonFermion_4D_nowing{NC}(NX::T, NY::T, NZ::T, NT::T) where {T<:Integer,NC}
        NG = 4
        NDW = 0
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW, NG)
        fshifted = zero(f)
        Dirac_operator = "Wilson"
        return new{NC}(f, NC, NX, NY, NZ, NT, NG, NDW, Dirac_operator, fshifted)
    end


end

const boundarycondition_default = [1, 1, 1, -1]




function shift_fermion(F::WilsonFermion_4D_nowing{NC}, ν::T;
    boundarycondition=boundarycondition_default) where {T<:Integer,NC}
    if ν == 1
        shift = (1, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1)
    end

    return Shifted_fermionfields_4D_nowing(F, shift;
        boundarycondition)
end


function shift_fermion(
    F::TF,
    shift::NTuple{Dim,T};
    boundarycondition=boundarycondition_default
) where {Dim,T<:Integer,TF<:WilsonFermion_4D_nowing}
    return Shifted_fermionfields_4D_nowing(F, shift; boundarycondition)
end




function shifted_fermion!(
    x::WilsonFermion_4D_nowing{NC},
    boundarycondition,
    shift,
) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    factor_t = 1
    factor_z = 1
    factor_y = 1
    factor_x = 1
    bc = boundarycondition

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    #for ic=1:NC
    for ig = 1:4
        for it = 1:NT
            it_shifted = it + shift[4]
            inside_up = it_shifted > NT
            inside_down = it_shifted < 1
            factor_t = ifelse(inside_up || inside_down, bc[4], 1)
            it_shifted += ifelse(inside_up, -NT, 0)
            it_shifted += ifelse(inside_down, +NT, 0)
            for iz = 1:NZ
                iz_shifted = iz + shift[3]
                inside_up = iz_shifted > NZ
                inside_down = iz_shifted < 1
                factor_z = ifelse(inside_up || inside_down, bc[3], 1)
                iz_shifted += ifelse(inside_up, -NZ, 0)
                iz_shifted += ifelse(inside_down, +NZ, 0)
                for iy = 1:NY
                    iy_shifted = iy + shift[2]
                    inside_up = iy_shifted > NY
                    inside_down = iy_shifted < 1
                    factor_y = ifelse(inside_up || inside_down, bc[2], 1)
                    iy_shifted += ifelse(inside_up, -NY, 0)
                    iy_shifted += ifelse(inside_down, +NY, 0)
                    for ix = 1:NX
                        ix_shifted = ix + shift[1]
                        inside_up = ix_shifted > NX
                        inside_down = ix_shifted < 1
                        factor_x = ifelse(inside_up || inside_down, bc[1], 1)
                        ix_shifted += ifelse(inside_up, -NX, 0)
                        ix_shifted += ifelse(inside_down, +NX, 0)
                        @inbounds @simd for ic = 1:NC
                            #@code_warntype x.f[ic,ix_shifted,iy_shifted,iz_shifted,it_shifted,ig]
                            x.fshifted[ic, ix, iy, iz, it, ig] =
                                factor_x *
                                factor_y *
                                factor_z *
                                factor_t *
                                x[ic, ix_shifted, iy_shifted, iz_shifted, it_shifted, ig]
                        end
                    end
                end
            end
        end
    end
    #end

end





function Base.length(x::WilsonFermion_4D_nowing{NC}) where {NC}
    return NC * x.NX * x.NY * x.NZ * x.NT * x.NG
end

function Base.setindex!(
    x::WilsonFermion_4D_nowing{NC},
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC}
    @inbounds x.f[i1, i2, i3, i4, i5, i6] = v
end

function setindex_global!(
    x::WilsonFermion_4D_nowing{NC},
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC}
    @inbounds x.f[i1, i2, i3, i4, i5, i6] = v
end


@inline function setvalue_fermion!(
    x::WilsonFermion_4D_nowing{NC},
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC}
    @inbounds x.f[i1, i2, i3, i4, i5, i6] = v
end


function Base.getindex(x::WilsonFermion_4D_nowing{NC}, i1, i2, i3, i4, i5, i6) where {NC}
    @inbounds return x.f[i1, i2, i3, i4, i5, i6]
end

function Base.getindex(
    x::WilsonFermion_4D_nowing{NC},
    i1::N,
    i2::N,
    i3::N,
    i4::N,
    i5::N,
    i6::N,
) where {NC,N<:Integer}
    @inbounds return x.f[i1, i2, i3, i4, i5, i6]
end



function Base.similar(x::T) where {T<:WilsonFermion_4D_nowing}
    return WilsonFermion_4D_nowing(x.NC, x.NX, x.NY, x.NZ, x.NT)
end

function Base.zero(x::T) where {T<:WilsonFermion_4D_nowing}
    return WilsonFermion_4D_nowing(x.NC, x.NX, x.NY, x.NZ, x.NT)
end

function set_wing_fermion!(a::WilsonFermion_4D_nowing{NC}, boundarycondition) where {NC}
    return
end

function set_wing_fermion!(
    a::WilsonFermion_4D_nowing{NC},
    boundarycondition,
    iseven::Bool,
) where {NC}
    return
end




function add_fermion!(
    c::WilsonFermion_4D_nowing{NC},
    α::Number,
    a::T1,
    β::Number,
    b::T2,
    iseven,
) where {NC,T1<:Abstractfermion,T2<:Abstractfermion}#c += alpha*a + beta*b
    n1, n2, n3, n4, n5, n6 = size(c.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            it = i5
            for i4 = 1:n4
                iz = i4
                for i3 = 1:n3
                    iy = i3
                    for i2 = 1:n2
                        ix = i2
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

function add_fermion!(
    c::WilsonFermion_4D_nowing{NC},
    α::Number,
    a::T1,
    iseven::Bool,
) where {NC,T1<:Abstractfermion}#c += alpha*a + beta*b
    n1, n2, n3, n4, n5, n6 = size(c.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            it = i5
            for i4 = 1:n4
                iz = i4
                for i3 = 1:n3
                    iy = i3
                    for i2 = 1:n2
                        ix = i2
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
    A;
    boundarycondition=boundarycondition_default
) where {T<:WilsonFermion_4D_nowing,G<:AbstractGaugefields} #(1 - K^2 Teo Toe) xe
    iseven = true
    isodd = false
    temp = A._temporary_fermi[7]#temps[4]
    temp2 = A._temporary_fermi[6]#temps[4]

    #println("Wx")
    #@time Wx!(xout,U,x,A) 
    clear_fermion!(xout, iseven)


    #Tx!(temp,U,x,A) 
    Toex!(temp, U, x, A, iseven; boundarycondition) #Toe
    #Tx!(temp2,U,temp,A) 
    Toex!(temp2, U, temp, A, isodd; boundarycondition) #Teo

    #set_nowing_fermion!(temp,A.boundarycondition)
    #add_fermion!(xout,1,x,-1,temp2)
    add_fermion!(xout, 1, x, -1, temp2, iseven)
    set_wing_fermion!(xout, A.boundarycondition, iseven)



    #Wx!(xout,U,x,A) 
    return

    #Tx!(temp2,U,x,A) #Toe
end

function WWdagx!(
    xout::T,
    U::Array{G,1},
    x::T,
    A;
    boundarycondition=boundarycondition_default
) where {T<:WilsonFermion_4D_nowing,G<:AbstractGaugefields} #(1 - K^2 Teo Toe) xe
    iseven = true
    isodd = false
    temp = A._temporary_fermi[7]#temps[4]
    temp2 = A._temporary_fermi[6]#temps[4]

    clear_fermion!(xout)

    #Tx!(temp,U,x,A) 
    Tdagoex!(temp, U, x, A, iseven; boundarycondition) #Toe
    #Tx!(temp2,U,temp,A) 
    Tdagoex!(temp2, U, temp, A, isodd; boundarycondition) #Teo

    #set_nowing_fermion!(temp,A.boundarycondition)
    #add_fermion!(xout,1,x,-1,temp2)
    add_fermion!(xout, 1, x, -1, temp2, iseven)
    set_wing_fermion!(xout, A.boundarycondition)


    return
end

function clear_fermion!(a::WilsonFermion_4D_nowing{NC}, iseven) where {NC}
    n1, n2, n3, n4, n5, n6 = size(a.f)
    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            it = i5
            for i4 = 1:n4
                iz = i4
                for i3 = 1:n3
                    iy =
                        i3 - for i2 = 1:n2
                            ix = i2
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
    x::WilsonFermion_4D_nowing{NC},
    A::TA,
) where {TA<:AbstractMatrix,NC}
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
                    @inbounds @simd for ix = 1:NX
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
    x::WilsonFermion_4D_nowing{NC},
    A::TA,
    iseven::Bool,
) where {TA<:AbstractMatrix,NC}
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
                    @inbounds @simd for ix = 1:NX
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

#function LinearAlgebra.mul!(xout::WilsonFermion_4D_nowing{NC},A::TA,x::WilsonFermion_4D_nowing{NC}) where {TA <: AbstractMatrix, NC,NDW}
function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_nowing{NC},
    A::TA,
    x::Abstractfermion,
) where {TA<:AbstractMatrix,NC}

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
                    @inbounds @simd for ix = 1:NX
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
    xout::WilsonFermion_4D_nowing{NC},
    A::TA,
    x::WilsonFermion_4D_nowing{NC},
    iseven::Bool,
) where {TA<:AbstractMatrix,NC}
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
                    @inbounds @simd for ix = 1:NX
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
    xout::WilsonFermion_4D_nowing{NC},
    x::WilsonFermion_4D_nowing{NC},
    A::TA,
) where {TA<:AbstractMatrix,NC}
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
                    @inbounds @simd for ix = 1:NX
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
    xout::WilsonFermion_4D_nowing{NC},
    x::AbstractFermionfields_4D{NC},
    A::TA,
) where {TA<:AbstractMatrix,NC}
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
                    @inbounds @simd for ix = 1:NX
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
    xout::WilsonFermion_4D_nowing{NC},
    x::WilsonFermion_4D_nowing{NC},
    A::TA,
    iseven::Bool,
) where {TA<:AbstractMatrix,NC}
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
                    @inbounds @simd for ix = 1:NX
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

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_nowing{NC},
    x::WilsonFermion_4D_nowing{NC},
    A::σμν{μ,ν},
) where {μ,ν,NC}
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
                    @inbounds for ix = 1:NX
                        @simd for iα = 1:4
                            iβ = A.indices[iα]
                            xout[ic, ix, iy, iz, it, iα] = A.σ[iα] * x[ic, ix, iy, iz, it, iβ]
                        end
                    end
                end
            end
        end
    end

end

function LinearAlgebra.dot(
    a::WilsonFermion_4D_nowing{NC},
    b::WilsonFermion_4D_nowing{NC},
    iseven::Bool,
) where {NC}
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
    Y::WilsonFermion_4D_nowing{NC},
    iseven::Bool,
) where {NC,T<:AbstractFermionfields_4D}
    n1, n2, n3, n4, n5, n6 = size(Y.f)
    #println("axpby")

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            it = i5
            for i4 = 1:n4
                iz = i4
                for i3 = 1:n3
                    iy = i3
                    for i2 = 1:n2
                        ix = i2
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            @inbounds @simd for i1 = 1:NC
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
function mul_γ5x!(y::WilsonFermion_4D_nowing{NC}, x::WilsonFermion_4D_nowing{NC}) where {NC}
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
                        @inbounds @simd for i1 = 1:NC
                            y.f[i1, i2, i3, i4, i5, i6] =
                                x.f[i1, i2, i3, i4, i5, i6] * ifelse(i6 <= 2, -1, 1)
                        end
                    end
                end
            end
        end
    end



end

function apply_γ5!(x::WilsonFermion_4D_nowing{NC}) where {NC}
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
                        @inbounds @simd for ic = 1:NC
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
    y::WilsonFermion_4D_nowing{NC},
    x::WilsonFermion_4D_nowing{NC},
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
                    @inbounds @simd for ix = 1:NX
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
    y::WilsonFermion_4D_nowing{NC},
    x::WilsonFermion_4D_nowing{NC},
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
                    @inbounds @simd for ix = 1:NX
                        y[ic, ix, iy, iz, it, 3] += factor * x[ic, ix, iy, iz, it, 3]
                        y[ic, ix, iy, iz, it, 4] += factor * x[ic, ix, iy, iz, it, 4]
                    end
                end
            end
        end
    end
end

function mul_1minusγ5x!(
    y::WilsonFermion_4D_nowing{NC},
    x::WilsonFermion_4D_nowing{NC},
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
                    @inbounds @simd for ix = 1:NX
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
    y::WilsonFermion_4D_nowing{NC},
    x::WilsonFermion_4D_nowing{NC},
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

function mul_1minusγ1x!(y::WilsonFermion_4D_nowing{NC}, x) where {NC}#(1-gamma_1)
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    #@inbounds 
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


function mul_1plusγ1x!(y::WilsonFermion_4D_nowing{NC}, x) where {NC}#(1+gamma_1)
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    #@inbounds for ic = 1:NC
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

"""
           (       -i )              (       -1 )
 GAMMA1 =  (     -i   )     GAMMA2 = (     +1   )
           (   +i     )              (   +1     )
           ( +i       )              ( -1       )
"""
function mul_1minusγ2x!(y::WilsonFermion_4D_nowing{NC}, x) where {NC}#(1-gamma_2)
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    #@inbounds for ic = 1:NC
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    for ic = 1:NC
                        v1 = x[ic, ix, iy, iz, it, 1] + x[ic, ix, iy, iz, it, 4]
                        v2 = x[ic, ix, iy, iz, it, 2] - x[ic, ix, iy, iz, it, 3]
                        v3 = x[ic, ix, iy, iz, it, 3] - x[ic, ix, iy, iz, it, 2]
                        v4 = x[ic, ix, iy, iz, it, 4] + x[ic, ix, iy, iz, it, 1]
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

function mul_1plusγ2x!(y::WilsonFermion_4D_nowing{NC}, x) where {NC}#(1-gamma_5)/2
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    #@inbounds for ic = 1:NC
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    for ic = 1:NC
                        v1 = x[ic, ix, iy, iz, it, 1] - x[ic, ix, iy, iz, it, 4]
                        v2 = x[ic, ix, iy, iz, it, 2] + x[ic, ix, iy, iz, it, 3]
                        v3 = x[ic, ix, iy, iz, it, 3] + x[ic, ix, iy, iz, it, 2]
                        v4 = x[ic, ix, iy, iz, it, 4] - x[ic, ix, iy, iz, it, 1]
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


"""
               (     -i   )              (     -1   )
     GAMMA3 =  (       +i )     GAMMA4 = (       -1 )
               ( +i       )              ( -1       )
               (   -i     )              (   -1     )

"""
function mul_1minusγ3x!(y::WilsonFermion_4D_nowing{NC}, x) where {NC}#(1-gamma_3)
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    #@inbounds for ic = 1:NC
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    for ic = 1:NC
                        v1 = x[ic, ix, iy, iz, it, 1] + im * x[ic, ix, iy, iz, it, 3]
                        v2 = x[ic, ix, iy, iz, it, 2] - im * x[ic, ix, iy, iz, it, 4]
                        v3 = x[ic, ix, iy, iz, it, 3] - im * x[ic, ix, iy, iz, it, 1]
                        v4 = x[ic, ix, iy, iz, it, 4] + im * x[ic, ix, iy, iz, it, 2]
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

function mul_1plusγ3x!(y::WilsonFermion_4D_nowing{NC}, x) where {NC}#(1+gamma_3)
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    #@inbounds for ic = 1:NC
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    for ic = 1:NC
                        v1 = x[ic, ix, iy, iz, it, 1] - im * x[ic, ix, iy, iz, it, 3]
                        v2 = x[ic, ix, iy, iz, it, 2] + im * x[ic, ix, iy, iz, it, 4]
                        v3 = x[ic, ix, iy, iz, it, 3] + im * x[ic, ix, iy, iz, it, 1]
                        v4 = x[ic, ix, iy, iz, it, 4] - im * x[ic, ix, iy, iz, it, 2]
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



"""
               (     -i   )              (     -1   )
     GAMMA3 =  (       +i )     GAMMA4 = (       -1 )
               ( +i       )              ( -1       )
               (   -i     )              (   -1     )

"""
function mul_1minusγ4x!(y::WilsonFermion_4D_nowing{NC}, x) where {NC}#(1-gamma_4)
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    #@inbounds for ic = 1:NC
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    for ic = 1:NC
                        v1 = x[ic, ix, iy, iz, it, 1] + x[ic, ix, iy, iz, it, 3]
                        v2 = x[ic, ix, iy, iz, it, 2] + x[ic, ix, iy, iz, it, 4]
                        v3 = x[ic, ix, iy, iz, it, 3] + x[ic, ix, iy, iz, it, 1]
                        v4 = x[ic, ix, iy, iz, it, 4] + x[ic, ix, iy, iz, it, 2]
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

function mul_1plusγ4x!(y::WilsonFermion_4D_nowing{NC}, x) where {NC}#(1+gamma_4)
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #NC = x.NC
    #@inbounds for ic = 1:NC

    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    for ic = 1:NC
                        v1 = x[ic, ix, iy, iz, it, 1] - x[ic, ix, iy, iz, it, 3]
                        v2 = x[ic, ix, iy, iz, it, 2] - x[ic, ix, iy, iz, it, 4]
                        v3 = x[ic, ix, iy, iz, it, 3] - x[ic, ix, iy, iz, it, 1]
                        v4 = x[ic, ix, iy, iz, it, 4] - x[ic, ix, iy, iz, it, 2]
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


"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::WilsonFermion_4D_nowing{NC}) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NG = x.NG
    #n6 = size(x.f)[6]
    σ = sqrt(1 / 2)


    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for ialpha = 1:NG
                        for ic = 1:NC
                            v = σ * randn() + im * σ * randn()

                            #setvalue!(x,v,ic,ialpha,ix,iy,iz,it)
                            x[ic, ix, iy, iz, it, ialpha] = v# σ*randn()+im*σ*randn()
                        end
                    end
                end
            end
        end
    end

    set_wing_fermion!(x)
    return
end

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(
    x::WilsonFermion_4D_nowing{NC},
    randomfunc,
    σ,
) where {NC}

    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NG = x.NG
    #n6 = size(x.f)[6]
    #σ = sqrt(1/2)



    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for mu = 1:NG
                        for ic = 1:NC

                            v1 = sqrt(-log(randomfunc() + 1e-10))
                            v2 = 2pi * randomfunc()

                            xr = v1 * cos(v2)
                            xi = v1 * sin(v2)

                            v = σ * xr + σ * im * xi

                            #println(v)
                            #setvalue!(x,v,ic,mu,ix,iy,iz,it)

                            x[ic, ix, iy, iz, it, mu] = v# σ*xr + σ*im*xi
                        end
                    end
                end
            end
        end
    end
    set_wing_fermion!(x)

    return
end

function apply_σ!(a::WilsonFermion_4D_nowing{NC}, σ::σμν{μ,ν}, b::WilsonFermion_4D_nowing{NC}; factor=1) where {NC,μ,ν}
    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT
    @inbounds for iα = 1:4
        value = σ.σ[iα]
        iβ = σ.indices[iα]
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        @simd for ic = 1:NC
                            a[ic, ix, iy, iz, it, iα] += factor * value * b[ic, ix, iy, iz, it, iβ]
                        end
                    end
                end
            end
        end
    end
end


function cloverterm!(vec::WilsonFermion_4D_nowing{NC}, cloverterm, x::WilsonFermion_4D_nowing{NC}) where {NC}
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

                            vec[k1, ix, iy, iz, it, 1] += CloverFμν[1][k1, k2, ix, iy, iz, it] * (-c1) +
                                                          +CloverFμν[2][k1, k2, ix, iy, iz, it] * (-im * c2) +
                                                          +CloverFμν[3][k1, k2, ix, iy, iz, it] * (-c2) +
                                                          +CloverFμν[4][k1, k2, ix, iy, iz, it] * (-c2) +
                                                          +CloverFμν[5][k1, k2, ix, iy, iz, it] * (im * c2) +
                                                          +CloverFμν[6][k1, k2, ix, iy, iz, it] * (-c1)
                            #   println("$ix $iy $iz $it $k1 $k2 $(vec[k1,ix,iy,iz,it,1] )")



                            vec[k1, ix, iy, iz, it, 2] += CloverFμν[1][k1, k2, ix, iy, iz, it] * (c2) +
                                                          +CloverFμν[2][k1, k2, ix, iy, iz, it] * (im * c1) +
                                                          +CloverFμν[3][k1, k2, ix, iy, iz, it] * (-c1) +
                                                          +CloverFμν[4][k1, k2, ix, iy, iz, it] * (-c1) +
                                                          +CloverFμν[5][k1, k2, ix, iy, iz, it] * (-im * c1) +
                                                          +CloverFμν[6][k1, k2, ix, iy, iz, it] * (c2)

                            vec[k1, ix, iy, iz, it, 3] += CloverFμν[1][k1, k2, ix, iy, iz, it] * (-c3) +
                                                          +CloverFμν[2][k1, k2, ix, iy, iz, it] * (-im * c4) +
                                                          +CloverFμν[3][k1, k2, ix, iy, iz, it] * (c4) +
                                                          +CloverFμν[4][k1, k2, ix, iy, iz, it] * (-c4) +
                                                          +CloverFμν[5][k1, k2, ix, iy, iz, it] * (-im * c4) +
                                                          +CloverFμν[6][k1, k2, ix, iy, iz, it] * (c3)

                            vec[k1, ix, iy, iz, it, 4] += CloverFμν[1][k1, k2, ix, iy, iz, it] * (c4) +
                                                          +CloverFμν[2][k1, k2, ix, iy, iz, it] * (im * c3) +
                                                          +CloverFμν[3][k1, k2, ix, iy, iz, it] * (c3) +
                                                          +CloverFμν[4][k1, k2, ix, iy, iz, it] * (-c3) +
                                                          +CloverFμν[5][k1, k2, ix, iy, iz, it] * (im * c3) +
                                                          +CloverFμν[6][k1, k2, ix, iy, iz, it] * (-c4)


                        end
                    end
                end
            end

        end
    end

    #println("vec = ",vec*vec)

end

function Ux_ν!(
    y::WilsonFermion_4D_nowing{3},
    A::T,
    x::T3,
    ν::Integer;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    if ν == 1
        Uxplus_1!(y, A, x; boundarycondition)
    elseif ν == -1
        Uxminus_1!(y, A, x; boundarycondition)
    elseif ν == 2
        Uxplus_2!(y, A, x; boundarycondition)
    elseif ν == -2
        Uxminus_2!(y, A, x; boundarycondition)
    elseif ν == 3
        Uxplus_3!(y, A, x; boundarycondition)
    elseif ν == -3
        Uxminus_3!(y, A, x; boundarycondition)
    elseif ν == 4
        Uxplus_4!(y, A, x; boundarycondition)
    elseif ν == -4
        Uxminus_4!(y, A, x; boundarycondition)
    else
        xplus = shift_fermion(x, ν; boundarycondition)
        mul!(y, A, xplus)
    end
end


function Ux_afterν!(
    y::WilsonFermion_4D_nowing{3},
    A::T,
    x::T3,
    ν::Integer;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}

    if ν == 1
        Uxplus_after1!(y, A, x; boundarycondition)
    elseif ν == -1
        Uxminus_after1!(y, A, x; boundarycondition)
    elseif ν == 2
        Uxplus_after2!(y, A, x; boundarycondition)
    elseif ν == -2
        Uxminus_after2!(y, A, x; boundarycondition)
    elseif ν == 3
        Uxplus_after3!(y, A, x; boundarycondition)
    elseif ν == -3
        Uxminus_after3!(y, A, x; boundarycondition)
    elseif ν == 4
        Uxplus_after4!(y, A, x; boundarycondition)
    elseif ν == -4
        Uxminus_after4!(y, A, x; boundarycondition)
    else

        mul!(y, A, x)
        x_shifted = shift_fermion(y, ν; boundarycondition)
        substitute_fermion!(y, x_shifted)
    end
end

