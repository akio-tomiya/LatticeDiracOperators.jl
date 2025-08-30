using JLD2

abstract type AbstractFermionfields_4D{NC} <: AbstractFermionfields{NC,4} end

mutable struct Data_sent_fermion{NC} #data format for MPI
    count::Int64
    data::Array{ComplexF64,3}
    positions::Vector{Int64}

    function Data_sent_fermion(N, NC; NG=4)
        data = zeros(ComplexF64, NC, NG, N)
        count = 0
        positions = zeros(Int64, N)

        return new{NC}(count, data, positions)
    end
end

function Base.setindex!(x::T, v, i1, i2, i3, i4, i5, i6) where {T<:AbstractFermionfields_4D}
    @inbounds x.f[i1, i2+x.NDW, i3+x.NDW, i4+x.NDW, i5+x.NDW, i6] = v
end


function setindex_global!(
    x::T,
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {T<:AbstractFermionfields_4D}
    @inbounds x.f[i1, i2+x.NDW, i3+x.NDW, i4+x.NDW, i5+x.NDW, i6] = v
end


function Base.getindex(x::T, i1, i2, i3, i4, i5, i6) where {T<:AbstractFermionfields_4D}
    #=
    i2new = i2 .+ x.NDW
    i3new = i3 .+ x.NDW
    i4new = i4 .+ x.NDW
    i5new = i5 .+ x.NDW
    @inbounds return x.f[i1,i2new,i3new,i4new,i5new,i6]
    =#
    @inbounds return x.f[i1, i2.+x.NDW, i3.+x.NDW, i4.+x.NDW, i5.+x.NDW, i6]
end

@inline function get_latticeindex_fermion(i, NC, NX, NY, NZ, NT)
    #i =(((((ig-1)*NT+it-1)*NZ+iz-1)*NY+iy-1)*NX+ix-1)*NC+ic
    ic = (i - 1) % NC + 1
    ii = div(i - ic, NC)
    #ii = (((ig-1)*NT+it-1)*NZ+iz-1)*NY+iy-1)*NX+ix-1
    ix = ii % NX + 1
    ii = div(ii - (ix - 1), NX)
    #ii = ((ig-1)*NT+it-1)*NZ+iz-1)*NY+iy-1
    iy = ii % NY + 1
    ii = div(ii - (iy - 1), NY)
    #ii = ((ig-1)*NT+it-1)*NZ+iz-1
    iz = ii % NZ + 1
    ii = div(ii - (iz - 1), NZ)
    #ii = (ig-1)*NT+it-1
    it = ii % NT + 1
    ig = div(ii - (it - 1), NT) + 1
    return ic, ix, iy, iz, it, ig
end

Base.length(x::T) where {T<:AbstractFermionfields_4D} =
    x.NC * x.NX * x.NY * x.NZ * x.NT * x.NG

function Base.size(x::AbstractFermionfields_4D{NC}) where {NC}
    return (x.NC, x.NX, x.NY, x.NZ, x.NT, x.NG)
end


function Base.iterate(x::T, state=1) where {T<:AbstractFermionfields_4D}
    if state > length(x)
        return nothing
    end

    return (x[state], state + 1)
end


function Base.setindex!(x::T, v, i) where {T<:AbstractFermionfields_4D}
    ic, ix, iy, iz, it, ig = get_latticeindex_fermion(i, x.NC, x.NX, x.NY, x.NZ, x.NT)
    @inbounds x[ic, ix, iy, iz, it, ig] = v
end

function Base.getindex(x::T, i) where {T<:AbstractFermionfields_4D}
    ic, ix, iy, iz, it, ig = get_latticeindex_fermion(i, x.NC, x.NX, x.NY, x.NZ, x.NT)
    @inbounds return x[ic, ix, iy, iz, it, ig]
end



function Base.getindex(
    F::Adjoint_fermionfields{T},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {T<:Abstractfermion}  #F'
    @inbounds return conj(F.parent[i1, i2, i3, i4, i5, i6])
end

function Base.setindex!(
    F::Adjoint_fermionfields{T},
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    μ,
) where {T<:Abstractfermion}
    error("type $(typeof(F)) has no setindex method. This type is read only.")
end


function clear_fermion!(a::Vector{<:AbstractFermionfields_4D{NC}}) where {NC}
    for μ = 1:4
        clear_fermion!(a[μ])
    end
end

function clear_fermion!(a::AbstractFermionfields_4D{NC}) where {NC}
    n1, n2, n3, n4, n5, n6 = size(a.f)
    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            for i4 = 1:n4
                for i3 = 1:n3
                    for i2 = 1:n2
                        @simd for i1 = 1:NC
                            a.f[i1, i2, i3, i4, i5, i6] = 0
                        end
                    end
                end
            end
        end
    end
end

function save_fermionfield(a::AbstractFermionfields_4D{NC}, filename) where {NC}
    jldsave(filename; ϕ=a)
    return
end

function load_fermionfield!(a::AbstractFermionfields_4D{NC}, filename) where {NC}
    jldopen(filename, "r") do file
        substitute_fermion!(a, file["ϕ"])
    end
end


function substitute_fermion!(
    a::AbstractFermionfields_4D{NC},
    b::AbstractFermionfields_4D{NC},
) where {NC}
    n1, n2, n3, n4, n5, n6 = size(a.f)
    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            for i4 = 1:n4
                for i3 = 1:n3
                    for i2 = 1:n2
                        @simd for i1 = 1:NC
                            a.f[i1, i2, i3, i4, i5, i6] = b.f[i1, i2, i3, i4, i5, i6]
                        end
                    end
                end
            end
        end
    end
end

function substitute_fermion!(a::AbstractFermionfields_4D{NC}, b::Abstractfermion) where {NC}
    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT
    NG = a.NG
    @inbounds for i6 = 1:NG
        for i5 = 1:NT
            for i4 = 1:NZ
                for i3 = 1:NY
                    for i2 = 1:NX
                        @simd for i1 = 1:NC
                            a[i1, i2, i3, i4, i5, i6] = b[i1, i2, i3, i4, i5, i6]
                        end
                    end
                end
            end
        end
    end
    set_wing_fermion!(a)
end

struct Shifted_fermionfields_4D_accelerator{NC,T} <: Shifted_fermionfields{NC,4}
    parent::T
    #parent::T
    shift::NTuple{4,Int8}
    NC::Int64
    bc::NTuple{4,Int8}

    #function Shifted_Gaugefields(U::T,shift,Dim) where {T <: AbstractGaugefields}
    function Shifted_fermionfields_4D_accelerator(
        F,
        shift;
        boundarycondition=boundarycondition_default_accelerator,
    )
        NC = F.NC
        bc = Tuple(boundarycondition)

        shifted_fermion!(F, boundarycondition, shift)
        return new{NC,typeof(F)}(F, shift, NC, bc)
    end
end

struct Shifted_fermionfields_4D_nowing{NC,T} <: Shifted_fermionfields{NC,4}
    parent::T
    #parent::T
    shift::NTuple{4,Int8}
    NC::Int64

    #function Shifted_Gaugefields(U::T,shift,Dim) where {T <: AbstractGaugefields}
    function Shifted_fermionfields_4D_nowing(
        F,
        shift;
        boundarycondition=boundarycondition_default,
    )
        NC = F.NC
        shifted_fermion!(F, boundarycondition, shift)
        return new{NC,typeof(F)}(F, shift, NC)
    end
end


function Base.getindex(F::Shifted_fermionfields_4D_nowing, i1, i2, i3, i4, i5, i6)
    @inbounds return F.parent.fshifted[i1, i2, i3, i4, i5, i6]
end

function Base.getindex(
    F::Shifted_fermionfields_4D_nowing,
    i1::N,
    i2::N,
    i3::N,
    i4::N,
    i5::N,
    i6::N,
) where {N<:Integer}
    @inbounds return F.parent.fshifted[i1, i2, i3, i4, i5, i6]
end



struct Shifted_fermionfields_4D{NC,T} <: Shifted_fermionfields{NC,4}
    parent::T
    #parent::T
    shift::NTuple{4,Int8}
    NC::Int64

    #function Shifted_Gaugefields(U::T,shift,Dim) where {T <: AbstractGaugefields}
    function Shifted_fermionfields_4D(F::AbstractFermionfields_4D{NC}, shift) where {NC}
        return new{NC,typeof(F)}(F, shift, NC)
    end
end



function Base.size(x::Shifted_fermionfields_4D)
    return size(x.parent)
end
using InteractiveUtils

#=
function shift_fermion(U::AbstractFermionfields_4D{NC},ν::T) where {T <: Integer,NC}
    return shift_fermion(U,Val(ν))
end

function shift_fermion(U::AbstractFermionfields_4D{NC},::Val{1}) where {T <: Integer,NC}
    shift = (1,0,0,0)
    return Shifted_fermionfields_4D(U,shift)
end

function shift_fermion(U::AbstractFermionfields_4D{NC},::Val{2}) where {T <: Integer,NC}
    shift = (0,1,0,0)
    return Shifted_fermionfields_4D(U,shift)
end

function shift_fermion(U::AbstractFermionfields_4D{NC},::Val{3}) where {T <: Integer,NC}
    shift = (0,0,1,0)
    return Shifted_fermionfields_4D(U,shift)
end

function shift_fermion(U::AbstractFermionfields_4D{NC},::Val{4}) where {T <: Integer,NC}
    shift = (0,0,0,1)
    return Shifted_fermionfields_4D(U,shift)
end

function shift_fermion(U::AbstractFermionfields_4D{NC},::Val{-1}) where {T <: Integer,NC}
    shift = (-1,0,0,0)
    return Shifted_fermionfields_4D(U,shift)
end

function shift_fermion(U::AbstractFermionfields_4D{NC},::Val{-2}) where {T <: Integer,NC}
    shift = (0,-1,0,0)
    return Shifted_fermionfields_4D(U,shift)
end

function shift_fermion(U::AbstractFermionfields_4D{NC},::Val{-3}) where {T <: Integer,NC}
    shift = (0,0,-1,0)
    return Shifted_fermionfields_4D(U,shift)
end

function shift_fermion(U::AbstractFermionfields_4D{NC},::Val{-4}) where {T <: Integer,NC}
    shift = (0,0,0,-1)
    return Shifted_fermionfields_4D(U,shift)
end
=#


#lattice shift
function shift_fermion(F::AbstractFermionfields_4D{NC}, ν::T) where {T<:Integer,NC}
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

    return Shifted_fermionfields_4D(F, shift)
end


function shift_fermion(
    F::TF,
    shift::NTuple{Dim,T},
) where {Dim,T<:Integer,TF<:AbstractFermionfields_4D}
    return Shifted_fermionfields_4D(F, shift)
end

function Base.setindex!(F::T, v, i1, i2, i3, i4, i5, i6) where {T<:Shifted_fermionfields_4D}
    error("type $(typeof(F)) has no setindex method. This type is read only.")
end

function Base.getindex(F::T, i1, i2, i3, i4, i5, i6) where {T<:Shifted_fermionfields_4D}
    @inbounds return F.parent[
        i1,
        i2.+F.shift[1],
        i3.+F.shift[2],
        i4.+F.shift[3],
        i5.+F.shift[4],
        i6,
    ]
end

function Base.getindex(
    F::T,
    i1::N,
    i2::N,
    i3::N,
    i4::N,
    i5::N,
    i6::N,
) where {T<:Shifted_fermionfields_4D,N<:Integer}
    @inbounds return F.parent[
        i1,
        i2+F.shift[1],
        i3+F.shift[2],
        i4+F.shift[3],
        i5+F.shift[4],
        i6,
    ]
end

#=

function Base.getindex(F::Shifted_fermionfields_4D{NC,T,Val{(1,0,0,0)}},i1,i2,i3,i4,i5,i6)  where {NC,T,shift}
    @inbounds return F.parent[i1,i2.+ 1,i3,i4,i5,i6]
end

function Base.getindex(F::Shifted_fermionfields_4D{NC,T,Val{(-1,0,0,0)}},i1,i2,i3,i4,i5,i6)  where {NC,T,shift}
    @inbounds return F.parent[i1,i2 .- 1,i3,i4,i5,i6]
end

function Base.getindex(F::Shifted_fermionfields_4D{NC,T,Val{(0,1,0,0)}},i1,i2,i3,i4,i5,i6)  where {NC,T,shift}
    @inbounds return F.parent[i1,i2,i3 .+ 1,i4,i5,i6]
end

function Base.getindex(F::Shifted_fermionfields_4D{NC,T,Val{(0,-1,0,0)}},i1,i2,i3,i4,i5,i6)  where {NC,T,shift}
    @inbounds return F.parent[i1,i2,i3 .- 1,i4,i5,i6]
end

function Base.getindex(F::Shifted_fermionfields_4D{NC,T,Val{(0,0,1,0)}},i1,i2,i3,i4,i5,i6)  where {NC,T,shift}
    @inbounds return F.parent[i1,i2,i3 ,i4 .+ 1,i5,i6]
end

function Base.getindex(F::Shifted_fermionfields_4D{NC,T,Val{(0,0,-1,0)}},i1,i2,i3,i4,i5,i6)  where {NC,T,shift}
    @inbounds return F.parent[i1,i2,i3,i4 .- 1 ,i5,i6]
end

function Base.getindex(F::Shifted_fermionfields_4D{NC,T,Val{(0,0,0,1)}},i1,i2,i3,i4,i5,i6)  where {NC,T,shift}
    @inbounds return F.parent[i1,i2,i3 ,i4 ,i5 .+ 1,i6]
end

function Base.getindex(F::Shifted_fermionfields_4D{NC,T,Val{(0,0,0,-1)}},i1,i2,i3,i4,i5,i6)  where {NC,T,shift}
    @inbounds return F.parent[i1,i2,i3,i4 ,i5 .- 1,i6]
end
=#

function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{NC},
    A::T,
    x::T3,
) where {NC,T<:Abstractfields,T3<:Abstractfermion}
    #@assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    @assert NC != 3 "NC should not be 3"
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
                        for k1 = 1:NC
                            y[k1, ix, iy, iz, it, ialpha] = 0
                            @simd for k2 = 1:NC
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

@inline function updatefunc!(y, A, x, ix, iy, iz, it, ialpha)
    #@code_llvm  @inbounds x[1,ix,iy,iz,it,ialpha]
    #@code_typed x[1,ix,iy,iz,it,ialpha]
    #error("dd")
    #@code_lowered x[1,ix,iy,iz,it,ialpha]
    x1 = x[1, ix, iy, iz, it, ialpha]
    x2 = x[2, ix, iy, iz, it, ialpha]
    x3 = x[3, ix, iy, iz, it, ialpha]
    y[1, ix, iy, iz, it, ialpha] =
        A[1, 1, ix, iy, iz, it] * x1 +
        A[1, 2, ix, iy, iz, it] * x2 +
        A[1, 3, ix, iy, iz, it] * x3
    y[2, ix, iy, iz, it, ialpha] =
        A[2, 1, ix, iy, iz, it] * x1 +
        A[2, 2, ix, iy, iz, it] * x2 +
        A[2, 3, ix, iy, iz, it] * x3
    y[3, ix, iy, iz, it, ialpha] =
        A[3, 1, ix, iy, iz, it] * x1 +
        A[3, 2, ix, iy, iz, it] * x2 +
        A[3, 3, ix, iy, iz, it] * x3
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    #println(x.shift)

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=
                        x1 = x[1, ix, iy, iz, it, ialpha]
                        x2 = x[2, ix, iy, iz, it, ialpha]
                        x3 = x[3, ix, iy, iz, it, ialpha]




                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#

                        #if (ix, iy, iz, it) == (1, 1, 1, 1)
                        #    println((x1, x2, x3))
                        #    println((y[1, ix, iy, iz, it, ialpha], y[2, ix, iy, iz, it, ialpha], y[3, ix, iy, iz, it, ialpha]))

                        #end
                    end
                end
            end
        end
    end
end


function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3,
    iseven::Bool,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                        if evenodd == iseven
                            #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                            #error("oo")
                            # #=
                            x1 = x[1, ix, iy, iz, it, ialpha]
                            x2 = x[2, ix, iy, iz, it, ialpha]
                            x3 = x[3, ix, iy, iz, it, ialpha]
                            y[1, ix, iy, iz, it, ialpha] =
                                A[1, 1, ix, iy, iz, it] * x1 +
                                A[1, 2, ix, iy, iz, it] * x2 +
                                A[1, 3, ix, iy, iz, it] * x3
                            y[2, ix, iy, iz, it, ialpha] =
                                A[2, 1, ix, iy, iz, it] * x1 +
                                A[2, 2, ix, iy, iz, it] * x2 +
                                A[2, 3, ix, iy, iz, it] * x3
                            y[3, ix, iy, iz, it, ialpha] =
                                A[3, 1, ix, iy, iz, it] * x1 +
                                A[3, 2, ix, iy, iz, it] * x2 +
                                A[3, 3, ix, iy, iz, it] * x3
                        end
                        # =#
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{2},
    A::T,
    x::T3,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 2 == x.NC "dimension mismatch! NC in y is 2 but NC in x is $(x.NC)"
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
                        x1 = x[1, ix, iy, iz, it, ialpha]
                        x2 = x[2, ix, iy, iz, it, ialpha]
                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 + A[1, 2, ix, iy, iz, it] * x2
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 + A[2, 2, ix, iy, iz, it] * x2

                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{2},
    A::T,
    x::T3,
    iseven::Bool,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 2 == x.NC "dimension mismatch! NC in y is 2 but NC in x is $(x.NC)"
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
                            x1 = x[1, ix, iy, iz, it, ialpha]
                            x2 = x[2, ix, iy, iz, it, ialpha]
                            y[1, ix, iy, iz, it, ialpha] =
                                A[1, 1, ix, iy, iz, it] * x1 + A[1, 2, ix, iy, iz, it] * x2
                            y[2, ix, iy, iz, it, ialpha] =
                                A[2, 1, ix, iy, iz, it] * x1 + A[2, 2, ix, iy, iz, it] * x2
                        end

                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{NC},
    x::T3,
    A::T,
) where {NC,T<:Abstractfields,T3<:Abstractfermion}
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
                        for k1 = 1:NC
                            y[k1, ix, iy, iz, it, ialpha] = 0
                            @simd for k2 = 1:NC
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
    y::AbstractFermionfields_4D{NC},
    x::T3,
    A::T,
    iseven::Bool,
) where {NC,T<:Abstractfields,T3<:Abstractfermion}
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

                            for k1 = 1:NC
                                y[k1, ix, iy, iz, it, ialpha] = 0
                                @simd for k2 = 1:NC
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
    y::AbstractFermionfields_4D{3},
    x::T3,
    A::T,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
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
                        x1 = x[1, ix, iy, iz, it, ialpha]
                        x2 = x[2, ix, iy, iz, it, ialpha]
                        x3 = x[3, ix, iy, iz, it, ialpha]
                        y[1, ix, iy, iz, it, ialpha] =
                            x1 * A[1, 1, ix, iy, iz, it] +
                            x2 * A[2, 1, ix, iy, iz, it] +
                            x3 * A[3, 1, ix, iy, iz, it]
                        y[2, ix, iy, iz, it, ialpha] =
                            x1 * A[1, 2, ix, iy, iz, it] +
                            x2 * A[2, 2, ix, iy, iz, it] +
                            x3 * A[3, 2, ix, iy, iz, it]
                        y[3, ix, iy, iz, it, ialpha] =
                            x1 * A[1, 3, ix, iy, iz, it] +
                            x2 * A[2, 3, ix, iy, iz, it] +
                            x3 * A[3, 3, ix, iy, iz, it]
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{2},
    x::T3,
    A::T,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 2 == x.NC "dimension mismatch! NC in y is 2 but NC in x is $(x.NC)"
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
                        x1 = x[1, ix, iy, iz, it, ialpha]
                        x2 = x[2, ix, iy, iz, it, ialpha]
                        y[1, ix, iy, iz, it, ialpha] =
                            x1 * A[1, 1, ix, iy, iz, it] + x2 * A[2, 1, ix, iy, iz, it]
                        y[2, ix, iy, iz, it, ialpha] =
                            x1 * A[1, 2, ix, iy, iz, it] + x2 * A[2, 2, ix, iy, iz, it]

                    end
                end
            end
        end
    end
end



function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{NC},
    A::T,
    x::T3,
) where {NC,T<:Number,T3<:Abstractfermion}
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
                        for k1 = 1:NC
                            y[k1, ix, iy, iz, it, ialpha] =
                                A * x[k1, ix, iy, iz, it, ialpha]
                        end
                    end
                end
            end
        end
    end
end

"""
mul!(u,x,y) -> u_{ab} = x_a*y_b
"""
function LinearAlgebra.mul!(
    u::T1,
    x::AbstractFermionfields_4D{NC},
    y::AbstractFermionfields_4D{NC}; clear=true
) where {T1<:AbstractGaugefields,NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NG = x.NG
    if clear
        clear_U!(u)
    else
        #println(sum(abs.(y.f)))
        #println(sum(abs.(x.f)))
        #error("mul")
    end


    for ik = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ib = 1:NC
                            @simd for ia = 1:NC
                                c = x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]

                                u[ia, ib, ix, iy, iz, it] += c
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(u)
end

#=

function LinearAlgebra.mul!(
    u::T1,
    x::Abstractfermion,
    y::Adjoint_fermionfields{<:AbstractFermionfields_4D{NC}};clear=true
) where {T1<:AbstractGaugefields,NC}
    _, NX, NY, NZ, NT, NG = size(y)
    if clear
        clear_U!(u)
    end



    for ik = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ib = 1:NC
                            @simd for ia = 1:NC
                                u[ia, ib, ix, iy, iz, it] +=
                                    x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(u)
end

=#

#=
function LinearAlgebra.mul!(
    u::T1,
    x::Adjoint_fermionfields{<:Shifted_fermionfields_4D{NC,T}},
    y::Abstractfermion;clear=true
) where {T1<:AbstractGaugefields,NC,T}
    _, NX, NY, NZ, NT, NG = size(x)
    if clear
        clear_U!(u)
    end


    #clear_U!(u)

    for ik = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ib = 1:NC
                            @simd for ia = 1:NC
                                u[ia, ib, ix, iy, iz, it] +=
                                    x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]
                            end
                        end
                    end
                end
            end
        end
    end


    set_wing_U!(u)
end
=#

#=
function LinearAlgebra.mul!(
    u::T1,
    x::Adjoint_fermionfields{<:AbstractFermionfields_4D{NC}},
    y::Abstractfermion;clear=true
) where {T1<:AbstractGaugefields,NC}
    _, NX, NY, NZ, NT, NG = size(x)
    #clear_U!(u)
    if clear
        clear_U!(u)
    else
    #    println(sum(abs.(u.U)))
    end



    for ik = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ib = 1:NC
                            @simd for ia = 1:NC
                                c =  x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]
                                u[ia, ib, ix, iy, iz, it] += c 
                                   # x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]

                                    #if clear == false
                                    #    println("c = $c")
                                    #    println(y[ib, ix, iy, iz, it, ik])
                                    #end
                            end
                        end
                    end
                end
            end
        end
    end

    set_wing_U!(u)
end

=#

function LinearAlgebra.mul!(
    u::T1,
    x::Adjoint_fermionfields,
    y::AbstractFermionfields_4D{NC}; clear=true
) where {T1<:AbstractGaugefields,NC}
    _, NX, NY, NZ, NT, NG = size(y)
    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end



    for ik = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ib = 1:NC
                            @simd for ia = 1:NC
                                c = x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]
                                u[ia, ib, ix, iy, iz, it] += c
                                # x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]

                                #if clear == false
                                #    println("c = $c")
                                #    println(y[ib, ix, iy, iz, it, ik])
                                #end
                            end
                        end
                    end
                end
            end
        end
    end

    set_wing_U!(u)
end


function LinearAlgebra.mul!(
    u::T1,
    x::AbstractFermionfields_4D{NC},
    y::Adjoint_fermionfields, ; clear=true
) where {T1<:AbstractGaugefields,NC}
    _, NX, NY, NZ, NT, NG = size(x)
    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end



    for ik = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ib = 1:NC
                            @simd for ia = 1:NC
                                c = x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]
                                u[ia, ib, ix, iy, iz, it] += c
                                # x[ia, ix, iy, iz, it, ik] * y[ib, ix, iy, iz, it, ik]

                                #if clear == false
                                #    println("c = $c")
                                #    println(y[ib, ix, iy, iz, it, ik])
                                #end
                            end
                        end
                    end
                end
            end
        end
    end

    set_wing_U!(u)
end




function cross!(
    u::T1,
    x::Adjoint_fermionfields{<:Shifted_fermionfields_4D{NC,T}},
    y::Abstractfermion,
) where {T1<:AbstractGaugefields,NC,T}
    mul!(u, y, x)
end

function cross!(
    u::T1,
    x::AbstractFermionfields_4D{NC},
    y::Abstractfermion,
) where {T1<:AbstractGaugefields,NC}
    mul!(u, y, x)
end

function cross!(
    u::T1,
    x::Abstractfermion,
    y::Adjoint_fermionfields{<:AbstractFermionfields_4D{NC}},
) where {T1<:AbstractGaugefields,NC}
    mul!(u, y, x)
end

function cross!(
    u::T1,
    x::Adjoint_fermionfields{<:AbstractFermionfields_4D{NC}},
    y::Abstractfermion,
) where {T1<:AbstractGaugefields,NC}
    mul!(u, y, x)
end



function convert_to_normalvector(y::AbstractFermionfields_4D{NC}) where {NC}
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG
    NV = NX * NY * NZ * NT * NG * NC
    x = zeros(ComplexF64, NV)
    count = 0
    for ialpha = 1:NG
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for k1 = 1:NC
                            count += 1
                            x[count] = y[k1, ix, iy, iz, it, ialpha]
                        end
                    end
                end
            end
        end
    end
    return x
end


"""
mul!(y,A,x,α,β) -> α*A*x+β*y -> y
"""
function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{NC},
    A::T,
    x::T3,
    α::TA,
    β::TB,
) where {NC,T<:Number,T3<:Abstractfermion,TA<:Number,TB<:Number}
    @assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG
    if A == one(A)
        @inbounds for ialpha = 1:NG
            for it = 1:NT
                for iz = 1:NZ
                    for iy = 1:NY
                        for ix = 1:NX
                            for k1 = 1:NC
                                y[k1, ix, iy, iz, it, ialpha] =
                                    α * x[k1, ix, iy, iz, it, ialpha] +
                                    β * y[k1, ix, iy, iz, it, ialpha]
                            end
                        end
                    end
                end
            end
        end
    else
        @inbounds for ialpha = 1:NG
            for it = 1:NT
                for iz = 1:NZ
                    for iy = 1:NY
                        for ix = 1:NX
                            for k1 = 1:NC
                                y[k1, ix, iy, iz, it, ialpha] =
                                    A * α * x[k1, ix, iy, iz, it, ialpha] +
                                    β * y[k1, ix, iy, iz, it, ialpha]
                            end
                        end
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields_4D{NC},
    A::T,
    x::T3,
    α::TA,
    β::TB,
    iseven::Bool,
) where {NC,T<:Number,T3<:Abstractfermion,TA<:Number,TB<:Number}
    @assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG
    if A == one(A)
        @inbounds for ialpha = 1:NG
            for it = 1:NT
                for iz = 1:NZ
                    for iy = 1:NY
                        for ix = 1:NX
                            evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                            if evenodd == iseven
                                for k1 = 1:NC
                                    y[k1, ix, iy, iz, it, ialpha] =
                                        α * x[k1, ix, iy, iz, it, ialpha] +
                                        β * y[k1, ix, iy, iz, it, ialpha]
                                end
                            end
                        end
                    end
                end
            end
        end
    else
        @inbounds for ialpha = 1:NG
            for it = 1:NT
                for iz = 1:NZ
                    for iy = 1:NY
                        for ix = 1:NX
                            evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                            if evenodd == iseven
                                for k1 = 1:NC
                                    y[k1, ix, iy, iz, it, ialpha] =
                                        A * α * x[k1, ix, iy, iz, it, ialpha] +
                                        β * y[k1, ix, iy, iz, it, ialpha]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    a::Number,
    X::T,
    b::Number,
    Y::AbstractFermionfields_4D{NC},
) where {NC,T<:AbstractFermionfields_4D}
    n1, n2, n3, n4, n5, n6 = size(Y.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            for i4 = 1:n4
                for i3 = 1:n3
                    for i2 = 1:n2
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
    return Y
end


function LinearAlgebra.axpy!(
    a::Number,
    X::T,
    Y::AbstractFermionfields_4D{NC},
) where {NC,T<:AbstractFermionfields_4D}
    LinearAlgebra.axpby!(a, X, 1, Y)
    return Y
end

function Base.:*(a::Number, x::AbstractFermionfields_4D{NC}) where {NC}
    y = similar(x)
    n1, n2, n3, n4, n5, n6 = size(y.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            for i4 = 1:n4
                for i3 = 1:n3
                    for i2 = 1:n2
                        @simd for i1 = 1:NC
                            y.f[i1, i2, i3, i4, i5, i6] = a * x.f[i1, i2, i3, i4, i5, i6]
                        end
                    end
                end
            end
        end
    end
    return y
end

function LinearAlgebra.rmul!(x::AbstractFermionfields_4D{NC}, a::Number) where {NC}
    n1, n2, n3, n4, n5, n6 = size(x.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            for i4 = 1:n4
                for i3 = 1:n3
                    for i2 = 1:n2
                        @simd for i1 = 1:NC
                            x.f[i1, i2, i3, i4, i5, i6] = a * x.f[i1, i2, i3, i4, i5, i6]
                        end
                    end
                end
            end
        end
    end
    return x
end




function add_fermion!(
    c::AbstractFermionfields_4D{NC},
    α::Number,
    a::T1,
    β::Number,
    b::T2,
) where {NC,T1<:Abstractfermion,T2<:Abstractfermion}#c += alpha*a + beta*b
    n1, n2, n3, n4, n5, n6 = size(c.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            for i4 = 1:n4
                for i3 = 1:n3
                    for i2 = 1:n2
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
    return
end


function add_fermion!(
    c::AbstractFermionfields_4D{NC},
    α::Number,
    a::T1,
) where {NC,T1<:Abstractfermion}#c += alpha*a 
    n1, n2, n3, n4, n5, n6 = size(c.f)

    @inbounds for i6 = 1:n6
        for i5 = 1:n5
            for i4 = 1:n4
                for i3 = 1:n3
                    for i2 = 1:n2
                        @simd for i1 = 1:NC
                            #println(a.f[i1,i2,i3,i4,i5,i6],"\t",b.f[i1,i2,i3,i4,i5,i6] )
                            c.f[i1, i2, i3, i4, i5, i6] += α * a.f[i1, i2, i3, i4, i5, i6]

                        end
                    end
                end
            end
        end
    end
    #if (i2, i3, i4, i5) == (1, 1, 1, 1)
    #display(c.f[:, 1, 1, 1, 1, :])
    #println("a α = $α")
    #display(a.f[:, 1, 1, 1, 1, :])
    #end
    return
end

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
#=
function gauss_distribution_fermion!(x::AbstractFermionfields_4D{NC}) where {NC}
NX = x.NX
NY = x.NY
NZ = x.NZ
NT = x.NT
n6 = size(x.f)[6]
σ = sqrt(1 / 2)

for ialpha = 1:n6
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for ic = 1:NC
                        x[ic, ix, iy, iz, it, ialpha] = σ * randn() + im * σ * randn()
                    end
                end
            end
        end
    end
end
set_wing_fermion!(x)
return
end
=#

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(
    x::AbstractFermionfields_4D{NC},
    randomfunc,
    σ,
) where {NC}

    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.f)[6]
    #σ = sqrt(1/2)

    for mu = 1:n6
        for ic = 1:NC
            for it = 1:NT
                for iz = 1:NZ
                    for iy = 1:NY
                        for ix = 1:NX
                            v1 = sqrt(-log(randomfunc() + 1e-10))
                            v2 = 2pi * randomfunc()

                            xr = v1 * cos(v2)
                            xi = v1 * sin(v2)

                            x[ic, ix, iy, iz, it, mu] = σ * xr + σ * im * xi
                        end
                    end
                end
            end
        end
    end

    set_wing_fermion!(x)

    return
end

function gauss_distribution_fermion!(x::AbstractFermionfields_4D{NC}) where {NC}
    σ = 1
    gauss_distribution_fermion!(x, rand, σ)
end

function gauss_distribution_fermion!(x::AbstractFermionfields_4D{NC}, randomfunc) where {NC}
    σ = 1
    gauss_distribution_fermion!(x, randomfunc, σ)
end

function Z2_distribution_fermion!(x::AbstractFermionfields_4D{NC}) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.f)[6]
    #σ = sqrt(1/2)

    for mu = 1:n6
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        @inbounds @simd for ic = 1:NC
                            x[ic, ix, iy, iz, it, mu] = rand([-1, 1])
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
c     Random number function Z4  Noise
c     https://arxiv.org/pdf/1611.01193.pdf
c-------------------------------------------------c
    """
function Z4_distribution_fermi!(x::AbstractFermionfields_4D{NC}) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.f)[6]
    θ = 0.0
    N::Int32 = 4
    Ninv = Float64(1 / N)
    for ialpha = 1:n6
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        @inbounds @simd for ic = 1:NC
                            θ = Float64(rand(0:N-1)) * π * Ninv # r \in [0,π/4,2π/4,3π/4]
                            x[ic, ix, iy, iz, it, ialpha] = cos(θ) + im * sin(θ)
                        end
                    end
                end
            end
        end
    end

    set_wing_fermion!(x)

    return
end

function uniform_distribution_fermion!(x::AbstractFermionfields_4D{NC}) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.f)[6]
    #σ = sqrt(1/2)

    for mu = 1:n6
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ic = 1:NC
                            x[ic, ix, iy, iz, it, mu] = rand() * 2 - 1 #each element should be in (-1,1)   
                        end
                    end
                end
            end
        end
    end

    set_wing_fermion!(x)

    return
end

function fermion2vector!(vector, x::AbstractFermionfields_4D{NC}) where {NC}
    vector .= 0
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.f)[6]
    #σ = sqrt(1/2)

    count = 0
    for mu = 1:n6
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for ic = 1:NC
                            count += 1
                            vector[count] = x[ic, ix, iy, iz, it, mu]
                        end
                    end
                end
            end
        end
    end

end

function fermions2vectors!(vector, x::Vector{<:AbstractFermionfields_4D{NC}}) where {NC}
    M = length(x)
    n, m = size(vector)
    @assert M == m

    vector .= 0
    NX = x[1].NX
    NY = x[1].NY
    NZ = x[1].NZ
    NT = x[1].NT
    n6 = size(x[1].f)[6]
    #σ = sqrt(1/2)

    for im = 1:M
        count = 0
        for mu = 1:n6
            for it = 1:NT
                for iz = 1:NZ
                    for iy = 1:NY
                        for ix = 1:NX
                            for ic = 1:NC
                                count += 1
                                vector[count, im] = x[im][ic, ix, iy, iz, it, mu]
                            end
                        end
                    end
                end
            end
        end
    end

end


function LinearAlgebra.dot(
    a::AbstractFermionfields_4D{NC},
    b::AbstractFermionfields_4D{NC},
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
                        @simd for ic = 1:NC
                            c += conj(a[ic, ix, iy, iz, it, α]) * b[ic, ix, iy, iz, it, α]
                        end
                    end
                end
            end
        end
    end
    return c
end


function apply_σμν!(a, μ, ν, b)
    @assert μ != ν """μ should not be equal to ν 
    μ,ν : $μ $ν
    """
    @assert 1 <= μ <= 4 "μ should be 1,2,or 3. now $μ"
    @assert 1 <= ν <= 4 "μ should be 1,2,or 3. now $ν"

    σ = σμν(μ, ν)
    clear_fermion!(a)
    apply_σ!(a, σ, b)
end

#=
function apply_σ!(a::Abstractfermion,σ::σμν{μ,ν},b::Abstractfermion;factor=1) where {μ,ν}
    error("apply_σ! is not implemented in type a:$(typeof(a)),b:$(typeof(b))")
end
=#

function apply_σ!(a::AbstractFermionfields_4D{NC}, σ::σμν{μ,ν}, b::AbstractFermionfields_4D{NC}; factor=1) where {NC,μ,ν}
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

function apply_σ!(a::AbstractFermionfields_4D{NC}, σ::σμν{μ,ν}, b::Shifted_fermionfields{NC,4}; factor=1) where {NC,μ,ν}
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


function Ux_ν!(
    y::AbstractFermionfields_4D{NC},
    A::T,
    x::T3,
    ν::Integer;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion,NC}

    xplus = shift_fermion(x, ν; boundarycondition)
    mul!(y, A, xplus)

end

struct Uxplusminus_ν
end

function Uxplus_1!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=
                        ix_shifted = ix + 1
                        inside_up = ix_shifted > NX
                        factor_x = ifelse(inside_up, boundarycondition[1], 1)
                        ix_shifted += ifelse(inside_up, -NX, 0)

                        x1 = x[1, ix_shifted, iy, iz, it, ialpha] * factor_x
                        x2 = x[2, ix_shifted, iy, iz, it, ialpha] * factor_x
                        x3 = x[3, ix_shifted, iy, iz, it, ialpha] * factor_x

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end


function Uxminus_1!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=
                        ix_shifted = ix - 1
                        inside_down = ix_shifted < 1
                        factor_x = ifelse(inside_down, boundarycondition[1], 1)
                        ix_shifted += ifelse(inside_down, NX, 0)

                        x1 = x[1, ix_shifted, iy, iz, it, ialpha] * factor_x
                        x2 = x[2, ix_shifted, iy, iz, it, ialpha] * factor_x
                        x3 = x[3, ix_shifted, iy, iz, it, ialpha] * factor_x

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end



function Uxplus_2!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    iy_shifted = iy + 1
                    inside_up = iy_shifted > NY
                    factor = ifelse(inside_up, boundarycondition[2], 1)
                    iy_shifted += ifelse(inside_up, -NY, 0)

                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy_shifted, iz, it, ialpha] * factor
                        x2 = x[2, ix, iy_shifted, iz, it, ialpha] * factor
                        x3 = x[3, ix, iy_shifted, iz, it, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end


function Uxminus_2!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    iy_shifted = iy - 1
                    inside_down = iy_shifted < 1
                    factor = ifelse(inside_down, boundarycondition[2], 1)
                    iy_shifted += ifelse(inside_down, NY, 0)

                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy_shifted, iz, it, ialpha] * factor
                        x2 = x[2, ix, iy_shifted, iz, it, ialpha] * factor
                        x3 = x[3, ix, iy_shifted, iz, it, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end




function Uxplus_3!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                iz_shifted = iz + 1
                inside_up = iz_shifted > NZ
                factor = ifelse(inside_up, boundarycondition[3], 1)
                iz_shifted += ifelse(inside_up, -NZ, 0)

                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy, iz_shifted, it, ialpha] * factor
                        x2 = x[2, ix, iy, iz_shifted, it, ialpha] * factor
                        x3 = x[3, ix, iy, iz_shifted, it, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end


function Uxminus_3!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                iz_shifted = iz - 1
                inside_down = iz_shifted < 1
                factor = ifelse(inside_down, boundarycondition[3], 1)
                iz_shifted += ifelse(inside_down, NZ, 0)

                for iy = 1:NY

                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy, iz_shifted, it, ialpha] * factor
                        x2 = x[2, ix, iy, iz_shifted, it, ialpha] * factor
                        x3 = x[3, ix, iy, iz_shifted, it, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end



function Uxplus_4!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            it_shifted = it + 1
            inside_up = it_shifted > NT
            factor = ifelse(inside_up, boundarycondition[4], 1)
            it_shifted += ifelse(inside_up, -NT, 0)

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ

                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy, iz, it_shifted, ialpha] * factor
                        x2 = x[2, ix, iy, iz, it_shifted, ialpha] * factor
                        x3 = x[3, ix, iy, iz, it_shifted, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end


function Uxminus_4!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            it_shifted = it - 1
            inside_down = it_shifted < 1
            factor = ifelse(inside_down, boundarycondition[4], 1)
            it_shifted += ifelse(inside_down, NT, 0)

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ

                for iy = 1:NY

                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy, iz, it_shifted, ialpha] * factor
                        x2 = x[2, ix, iy, iz, it_shifted, ialpha] * factor
                        x3 = x[3, ix, iy, iz, it_shifted, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it] * x1 +
                            A[1, 2, ix, iy, iz, it] * x2 +
                            A[1, 3, ix, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it] * x1 +
                            A[2, 2, ix, iy, iz, it] * x2 +
                            A[2, 3, ix, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it] * x1 +
                            A[3, 2, ix, iy, iz, it] * x2 +
                            A[3, 3, ix, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end

"""
(Ux)
"""
function Ux_afterν!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3,
    ν::Integer;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}



    mul!(y, A, x)
    set_wing_fermion!(y)
    x_shifted = shift_fermion(y, ν; boundarycondition)
    substitute_fermion!(y, x_shifted)

end


function Uxplus_after1!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=
                        ix_shifted = ix + 1
                        inside_up = ix_shifted > NX
                        factor_x = ifelse(inside_up, boundarycondition[1], 1)
                        ix_shifted += ifelse(inside_up, -NX, 0)

                        x1 = x[1, ix_shifted, iy, iz, it, ialpha] * factor_x
                        x2 = x[2, ix_shifted, iy, iz, it, ialpha] * factor_x
                        x3 = x[3, ix_shifted, iy, iz, it, ialpha] * factor_x

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix_shifted, iy, iz, it] * x1 +
                            A[1, 2, ix_shifted, iy, iz, it] * x2 +
                            A[1, 3, ix_shifted, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix_shifted, iy, iz, it] * x1 +
                            A[2, 2, ix_shifted, iy, iz, it] * x2 +
                            A[2, 3, ix_shifted, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix_shifted, iy, iz, it] * x1 +
                            A[3, 2, ix_shifted, iy, iz, it] * x2 +
                            A[3, 3, ix_shifted, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end


function Uxminus_after1!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=
                        ix_shifted = ix - 1
                        inside_down = ix_shifted < 1
                        factor_x = ifelse(inside_down, boundarycondition[1], 1)
                        ix_shifted += ifelse(inside_down, NX, 0)

                        x1 = x[1, ix_shifted, iy, iz, it, ialpha] * factor_x
                        x2 = x[2, ix_shifted, iy, iz, it, ialpha] * factor_x
                        x3 = x[3, ix_shifted, iy, iz, it, ialpha] * factor_x

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix_shifted, iy, iz, it] * x1 +
                            A[1, 2, ix_shifted, iy, iz, it] * x2 +
                            A[1, 3, ix_shifted, iy, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix_shifted, iy, iz, it] * x1 +
                            A[2, 2, ix_shifted, iy, iz, it] * x2 +
                            A[2, 3, ix_shifted, iy, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix_shifted, iy, iz, it] * x1 +
                            A[3, 2, ix_shifted, iy, iz, it] * x2 +
                            A[3, 3, ix_shifted, iy, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end



function Uxplus_after2!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    iy_shifted = iy + 1
                    inside_up = iy_shifted > NY
                    factor = ifelse(inside_up, boundarycondition[2], 1)
                    iy_shifted += ifelse(inside_up, -NY, 0)

                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy_shifted, iz, it, ialpha] * factor
                        x2 = x[2, ix, iy_shifted, iz, it, ialpha] * factor
                        x3 = x[3, ix, iy_shifted, iz, it, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy_shifted, iz, it] * x1 +
                            A[1, 2, ix, iy_shifted, iz, it] * x2 +
                            A[1, 3, ix, iy_shifted, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy_shifted, iz, it] * x1 +
                            A[2, 2, ix, iy_shifted, iz, it] * x2 +
                            A[2, 3, ix, iy_shifted, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy_shifted, iz, it] * x1 +
                            A[3, 2, ix, iy_shifted, iz, it] * x2 +
                            A[3, 3, ix, iy_shifted, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end


function Uxminus_after2!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                for iy = 1:NY
                    iy_shifted = iy - 1
                    inside_down = iy_shifted < 1
                    factor = ifelse(inside_down, boundarycondition[2], 1)
                    iy_shifted += ifelse(inside_down, NY, 0)

                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy_shifted, iz, it, ialpha] * factor
                        x2 = x[2, ix, iy_shifted, iz, it, ialpha] * factor
                        x3 = x[3, ix, iy_shifted, iz, it, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy_shifted, iz, it] * x1 +
                            A[1, 2, ix, iy_shifted, iz, it] * x2 +
                            A[1, 3, ix, iy_shifted, iz, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy_shifted, iz, it] * x1 +
                            A[2, 2, ix, iy_shifted, iz, it] * x2 +
                            A[2, 3, ix, iy_shifted, iz, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy_shifted, iz, it] * x1 +
                            A[3, 2, ix, iy_shifted, iz, it] * x2 +
                            A[3, 3, ix, iy_shifted, iz, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end




function Uxplus_after3!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                iz_shifted = iz + 1
                inside_up = iz_shifted > NZ
                factor = ifelse(inside_up, boundarycondition[3], 1)
                iz_shifted += ifelse(inside_up, -NZ, 0)

                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy, iz_shifted, it, ialpha] * factor
                        x2 = x[2, ix, iy, iz_shifted, it, ialpha] * factor
                        x3 = x[3, ix, iy, iz_shifted, it, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz_shifted, it] * x1 +
                            A[1, 2, ix, iy, iz_shifted, it] * x2 +
                            A[1, 3, ix, iy, iz_shifted, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz_shifted, it] * x1 +
                            A[2, 2, ix, iy, iz_shifted, it] * x2 +
                            A[2, 3, ix, iy, iz_shifted, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz_shifted, it] * x1 +
                            A[3, 2, ix, iy, iz_shifted, it] * x2 +
                            A[3, 3, ix, iy, iz_shifted, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end


function Uxminus_after3!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ
                iz_shifted = iz - 1
                inside_down = iz_shifted < 1
                factor = ifelse(inside_down, boundarycondition[3], 1)
                iz_shifted += ifelse(inside_down, NZ, 0)

                for iy = 1:NY

                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy, iz_shifted, it, ialpha] * factor
                        x2 = x[2, ix, iy, iz_shifted, it, ialpha] * factor
                        x3 = x[3, ix, iy, iz_shifted, it, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz_shifted, it] * x1 +
                            A[1, 2, ix, iy, iz_shifted, it] * x2 +
                            A[1, 3, ix, iy, iz_shifted, it] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz_shifted, it] * x1 +
                            A[2, 2, ix, iy, iz_shifted, it] * x2 +
                            A[2, 3, ix, iy, iz_shifted, it] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz_shifted, it] * x1 +
                            A[3, 2, ix, iy, iz_shifted, it] * x2 +
                            A[3, 3, ix, iy, iz_shifted, it] * x3
                        # =#
                    end
                end
            end
        end
    end
end



function Uxplus_after4!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            it_shifted = it + 1
            inside_up = it_shifted > NT
            factor = ifelse(inside_up, boundarycondition[4], 1)
            it_shifted += ifelse(inside_up, -NT, 0)

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ

                for iy = 1:NY
                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy, iz, it_shifted, ialpha] * factor
                        x2 = x[2, ix, iy, iz, it_shifted, ialpha] * factor
                        x3 = x[3, ix, iy, iz, it_shifted, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it_shifted] * x1 +
                            A[1, 2, ix, iy, iz, it_shifted] * x2 +
                            A[1, 3, ix, iy, iz, it_shifted] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it_shifted] * x1 +
                            A[2, 2, ix, iy, iz, it_shifted] * x2 +
                            A[2, 3, ix, iy, iz, it_shifted] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it_shifted] * x1 +
                            A[3, 2, ix, iy, iz, it_shifted] * x2 +
                            A[3, 3, ix, iy, iz, it_shifted] * x3
                        # =#
                    end
                end
            end
        end
    end
end


function Uxminus_after4!(
    y::AbstractFermionfields_4D{3},
    A::T,
    x::T3;
    boundarycondition=(1, 1, 1, -1)
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha = 1:NG
        for it = 1:NT
            it_shifted = it - 1
            inside_down = it_shifted < 1
            factor = ifelse(inside_down, boundarycondition[4], 1)
            it_shifted += ifelse(inside_down, NT, 0)

            #println("it = ",it, " ialpha = $ialpha")
            for iz = 1:NZ

                for iy = 1:NY

                    for ix = 1:NX
                        #updatefunc!(y,A,x,ix,iy,iz,it,ialpha)
                        #error("oo")
                        # #=

                        x1 = x[1, ix, iy, iz, it_shifted, ialpha] * factor
                        x2 = x[2, ix, iy, iz, it_shifted, ialpha] * factor
                        x3 = x[3, ix, iy, iz, it_shifted, ialpha] * factor

                        y[1, ix, iy, iz, it, ialpha] =
                            A[1, 1, ix, iy, iz, it_shifted] * x1 +
                            A[1, 2, ix, iy, iz, it_shifted] * x2 +
                            A[1, 3, ix, iy, iz, it_shifted] * x3
                        y[2, ix, iy, iz, it, ialpha] =
                            A[2, 1, ix, iy, iz, it_shifted] * x1 +
                            A[2, 2, ix, iy, iz, it_shifted] * x2 +
                            A[2, 3, ix, iy, iz, it_shifted] * x3
                        y[3, ix, iy, iz, it, ialpha] =
                            A[3, 1, ix, iy, iz, it_shifted] * x1 +
                            A[3, 2, ix, iy, iz, it_shifted] * x2 +
                            A[3, 3, ix, iy, iz, it_shifted] * x3
                        # =#
                    end
                end
            end
        end
    end
end



const γ1indices = (4, 3, 2, 1)
const γ1coeffs = (-im, -im, im, im)
const γ2indices = (4, 3, 2, 1)
const γ2coeffs = (-1, 1, 1, -1)
const γ3indices = (3, 4, 1, 2)
const γ3coeffs = (-im, im, im, -im)
const γ4indices = (3, 4, 1, 2)
const γ4coeffs = (-1, -1, -1, -1)

const γindices = (γ1indices, γ2indices, γ3indices, γ4indices)
const γcoeffs = (γ1coeffs, γ2coeffs, γ3coeffs, γ4coeffs)
