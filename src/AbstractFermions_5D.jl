abstract type AbstractFermionfields_5D{NC} <: AbstractFermionfields{NC,5}
end

function Base.setindex!(x::AbstractFermionfields_5D{NC},v,i1,i2,i3,i4,i5,i6,i7)  where {NC}
    @inbounds x.w[i7][i1,i2,i3,i4,i5,i6] = v
end

function Base.getindex(x::AbstractFermionfields_5D{NC},i1,i2,i3,i4,i5,i6,i7) where {NC}
    @inbounds return x.w[i7][i1,i2,i3,i4,i5,i6] 
end

Base.length(x::T) where T <: AbstractFermionfields_5D = x.L5*length(x.w[1])

function Base.setindex!(x::T,v,i)  where T <: AbstractFermionfields_5D
    #i = (iL-1)*NN+index
    NN = length(x.w[1])
    i4D = (i-1) % NN + 1
    iL = (i-i4D) ÷ NN + 1
    @inbounds x.w[iL][i4D] = v
end

function Base.getindex(x::T,i) where T <: AbstractFermionfields_5D
    #i = (iL-1)*NN+index
    NN = length(x.w[1])
    i4D = (i-1) % NN + 1
    iL = (i-i4D) ÷ NN + 1
    return @inbounds x.w[iL][i4D]
end

function save_fermionfield(a::AbstractFermionfields_5D{NC},filename) where NC
    jldsave(filename; ϕ=a)
    return
end

function load_fermionfield!(a::AbstractFermionfields_5D{NC},filename) where NC
    jldopen(filename, "r") do file
        substitute_fermion!(a,file["ϕ"])
    end
end


function substitute_fermion!(a::AbstractFermionfields_5D{NC},b::AbstractFermionfields_5D{NC}) where NC 
    L5 = length(a.w)
    n1,n2,n3,n4,n5,n6 = size(a.w[1].f)
    for iL = 1:L5
        @inbounds for i6=1:n6
            for i5=1:n5
                for i4=1:n4
                    for i3=1:n3
                        for i2=1:n2
                            @simd for i1=1:NC
                                a.w[iL].f[i1,i2,i3,i4,i5,i6]= b.w[iL].f[i1,i2,i3,i4,i5,i6]
                            end
                        end
                    end
                end
            end
        end
    end
end

function clear_fermion!(a::Vector{<: AbstractFermionfields_5D{NC}}) where NC 
    for μ=1:length(a)
        clear_fermion!(a[μ])
    end
end

function clear_fermion!(a::AbstractFermionfields_5D{NC}) where NC 
    L5 = length(a.w)
    n1,n2,n3,n4,n5,n6 = size(a.w[1].f)
    for iL = 1:L5
        @inbounds for i6=1:n6
            for i5=1:n5
                for i4=1:n4
                    for i3=1:n3
                        for i2=1:n2
                            @simd for i1=1:NC
                                a.w[iL].f[i1,i2,i3,i4,i5,i6]= 0
                            end
                        end
                    end
                end
            end
        end
    end
end

function add_fermion!(c::AbstractFermionfields_5D{NC},α::Number,a::T1,β::Number,b::T2) where {NC,T1 <: Abstractfermion,T2 <: Abstractfermion}#c += alpha*a + beta*b
    L5 = length(a.w)
    for iL=1:L5
        add_fermion!(c.w[iL],α,a.w[iL],β,b.w[iL])
    end
end

function set_wing_fermion!(F::AbstractFermionfields_5D{NC})  where NC
    set_wing_fermion!(F,default_boundaryconditions[4])
end

function set_wing_fermion!(F::AbstractFermionfields_5D{NC},boundarycondition)  where NC
    L5 = length(F.w)
    for iL = 1:L5
        set_wing_fermion!(F.w[iL],boundarycondition)
    end
end




"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::AbstractFermionfields_5D{NC}) where NC
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.w[1].f)[6]
    σ = sqrt(1/2)

    L5 = length(x.w)
    for iL = 1:L5
        for ialpha = 1:n6
            for it=1:NT
                for iz=1:NZ
                    for iy=1:NY
                        for ix=1:NX
                            for ic=1:NC 
                                x.w[iL][ic,ix,iy,iz,it,ialpha] = σ*randn()+im*σ*randn()
                            end
                        end
                    end
                end
            end
        end
        set_wing_fermion!(x.w[iL])
    end
    return
end

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::AbstractFermionfields_5D{NC},randomfunc,σ) where NC
  
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.w[1].f)[6]
    #σ = sqrt(1/2)

    L5 = length(x.w)
    @inbounds for iL = 1:L5

        for mu = 1:n6
            for ic=1:NC
                for it=1:NT
                    for iz=1:NZ
                        for iy=1:NY
                            for ix=1:NX
                                v1 = sqrt(-log(randomfunc()+1e-10))
                                v2 = 2pi*randomfunc()

                                xr = v1*cos(v2)
                                xi = v1 * sin(v2)

                                x.w[iL][ic,ix,iy,iz,it,mu] = σ*xr + σ*im*xi
                            end
                        end
                    end
                end
            end
        end

        set_wing_fermion!(x.w[iL])
    end

    return
end

function gauss_distribution_fermion!(x::AbstractFermionfields_5D{NC},randomfunc) where NC
    σ = 1
    gauss_distribution_fermion!(x,randomfunc,σ)
end

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(a::Number, X::T, b::Number, Y::AbstractFermionfields_5D{NC}) where {NC,T <: AbstractFermionfields_5D}
    L5 = length(X.w)
    #n1,n2,n3,n4,n5,n6 = size(X.w[1].f)
    for iL = 1:L5
        Y.w[iL] = axpby!(a,X.w[iL],b,Y.w[iL])
    end
    return Y
end

function LinearAlgebra.axpy!(a::Number, X::T, Y::AbstractFermionfields_5D{NC}) where {NC,T <: AbstractFermionfields_5D}
    LinearAlgebra.axpby!(a,X,1,Y)
    return Y
end

function LinearAlgebra.dot(a::AbstractFermionfields_5D{NC},b::AbstractFermionfields_5D{NC}) where {NC}
    L5 = length(a.w)
    c = 0.0im
    for iL = 1:L5
        c += dot(a.w[iL],b.w[iL])
    end
    return c
end

