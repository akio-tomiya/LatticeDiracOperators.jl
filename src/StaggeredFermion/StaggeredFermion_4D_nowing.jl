import Gaugefields: staggered_U

struct StaggeredFermion_4D_nowing{NC} <: AbstractFermionfields_4D{NC}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NG::Int64 #size of the Gamma matrix. In Staggered fermion, this is one. 
    NV::Int64
    f::Array{ComplexF64,6}
    Dirac_operator::String
    fshifted::Array{ComplexF64,6}


    function StaggeredFermion_4D_nowing(NC, NX, NY, NZ, NT)
        NG = 1
        NDW = 0
        NV = NC * NX * NY * NZ * NT * NG
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW, NG)
        fshifted = zero(f)
        Dirac_operator = "Staggered"
        return new{NC}(NC, NX, NY, NZ, NT, NDW, NG, NV, f, Dirac_operator, fshifted)
    end
end

function Base.size(x::StaggeredFermion_4D_nowing{NC}) where {NC}
    return (x.NC, x.NX, x.NY, x.NZ, x.NT, x.NG)
    #return (x.NV,)
end

function Base.length(x::StaggeredFermion_4D_nowing{NC}) where {NC}
    return NC * x.NX * x.NY * x.NZ * x.NT * x.NG
end

function Base.similar(x::T) where {T<:StaggeredFermion_4D_nowing}
    return StaggeredFermion_4D_nowing(x.NC, x.NX, x.NY, x.NZ, x.NT)
end


function Dx!(
    xout::T,
    U::Array{G,1},
    x::T,
    temps::Array{T,1},
    boundarycondition,
) where {T<:StaggeredFermion_4D_nowing,G<:AbstractGaugefields}
    #temp = temps[4]
    temp1 = temps[1]
    temp2 = temps[2]

    #clear!(temp)
    set_wing_fermion!(x, boundarycondition)
    clear_fermion!(xout)
    for ν = 1:4
        xplus = shift_fermion(x, ν; boundarycondition)
        Us = staggered_U(U[ν], ν)
        mul!(temp1, Us, xplus)


        xminus = shift_fermion(x, -ν; boundarycondition)
        Uminus = shift_U(U[ν], -ν)
        Uminus_s = staggered_U(Uminus, ν)
        mul!(temp2, Uminus_s', xminus)

        add_fermion!(xout, 0.5, temp1, -0.5, temp2)

        #fermion_shift!(temp1,U,ν,x)
        #fermion_shift!(temp2,U,-ν,x)
        #add!(xout,0.5,temp1,-0.5,temp2)

    end


    set_wing_fermion!(xout, boundarycondition)

    return
end

function clear_fermion!(x::StaggeredFermion_4D_nowing{NC}, evensite) where {NC}
    ibush = ifelse(evensite, 0, 1)
    for it = 1:x.NT
        for iz = 1:x.NZ
            for iy = 1:x.NY
                xran = 1+(1+ibush+iy+iz+it)%2:2:x.NX
                for ix in xran
                    @simd for ic = 1:NC
                        x[ic, ix, iy, iz, it, 1] = 0
                    end
                end
            end
        end
    end
    return
end

function shift_fermion(F::StaggeredFermion_4D_nowing{NC}, ν::T;
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
) where {Dim,T<:Integer,TF<:StaggeredFermion_4D_nowing}
    return Shifted_fermionfields_4D_nowing(F, shift; boundarycondition)
end


function shifted_fermion!(
    x::StaggeredFermion_4D_nowing{NC},
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
    for ig = 1:1
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


function set_wing_fermion!(a::StaggeredFermion_4D_nowing{NC}, boundarycondition) where {NC}
    return
end
