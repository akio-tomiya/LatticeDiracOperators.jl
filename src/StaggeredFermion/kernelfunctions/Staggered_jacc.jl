include("./jacckernel_staggared.jl")

function gauss_distribution_fermion!(
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    randomfunc,
    σ
) where {NC,TF}

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_gauss_distribution_staggeredfermion!, x.f,
        σ, NC)
    #end
    return
end

function gauss_distribution_fermion!(
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc}
) where {NC,TF}

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_gauss_distribution_staggeredfermion!, x.f,
        1, NC)
    #end


    return
end

function clear_fermion!(a::StaggeredFermion_4D_accelerator{NC,TF,:jacc}) where {NC,TF}
    #CUDA.@sync begin
    N = a.NX * a.NY * a.NZ * a.NT
    JACC.parallel_for(N, jacckernel_clear_staggeredfermion!, a.f, NC)
    #end
end



function add_fermion!(
    c::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    α::Number,
    a::T1,
    β::Number,
    B::T1,
) where {NC,T1<:StaggeredFermion_4D_accelerator,TF}#c += alpha*a 
    N = c.NX * c.NY * c.NZ * c.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_add_staggeredfermion!, c.f, α, a.f, β, B.f, NC)
    # end

end



function add_fermion!(
    c::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    α::Number,
    a::T1,
) where {NC,T1<:StaggeredFermion_4D_accelerator,TF}#c += alpha*a 
    N = c.NX * c.NY * c.NZ * c.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_add_staggeredfermion!, c.f, α, a.f, NC)
    #end

end


function shifted_fermion!(
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    boundarycondition,
    shift,
) where {NC,TF}

    bc = Tuple(boundarycondition)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    N = x.NX * x.NY * x.NZ * x.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_shifted_staggeredfermion!, x.f, x.fshifted, bc, shift, NC, NX, NY, NZ, NT)
    #end


end

function shifted_fermion!(
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc,TUv,TFshifted},
    boundarycondition,
    shift,
) where {NC,TF,TUv,TFshifted<:Nothing}

end




function LinearAlgebra.dot(
    A::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    B::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
) where {NC,TF}

    N = A.NX * A.NY * A.NZ * A.NT
    #CUDA.@sync begin
    s = JACC.parallel_reduce(N, +, jacckernel_staggereddot!, A.f, B.f, NC; init=zero(eltype(A.f)))
    #end
    #s = CUDA.reduce(+, A.temp_volume)

    return s
end

function substitute_fermion!(
    A::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    B::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
) where {NC,TF}

    N = A.NX * A.NY * A.NZ * A.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_substitute_staggeredfermion!, A.f, B.f, NC)
    #end


end


function substitute_fermion!(
    A::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    B::AbstractFermionfields_4D{NC},
) where {NC,TF}
    acpu = Array(A.f)

    N = A.NX * A.NY * A.NZ * A.NT
    for i = 1:N
        ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
        #for ig = 1:NG
        for ic = 1:NC
            acpu[ic, i] = B[ic, ix, iy, iz, it]
        end
        #end
    end
    #blockinfo = A.blockinfo

    agpu = JACC.Array(acpu)
    A.f .= agpu
end

function substitute_fermion!(
    A::AbstractFermionfields_4D{NC},
    B::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
) where {NC,TF}
    bcpu = Array(B.f)

    N = B.NX * B.NY * B.NZ * B.NT
    for i = 1:N
        ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
        #for ig = 1:NG
        for ic = 1:NC
            A[ic, ix, iy, iz, it] = bcpu[ic, i]
        end
        #end
    end

end