include("./jacckernel_wilson.jl")

function gauss_distribution_fermion!(
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    randomfunc,
    σ
) where {NC,TF,NG}

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_gauss_distribution_fermion!, x.f,
        σ, NC, NG)
    #end
    return
end

function gauss_distribution_fermion!(
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc}
) where {NC,TF,NG}

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_gauss_distribution_fermion!, x.f,
        1, NC, NG)
    #end


    return
end

function clear_fermion!(a::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc}) where {NC,TF,NG}
    #CUDA.@sync begin
    N = a.NX * a.NY * a.NZ * a.NT
    JACC.parallel_for(N, jacckernel_clear_fermion!, a.f, NC, NG)
    #end
end

#=
function add_fermion!(
    c:::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    α::Number,
    a::T1,
    β::Number,
    B::T2,
) where {NC,T1<:WilsonFermion_4D_accelerator,T2<:Abstractfermion,TF ,NG}#c += alpha*a + beta*b
    #CUDA.@sync begin
        JACC.parallel_for(N,jacckernel_add_fermion!(c.f, α, a.f,  β, B.f,NC, NG)
    end
end
=#

function add_fermion!(
    c::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    α::Number,
    a::T1,
    β::Number,
    B::T1,
) where {NC,T1<:WilsonFermion_4D_accelerator,TF,NG}#c += alpha*a 
    N = c.NX * c.NY * c.NZ * c.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_add_fermion!, c.f, α, a.f, β, B.f, NC, NG)
    # end

end



function add_fermion!(
    c::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    α::Number,
    a::T1,
) where {NC,T1<:WilsonFermion_4D_accelerator,TF,NG}#c += alpha*a 
    N = c.NX * c.NY * c.NZ * c.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_add_fermion!, c.f, α, a.f, NC, NG)
    #end

end


function shifted_fermion!(
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    boundarycondition,
    shift,
) where {NC,TF,NG}

    bc = Tuple(boundarycondition)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    N = x.NX * x.NY * x.NZ * x.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_shifted_fermion!, x.f, x.fshifted, bc, shift, NC, NX, NY, NZ, NT)
    #end


end

function shifted_fermion!(
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    boundarycondition,
    shift,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}

end




function mul_1plusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ5x!, y.f, x.f, NC)
    #end
end

function mul_1plusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ1x!, y.f, x.f, NC)
    #end
end

function mul_1plusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ2x!, y.f, x.f, NC)
    #end

end


function mul_1plusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ3x!, y.f, x.f, NC)
    #end
end

function mul_1plusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ4x!, y.f, x.f, NC)
    #end


end


function mul_1minusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ5x!, y.f, x.f, NC)
    #end


end


function mul_1minusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ1x!, y.f, x.f, NC)
    #end


end

function mul_1minusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ2x!, y.f, x.f, NC)
    #end

end


function mul_1minusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ3x!, y.f, x.f, NC)
    #end


end

function mul_1minusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ4x!, y.f, x.f, NC)
    #end
end


function mul_1plusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ5x!, y.f, x.parent.fshifted, NC)
    #end


end

function mul_1plusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ5x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    #end


end



function mul_1plusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ1x!, y.f, x.parent.fshifted, NC)
    #end

end

function mul_1plusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ1x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    #end

end

function mul_1plusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ2x!, y.f, x.parent.fshifted, NC)
    #end



end

function mul_1plusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ2x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    # end

end



function mul_1plusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ3x!, y.f, x.parent.fshifted, NC)
    #end


end

function mul_1plusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ3x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    #end

end


function mul_1plusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ4x!, y.f, x.parent.fshifted, NC)
    #end



end

function mul_1plusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1plusγ4x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    #end

end



function mul_1minusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ1x!, y.f, x.parent.fshifted, NC)
    #end


end

function mul_1minusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ1x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    #end

end


function mul_1minusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ2x!, y.f, x.parent.fshifted, NC)
    #end


end

function mul_1minusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ2x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    #end

end


function mul_1minusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ3x!, y.f, x.parent.fshifted, NC)
    #end


end

function mul_1minusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ3x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    #end

end

function mul_1minusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ4x!, y.f, x.parent.fshifted, NC)
    #end
end

function mul_1minusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_1minusγ4x_shifted!, y.f,
        x.parent.f, x.shift, NC, x.bc, NX, NY, NZ, NT)
    #end

end


function LinearAlgebra.dot(
    A::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    B::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}

    N = A.NX * A.NY * A.NZ * A.NT
    #CUDA.@sync begin
    s = JACC.parallel_reduce(N, +, jacckernel_dot!, A.f, B.f, NC; init=zero(eltype(A.f)))
    #end
    #s = CUDA.reduce(+, A.temp_volume)

    return s
end

function substitute_fermion!(
    A::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    B::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}

    N = A.NX * A.NY * A.NZ * A.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_substitute_fermion!, A.f, B.f, NC)
    #end


end


function substitute_fermion!(
    A::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    B::AbstractFermionfields_4D{NC},
) where {NC,TF,NG}
    acpu = Array(A.f)

    N = A.NX * A.NY * A.NZ * A.NT
    for i = 1:N
        ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
        for ig = 1:NG
            for ic = 1:NC
                acpu[ic, ig, i] = B[ic, ix, iy, iz, it, ig]
            end
        end
    end
    #blockinfo = A.blockinfo

    agpu = JACC.Array(acpu)
    A.f .= agpu
end

function substitute_fermion!(
    A::AbstractFermionfields_4D{NC},
    B::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}
    bcpu = Array(B.f)

    N = B.NX * B.NY * B.NZ * B.NT
    for i = 1:N
        ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
        for ig = 1:NG
            for ic = 1:NC
                A[ic, ix, iy, iz, it, ig] = bcpu[ic, ig, i] 
            end
        end
    end

end