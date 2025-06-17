using LinearAlgebra
#y = A'*x
function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::Adjoint_Gaugefields{T},
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:StaggeredFermion_4D_accelerator,TF}

    #CUDA.@sync begin
    N = y.NX * y.NY * y.NZ * y.NT
    JACC.parallel_for(N, jacckernel_mul_yUdagx_NC3_staggered!, y.f, A.parent.U, x.f)
    #end
end



function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::T,
    x::Shifted_fermionfields_4D_accelerator,
) where {T<:Gaugefields_4D_accelerator,TF}

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yUx_NC3_staggered!, y.f, A.U, x.parent.fshifted)
    #end

end


#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    α::Number,
    X::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    β::Number,
    Y::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
) where {NC,TF}
    N = Y.NX * Y.NY * Y.NZ * Y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_axpby_staggered!, α, X.f, β, Y.f, NC)
    #end
    return Y
end




function LinearAlgebra.mul!(
    xout::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    A::TA,
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
) where {TA<:AbstractMatrix,NC,TF}
    #Afb = zero(A)
    #Afb .= A
    Af = JACC.array(A[:, :])
    #println(typeof(Af))


    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_Ax_staggered!, xout.f, Af, x.f, NC)
    # end

end




function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:StaggeredFermion_4D_accelerator,TF}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yAx_NC3_staggered!, y.f, A.U, x.f)
    #end


end




function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:Shifted_fermionfields_4D_accelerator,TF}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yAx_NC3_staggered!, y.f, A.U, x.parent.fshifted)
    #end

end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc,TUv,TFshifted},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:Shifted_fermionfields_4D_accelerator,TF,TUv,TFshifted<:Nothing}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yAx_NC3_shifted_staggered!, y.f, A.U,
        x.parent.f, x.shift, x.bc, NX, NY, NZ, NT)
    #end


end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    A::T,
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
) where {NC,T<:Number,TF}
    @assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_ysx_NC_staggered!, y.f, A, x.f, NC)
    #end
end


function LinearAlgebra.mul!(
    u::T1,
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    y::StaggeredFermion_4D_accelerator{NC,TF,:jacc}, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,NC,TF}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_uxy_NC_staggered!, u.U, x.f, y.f, NC)
    # end

end


function LinearAlgebra.mul!(
    u::T1,
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    y::Adjoint_fermionfields, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,NC,TF}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    N = x.NX * x.NY * x.NZ * x.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_uxydag_NC_staggered!, u.U, x.f, y.parent.f, NC)
    #end


end




function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    x::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::T,
) where {T<:Gaugefields_4D_accelerator,TF}
    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxA_NC3_staggered!, y.f, x.f, A.U)
    #end
end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{T},
) where {T<:Gaugefields_4D_accelerator,TF,Ts<:Shifted_fermionfields_4D_accelerator}

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxdagAdag_NC3_staggered!, y.f, x.parent.parent.fshifted, A.parent.U)
    #end
end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc,TUv,TFshifted},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{T},
) where {T<:Gaugefields_4D_accelerator,TF,Ts<:Shifted_fermionfields_4D_accelerator,TUv,TFshifted<:Nothing}
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxdagAdagshifted_NC3_staggered!, y.f,
        x.parent.parent.f, A.parent.U, NG, x.parent.shift,
        x.parent.bc, NX, NY, NZ, NT)
    #end

end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc,TUv,TFshifted},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{Staggered_Gaugefields{T,μ}},
) where {T<:Gaugefields_4D_accelerator,μ,TF,Ts<:Shifted_fermionfields_4D_accelerator,TUv,TFshifted<:Nothing}
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxdagAdagshifted_NC3_staggered!, y.f,
        x.parent.parent.f, A.parent.U, NG, x.parent.shift,
        x.parent.bc, NX, NY, NZ, NT)
    #end

end

#const "d"

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc,TUv,TFshifted},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{Staggered_Gaugefields{Gaugefields_4D_accelerator{T},μ}}
) where {TF,TUv,TFshifted,Ts<:Shifted_staggeredfermionfields_4D_accelerator,μ,T<:Gaugefields_4D_accelerator}
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxdagAdagshifted_NC3_staggered!, y.f,
        x.parent.parent.f, A.parent.U, NG, x.parent.shift,
        x.parent.bc, NX, NY, NZ, NT)
end


function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{Staggered_Gaugefields{K,μ}}
) where {Ts,K,TF,μ}
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    #error("d")

    N = y.NX * y.NY * y.NZ * y.NT
    yin = y.f
    xin = x.parent.parent.f
    Ain = A.parent.parent.U
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxdagstaggeredAdagshifted_NC3_staggered!,
        yin, xin, Ain, x.parent.shift,
        x.parent.bc, NX, NY, NZ, NT, μ)
end




function LinearAlgebra.mul!(
    xout::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    x::StaggeredFermion_4D_accelerator{NC,TF,:jacc},
    A::TA,
) where {TA<:AbstractMatrix,NC,TF}
    Af = JACC.array(A[:, :])

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_xA_NC_staggered!, xout.f, x.f, Af, NC)
    #end
end


function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:StaggeredFermion_4D_accelerator,TF}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    #NX = y.NX
    #NY = y.NY
    #NZ = y.NZ
    #NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    JACC.parallel_for(N, jacckernel_mul_yAx_NC3_staggered!, y.f, A.U, x.f)
end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::Staggered_Gaugefields{T,μ},
    x::T3,
) where {T,μ,T3<:Shifted_staggeredfermionfields_4D_accelerator,TF}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    #NX = y.NX
    #NY = y.NY
    #NZ = y.NZ
    #NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    JACC.parallel_for(N,
        jacckernel_mul_ystaggeredAx_NC3_staggered!,
        y.f, A.parent.U, x.parent.fshifted,
        y.NX, y.NY, y.NZ, y.NT, μ)
end


function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::Adjoint_Gaugefields{T},
    x::T3,
) where {T,T3<:Shifted_staggeredfermionfields_4D_accelerator,TF}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    #NX = y.NX
    #NY = y.NY
    #NZ = y.NZ
    #NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    JACC.parallel_for(N,
        jacckernel_mul_yAdagx_NC3_staggered!,
        y.f, A.parent.U, x.parent.fshifted,
        y.NX, y.NY, y.NZ, y.NT, μ)
end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_accelerator{3,TF,:jacc},
    A::Adjoint_Gaugefields{Staggered_Gaugefields{T,μ}},
    x::T3,
) where {T<:Shifted_Gaugefields_4D_accelerator,μ,T3<:Shifted_staggeredfermionfields_4D_accelerator,TF}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    #NX = y.NX
    #NY = y.NY
    #NZ = y.NZ
    #NT = y.NT
    yin = y.f
    Ain = A.parent.parent.parent.Ushifted
    xin = x.parent.fshifted

    N = y.NX * y.NY * y.NZ * y.NT
    JACC.parallel_for(N,
        jacckernel_mul_ystaggeredAdagx_NC3_staggered!,
        yin, Ain, xin,
        y.NX, y.NY, y.NZ, y.NT, μ)
end


