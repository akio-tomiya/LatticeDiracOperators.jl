#y = A'*x
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:jacc},
    A::Adjoint_Gaugefields{T},
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF,NG}

    #CUDA.@sync begin
    N = y.NX * y.NY * y.NZ * y.NT
    JACC.parallel_for(N, jacckernel_mul_yUdagx_NC3!, y.f, A.parent.U, x.f, NG)
    #end
end




#y = A'*x
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    A::Adjoint_Gaugefields{T},
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF}

    #CUDA.@sync begin
    N = y.NX * y.NY * y.NZ * y.NT
    JACC.parallel_for(N, jacckernel_mul_yUdagx_NC3NG4!, y.f, A.parent.U, x.f)
    #end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:jacc},
    A::T,
    x::Shifted_fermionfields_4D_accelerator,
) where {T<:Gaugefields_4D_accelerator,TF,NG}

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yUx_NC3!, y.f, A.U, x.parent.fshifted, NG)
    #end

end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    A::T,
    x::Shifted_fermionfields_4D_accelerator,
) where {T<:Gaugefields_4D_accelerator,TF}

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yUx_NC3NG4!, y.f, A.U, x.parent.fshifted)
    #end

end

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    α::Number,
    X::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    β::Number,
    Y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,TF,NG}
    N = Y.NX * Y.NY * Y.NZ * Y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_axpby!, α, X.f, β, Y.f, NC)
    #end
    return Y
end

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    α::Number,
    X::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    β::Number,
    Y::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
) where {TF}
    N = Y.NX * Y.NY * Y.NZ * Y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_axpby_NC3NG4!, α, X.f, β, Y.f)
    #end
    return Y
end


function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    A::TA,
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {TA<:AbstractMatrix,NC,TF,NG}
    #Afb = zero(A)
    #Afb .= A
    Af = JACC.array(A[:, :])
    #println(typeof(Af))


    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_Ax!, xout.f, Af, x.f, NC)
    # end

end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    A::TA,
    x::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
) where {TA<:AbstractMatrix,TF}
    #Afb = zero(A)
    #Afb .= A
    Af = JACC.array(A[:, :])
    #println(typeof(Af))
    #println(typeof(Afb))
    #error("d")
    #    println(typeof(Af))
    #error("d")

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_Ax_NC3NG4!, xout.f, Af, x.f)
    #end

end



function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:jacc},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF,NG}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yAx_NC3!, y.f, A.U, x.f)
    #end


end



#=
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:jacc},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:Shifted_fermionfields_4D_accelerator,TF,NG}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yAx_NC3!, y.f, A.U, x.parent.fshifted)
    #end

end
=#

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:jacc,TUv,TFshifted},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:Shifted_fermionfields_4D_accelerator,TF,NG,TUv,TFshifted<:Nothing}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yAx_NC3_shifted!, y.f, A.U,
        x.parent.f, x.shift, x.bc, NX, NY, NZ, NT)
    #end


end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    A::T,
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
) where {NC,T<:Number,TF,NG}
    @assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_ysx_NC!, y.f, A, x.f, NC)
    #end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    A::T,
    x::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
) where {T<:Number,TF}
    #@assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_ysx_NC3NG4!, y.f, A, x.f)
    #end
end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc}, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,NC,TF,NG}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_uxy_NC!, u.U, x.f, y.f, NC, NG)
    # end

end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    y::WilsonFermion_4D_accelerator{3,TF,4,:jacc}, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,TF}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_uxy_NC3NG4!, u.U, x.f, y.f)
    #end

end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    y::Adjoint_fermionfields, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,NC,TF,NG}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    N = y.NX * y.NY * y.NZ * y.NT

    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_uxydag_NC!, u.U, x.f, y.parent.f, NC, NG)
    #end


end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    y::Adjoint_fermionfields, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,TF}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_uxydag_NC3NG4!, u.U, x.f, y.parent.f)
    #end


end



function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{3,TF,NG,:jacc},
    A::T,
) where {T<:Gaugefields_4D_accelerator,TF,NG}
    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxA_NC3!, y.f, x.f, A.U, NG)
    #end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:jacc},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{T},
) where {T<:Gaugefields_4D_accelerator,TF,NG,Ts<:Shifted_fermionfields_4D_accelerator}

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxdagAdag_NC3!, y.f, x.parent.parent.fshifted, A.parent.U, NG)
    #end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:jacc,TUv,TFshifted},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{T},
) where {T<:Gaugefields_4D_accelerator,TF,NG,Ts<:Shifted_fermionfields_4D_accelerator,TUv,TFshifted<:Nothing}
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    N = y.NX * y.NY * y.NZ * y.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_yxdagAdagshifted_NC3!, y.f,
        x.parent.parent.f, A.parent.U, NG, x.parent.shift,
        x.parent.bc, NX, NY, NZ, NT)
    #end

end



function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:jacc},
    A::TA,
) where {TA<:AbstractMatrix,NC,TF,NG}
    Af = JACC.array(A[:, :])

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_xA_NC!, xout.f, x.f, Af, NC)
    #end
end


function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    x::WilsonFermion_4D_accelerator{3,TF,4,:jacc},
    A::TA,
) where {TA<:AbstractMatrix,TF}
    Af = JACC.array(A[:, :])

    N = x.NX * x.NY * x.NZ * x.NT
    #CUDA.@sync begin
    JACC.parallel_for(N, jacckernel_mul_xA_NC3NG4!, xout.f, x.f, Af)
    #end
end
