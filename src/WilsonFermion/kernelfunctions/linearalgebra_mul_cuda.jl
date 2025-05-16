#y = A'*x
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    A::Adjoint_Gaugefields{T},
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF<:CUDA.CuArray,NG}

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_yUdagx_NC3!(y.f, A.parent.U, x.f, NG)
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    A::T,
    x::Shifted_fermionfields_4D_accelerator,
) where {T<:Gaugefields_4D_accelerator,TF<:CUDA.CuArray,NG}

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_yUx_NC3!(y.f, A.U, x.parent.fshifted, NG)
    end

end

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    α::Number,
    X::WilsonFermion_4D_accelerator{NC,TF,NG},
    β::Number,
    Y::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}
    CUDA.@sync begin
        CUDA.@cuda threads = X.blockinfo.blocksize blocks = X.blockinfo.rsize cudakernel_axpby!(α, X.f, β, Y.f, NC)
    end
    return Y
end

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    α::Number,
    X::WilsonFermion_4D_accelerator{3,TF,4},
    β::Number,
    Y::WilsonFermion_4D_accelerator{3,TF,4},
) where {TF<:CUDA.CuArray}
    CUDA.@sync begin
        CUDA.@cuda threads = X.blockinfo.blocksize blocks = X.blockinfo.rsize cudakernel_axpby_NC3NG4!(α, X.f, β, Y.f)
    end
    return Y
end


function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{NC,TF,NG},
    A::TA,
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {TA<:AbstractMatrix,NC,TF<:CUDA.CuArray,NG}
    #Afb = zero(A)
    #Afb .= A
    Af = CUDA.CuArray(A)
    #println(typeof(Af))
    #println(typeof(Afb))
    #error("d")

    CUDA.@sync begin
        CUDA.@cuda threads = xout.blockinfo.blocksize blocks = xout.blockinfo.rsize cudakernel_mul_Ax!(xout.f, Af, x.f, NC)
    end

end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{3,TF,4},
    A::TA,
    x::WilsonFermion_4D_accelerator{3,TF,4},
) where {TA<:AbstractMatrix,TF<:CUDA.CuArray}
    #Afb = zero(A)
    #Afb .= A
    Af = CUDA.CuArray(A)
    #println(typeof(Af))
    #println(typeof(Afb))
    #error("d")

    CUDA.@sync begin
        CUDA.@cuda threads = xout.blockinfo.blocksize blocks = xout.blockinfo.rsize cudakernel_mul_Ax_NC3NG4!(xout.f, Af, x.f)
    end

end



function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF<:CUDA.CuArray,NG}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_yAx_NC3!(y.f, A.U, x.f)
    end


end




function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:Shifted_fermionfields_4D_accelerator,TF<:CUDA.CuArray,NG}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_yAx_NC3!(y.f, A.U, x.parent.fshifted)
    end

end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,TUv,TFshifted},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:Shifted_fermionfields_4D_accelerator,TF<:CUDA.CuArray,NG,TUv,TFshifted<:Nothing}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_yAx_NC3_shifted!(y.f, A.U,
            x.parent.f, x.shift, x.parent.blockinfo, x.bc, NX, NY, NZ, NT)
    end


end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    A::T,
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,T<:Number,TF<:CUDA.CuArray,NG}
    @assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_ysx_NC!(y.f, A, x.f, NC)
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,4},
    A::T,
    x::WilsonFermion_4D_accelerator{3,TF,4},
) where {T<:Number,TF<:CUDA.CuArray}
    #@assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_ysx_NC3NG4!(y.f, A, x.f)
    end
end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
    y::WilsonFermion_4D_accelerator{NC,TF,NG}, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,NC,TF<:CUDA.CuArray,NG}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_uxy_NC!(u.U, x.f, y.f, NC, NG)
    end

end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{3,TF,4},
    y::WilsonFermion_4D_accelerator{3,TF,4}, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,TF<:CUDA.CuArray}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_uxy_NC3NG4!(u.U, x.f, y.f)
    end

end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
    y::Adjoint_fermionfields, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,NC,TF<:CUDA.CuArray,NG}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end


    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_uxydag_NC!(u.U, x.f, y.parent.f, NC, NG)
    end


end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{3,TF,4},
    y::Adjoint_fermionfields, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,TF<:CUDA.CuArray}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end


    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_uxydag_NC3NG4!(u.U, x.f, y.parent.f)
    end


end



function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    x::WilsonFermion_4D_accelerator{3,TF,NG},
    A::T,
) where {T<:Gaugefields_4D_accelerator,TF<:CUDA.CuArray,NG}

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_yxA_NC3!(y.f, x.f, A.U, NG)
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{T},
) where {T<:Gaugefields_4D_accelerator,TF<:CUDA.CuArray,NG,Ts<:Shifted_fermionfields_4D_accelerator}

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_yxdagAdag_NC3!(y.f, x.parent.parent.fshifted, A.parent.U, NG)
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,TUv,TFshifted},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{T},
) where {T<:Gaugefields_4D_accelerator,TF<:CUDA.CuArray,NG,Ts<:Shifted_fermionfields_4D_accelerator,TUv,TFshifted<:Nothing}
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_yxdagAdagshifted_NC3!(y.f,
            x.parent.parent.f, A.parent.U, NG, x.parent.shift,
            x.parent.parent.blockinfo, x.parent.bc, NX, NY, NZ, NT)
    end

end



function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
    A::TA,
) where {TA<:AbstractMatrix,NC,TF<:CUDA.CuArray,NG}
    Af = CUDA.CuArray(A)

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_xA_NC!(xout.f, x.f, Af, NC)
    end
end


function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{3,TF,4},
    x::WilsonFermion_4D_accelerator{3,TF,4},
    A::TA,
) where {TA<:AbstractMatrix,TF<:CUDA.CuArray}
    Af = CUDA.CuArray(A)

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_xA_NC3NG4!(xout.f, x.f, Af)
    end
end
