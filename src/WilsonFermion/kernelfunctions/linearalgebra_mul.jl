#y = A*x
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF,NG}

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yUx_NC3!(b, r, y.f, A.U, x.f, NG)
        end
    end

end

#y = A'*x
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    A::Adjoint_Gaugefields{T},
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF,NG}

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yUdagx_NC3!(b, r, y.f, A.parent.U, x.f, NG)
        end
    end

end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG},
    A::T,
    x::Shifted_fermionfields_4D{3,T3},
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF,NG}

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yUx_NC3!(b, r, y.f, A.U, x.parent.fshifted, NG)
        end
    end

end