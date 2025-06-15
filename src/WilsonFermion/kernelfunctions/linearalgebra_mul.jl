#y = A*x
#=
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF,NG}

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yUx_NC3!(b, r, y.f, A.U, x.f, NG)
        end
    end

end
=#

#y = A'*x
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    A::Adjoint_Gaugefields{T},
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF,NG}

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yUdagx_NC3!(b, r, y.f, A.parent.U, x.f, NG)
        end
    end
end

#=
function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    A::T,
    x::Shifted_fermionfields_4D_accelerator,
) where {T<:Gaugefields_4D_accelerator,TF,NG}

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yUx_NC3!(b, r, y.f, A.U, x.parent.fshifted, NG)
        end
    end

end
=#

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    α::Number,
    X::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    β::Number,
    Y::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
) where {NC,TF,NG}

    for r = 1:X.blockinfo.rsize
        for b = 1:X.blockinfo.blocksize
            kernel_axpby!(b, r, α, X.f, β, Y.f, NC)
        end
    end
    return Y
end

function LinearAlgebra.axpby!(
    α::Number,
    X::WilsonFermion_4D_accelerator{3,TF,4,:none},
    β::Number,
    Y::WilsonFermion_4D_accelerator{3,TF,4,:none},
) where {TF}

    for r = 1:X.blockinfo.rsize
        for b = 1:X.blockinfo.blocksize
            kernel_axpby_NC3NG4!(b, r, α, X.f, β, Y.f)
        end
    end
    return Y
end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    A::TA,
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
) where {TA<:AbstractMatrix,NC,TF,NG}

    for r = 1:xout.blockinfo.rsize
        for b = 1:xout.blockinfo.blocksize
            kernel_mul_Ax!(b, r, xout.f, A, x.f, NC)
        end
    end
end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{3,TF,4,:none},
    A::TA,
    x::WilsonFermion_4D_accelerator{3,TF,4,:none},
) where {TA<:AbstractMatrix,TF}

    for r = 1:xout.blockinfo.rsize
        for b = 1:xout.blockinfo.blocksize
            kernel_mul_Ax_NC3NG4!(b, r, xout.f, A, x.f)
        end
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    A::Adjoint_Wilson_operator_faster{Wilson_Dirac_operator_faster{Dim,T,fermion}},
    x::T3,
) where {NC,TF,NG,T,Dim,fermion,T3<:WilsonFermion_4D_accelerator}
    clear_fermion!(y)

    add_fermion!(y, A.parent.factor, x)

    for μ = 1:Dim
        temp1, it_temp1 = get_temp(A.parent._temporary_fermi)

        mul!(temp1, A.parent.D[μ]', x)
        #mul!(A.parent._temporary_fermi[1], A.parent.D[μ]', x)


        #add_fermion!(y, -A.parent.factor * A.parent.κ, A.parent._temporary_fermi[1])
        add_fermion!(y, -A.parent.factor * A.parent.κ, temp1)


        unused!(A.parent._temporary_fermi, it_temp1)
    end

end




function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    A::Adjoint_Wilson_Dirac_1storder_operator{
        Wilson_Dirac_1storder_operator{Dim,T,fermion},
    },
    x::T3,
) where {NC,TF,NG,T,Dim,fermion,T3<:WilsonFermion_4D_accelerator}
    if A.parent.μ == 1
        apply_Dirac_1storder_1_dagger!(
            y,
            x,
            A.parent.U,
            A.parent.boundarycondition,
            A.parent._temporary_fermi,
        )
        if any(isnan, y.f)
            error("NaN detected in array y!")
        end

    elseif A.parent.μ == 2
        apply_Dirac_1storder_2_dagger!(
            y,
            x,
            A.parent.U,
            A.parent.boundarycondition,
            A.parent._temporary_fermi,
        )
    elseif A.parent.μ == 3
        apply_Dirac_1storder_3_dagger!(
            y,
            x,
            A.parent.U,
            A.parent.boundarycondition,
            A.parent._temporary_fermi,
        )
    elseif A.parent.μ == 4
        apply_Dirac_1storder_4_dagger!(
            y,
            x,
            A.parent.U,
            A.parent.boundarycondition,
            A.parent._temporary_fermi,
        )
    else
        error("μ = $(A.parent.μ) is not supported!!")
    end
end


function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:WilsonFermion_4D_accelerator,TF,NG}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yAx_NC3!(b, r, y.f, A.U, x.f)
        end
    end

end


function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:Shifted_fermionfields_4D_accelerator,TF,NG}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yAx_NC3!(b, r, y.f, A.U, x.parent.fshifted)
        end
    end

end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none,TUv,TFshifted},
    A::T,
    x::T3,
) where {T<:Gaugefields_4D_accelerator,T3<:Shifted_fermionfields_4D_accelerator,TF,NG,TUv,TFshifted<:Nothing}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yAx_NC3_shifted!(b, r, y.f, A.U, x.parent.f, x.shift, x.parent.blockinfo, x.bc, NX, NY, NZ, NT)
        end
    end

end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    A::T,
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
) where {NC,T<:Number,TF,NG}
    @assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_ysx_NC!(b, r, y.f, A, x.f, NC)
        end
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,4,:none},
    A::T,
    x::WilsonFermion_4D_accelerator{4,TF,4,:none},
) where {T<:Number,TF}
    #@assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_ysx_NC3NG4!(b, r, y.f, A, x.f)
        end
    end
end



function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    y::WilsonFermion_4D_accelerator{NC,TF,NG,:none}, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,NC,TF,NG}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_uxy_NC!(b, r, u.U, x.f, y.f, NC, NG)
        end
    end
end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{3,TF,4,:none},
    y::WilsonFermion_4D_accelerator{3,TF,4,:none}, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,TF}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end

    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_uxy_NC3NG4!(b, r, u.U, x.f, y.f)
        end
    end
end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    y::Adjoint_fermionfields, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,NC,TF,NG}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_uxydag_NC!(b, r, u.U, x.f, y.parent.f, NC, NG)
        end
    end


end

function LinearAlgebra.mul!(
    u::T1,
    x::WilsonFermion_4D_accelerator{3,TF,4,:none},
    y::Adjoint_fermionfields, ; clear=true
) where {T1<:Gaugefields_4D_accelerator,TF}


    #clear_U!(u)
    if clear
        clear_U!(u)
    else
        #    println(sum(abs.(u.U)))
    end


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_uxydag_NC3NG4!(b, r, u.U, x.f, y.parent.f)
        end
    end


end



function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    x::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    A::T,
) where {T<:Gaugefields_4D_accelerator,TF,NG}
    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_yxA_NC3!(b, r, y.f, x.f, A.U, NG)
        end
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{T},
) where {T<:Gaugefields_4D_accelerator,TF,NG,Ts<:Shifted_fermionfields_4D_accelerator}
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yxdagAdag_NC3!(b, r, y.f, x.parent.parent.fshifted, A.parent.U, NG)
        end
    end
end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_accelerator{3,TF,NG,:none,TUv,TFshifted},
    x::Adjoint_fermionfields{Ts},
    A::Adjoint_Gaugefields{T},
) where {T<:Gaugefields_4D_accelerator,TF,NG,Ts<:Shifted_fermionfields_4D_accelerator,TUv,TFshifted<:Nothing}
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_yxdagAdagshifted_NC3!(b, r, y.f, x.parent.parent.f, A.parent.U, NG, x.parent.shift,
                x.parent.parent.blockinfo, x.parent.bc, NX, NY, NZ, NT)
        end
    end
end



function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    x::WilsonFermion_4D_accelerator{NC,TF,NG,:none},
    A::TA,
) where {TA<:AbstractMatrix,NC,TF,NG}
    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_xA_NC!(b, r, xout.f, x.f, A, NC)
        end
    end
end

function LinearAlgebra.mul!(
    xout::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    x::WilsonFermion_4D_accelerator{3,TF,NG,:none},
    A::TA,
) where {TA<:AbstractMatrix,TF,NG}
    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_xA_NC3!(b, r, xout.f, x.f, A)
        end
    end
end

