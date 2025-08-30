
abstract type WilsonFermion_4D{NC} <: AbstractFermionfields_4D{NC} end

function Wx!(
    xout::T,
    U::Array{G,1},
    x::T,
    A,
) where {T<:WilsonFermion_4D,G<:AbstractGaugefields}
    Wx!(xout, U, x, A, 4)
    return
end

function Dx!(
    xout::T1,
    U::Array{G,1},
    x::T2,
    A,
) where {T1<:WilsonFermion_4D,T2<:WilsonFermion_4D,G<:AbstractGaugefields}
    Dx!(xout, U, x, A, 4)
    return
end

function Ddagx!(
    xout::T1,
    U::Array{G,1},
    x::T2,
    A,
) where {T1<:WilsonFermion_4D,T2<:WilsonFermion_4D,G<:AbstractGaugefields}
    Ddagx!(xout, U, x, A, 4)
    return
end


function Tx!(
    xout::T,
    U::Array{G,1},
    x::T,
    A,
) where {T<:WilsonFermion_4D,G<:AbstractGaugefields} # Tx, W = (1 - T)x
    Tx!(xout, U, x, A, 4)
    return
end



function Wdagx!(
    xout::T,
    U::Array{G,1},
    x::T,
    A,
) where {T<:WilsonFermion_4D,G<:AbstractGaugefields}
    #,temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_4D,G <: AbstractGaugefields}
    Wdagx!(xout, U, x, A, 4)
    return
end



function calc_beff!(xout, U, x, A) #be + K Teo bo
    isodd = false
    temp = A._temporary_fermi[4]#temps[4]
    clear_fermion!(temp, isodd)
    Toex!(temp, U, x, A, isodd)

    iseven = true
    add_fermion!(xout, 1, x, 1, temp, iseven)

end

function calc_beff_dag!(xout, U, x, A) #be + K Teo bo
    isodd = false
    temp = A._temporary_fermi[4]#temps[4]
    clear_fermion!(temp)
    Tdagoex!(temp, U, x, A, isodd)

    iseven = true
    add_fermion!(xout, 1, x, 1, temp, iseven)

end

function mul_1plusγμx!(y::WilsonFermion_4D, x, μ)
    #println("mul")
    #display(x[:, 1, 1, 1, 1, :])
    if μ == 1
        mul_1plusγ1x!(y, x)
    elseif μ == 2
        mul_1plusγ2x!(y, x)
    elseif μ == 3
        mul_1plusγ3x!(y, x)
    elseif μ == 4
        mul_1plusγ4x!(y, x)
    end
end

function mul_1minusγμx!(y::WilsonFermion_4D, x, μ)
    if μ == 1
        mul_1minusγ1x!(y, x)
    elseif μ == 2
        mul_1minusγ2x!(y, x)
    elseif μ == 3
        mul_1minusγ3x!(y, x)
    elseif μ == 4
        mul_1minusγ4x!(y, x)
    end
end

"""
           (       -i )              (       -1 )
 GAMMA1 =  (     -i   )     GAMMA2 = (     +1   )
           (   +i     )              (   +1     )
           ( +i       )              ( -1       )
               (     -i   )              (     -1   )
     GAMMA3 =  (       +i )     GAMMA4 = (       -1 )
               ( +i       )              ( -1       )
               (   -i     )              (   -1     )

"""

function mul_x1plusγμ!(y::WilsonFermion_4D, x, μ)
    if μ == 1
        mul_1minusγ1x!(y, x)
    elseif μ == 2
        mul_1plusγ2x!(y, x)
    elseif μ == 3
        mul_1minusγ3x!(y, x)
    elseif μ == 4
        mul_1plusγ4x!(y, x)
    end
end

function mul_x1minusγμ!(y::WilsonFermion_4D, x, μ)
    if μ == 1
        mul_1plusγ1x!(y, x)
    elseif μ == 2
        mul_1minusγ2x!(y, x)
    elseif μ == 3
        mul_1plusγ3x!(y, x)
    elseif μ == 4
        mul_1minusγ4x!(y, x)
    end
end



function Toex!(
    xout::T,
    U::Array{G,1},
    x::T,
    A,
    iseven;
    boundarycondition=boundarycondition_default
) where {T<:WilsonFermion_4D,G<:AbstractGaugefields} #T_oe xe
    #temp = A._temporary_fermi[4]#temps[4]
    temp1 = A._temporary_fermi[1] #temps[1]
    temp2 = A._temporary_fermi[2] #temps[2]

    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    if iseven
        isodd = false
    else
        isodd = true
    end

    #clear_fermion!(temp,isodd)
    clear_fermion!(xout, isodd)
    #set_wing_fermion!(x)
    for ν = 1:4

        xplus = shift_fermion(x, ν; boundarycondition)
        #println(xplus)


        mul!(temp1, U[ν], xplus, isodd)




        #fermion_shift!(temp1,U,ν,x)

        #... Dirac multiplication

        mul!(temp1, view(A.rminusγ, :, :, ν), isodd)



        xminus = shift_fermion(x, -ν; boundarycondition)
        Uminus = shift_U(U[ν], -ν)


        mul!(temp2, Uminus', xminus, isodd)

        #
        #fermion_shift!(temp2,U,-ν,x)
        #mul!(temp2,view(x.rplusγ,:,:,ν),temp2)
        mul!(temp2, view(A.rplusγ, :, :, ν), isodd)

        add_fermion!(xout, A.hopp[ν], temp1, A.hopm[ν], temp2, isodd)

    end

    #clear_fermion!(xout,isodd)
    #add_fermion!(xout,1,x,-1,temp)

    set_wing_fermion!(xout, A.boundarycondition, isodd)

end

function Tdagoex!(
    xout::T,
    U::Array{G,1},
    x::T,
    A,
    iseven;
    boundarycondition=boundarycondition_default
) where {T<:WilsonFermion_4D,G<:AbstractGaugefields} #T_oe xe
    #temp = A._temporary_fermi[4]#temps[4]
    temp1 = A._temporary_fermi[1] #temps[1]
    temp2 = A._temporary_fermi[2] #temps[2]

    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    if iseven
        isodd = false
    else
        isodd = true
    end

    #clear_fermion!(temp,isodd)
    clear_fermion!(xout, isodd)
    #set_wing_fermion!(x)
    for ν = 1:4

        xplus = shift_fermion(x, ν; boundarycondition)
        #println(xplus)


        mul!(temp1, U[ν], xplus, isodd)




        #fermion_shift!(temp1,U,ν,x)

        #... Dirac multiplication

        mul!(temp1, view(A.rplusγ, :, :, ν), isodd)



        xminus = shift_fermion(x, -ν; boundarycondition)
        Uminus = shift_U(U[ν], -ν)


        mul!(temp2, Uminus', xminus, isodd)

        #
        #fermion_shift!(temp2,U,-ν,x)
        #mul!(temp2,view(x.rplusγ,:,:,ν),temp2)
        mul!(temp2, view(A.rminusγ, :, :, ν), isodd)

        add_fermion!(xout, A.hopp[ν], temp1, A.hopm[ν], temp2, isodd)

    end

    #clear_fermion!(xout,isodd)
    #add_fermion!(xout,1,x,-1,temp)

    set_wing_fermion!(xout, A.boundarycondition)

end

function WWx!(
    xout::T,
    U::Array{G,1},
    x::T,
    A;
    boundarycondition=boundarycondition_default
) where {T<:WilsonFermion_4D,G<:AbstractGaugefields} #(1 - K^2 Teo Toe) xe
    iseven = true
    isodd = false
    temp = A._temporary_fermi[7]#temps[4]
    temp2 = A._temporary_fermi[6]#temps[4]

    #println("Wx")
    #@time Wx!(xout,U,x,A) 
    clear_fermion!(xout, iseven)


    #Tx!(temp,U,x,A) 
    Toex!(temp, U, x, A, iseven; boundarycondition) #Toe
    #Tx!(temp2,U,temp,A) 
    Toex!(temp2, U, temp, A, isodd; boundarycondition) #Teo

    #set_wing_fermion!(temp,A.boundarycondition)
    #add_fermion!(xout,1,x,-1,temp2)
    add_fermion!(xout, 1, x, -1, temp2, iseven)
    set_wing_fermion!(xout, A.boundarycondition, iseven)



    #Wx!(xout,U,x,A) 
    return

    #Tx!(temp2,U,x,A) #Toe

    #Toex!(temp2,U,x,A,iseven) #Toe

    #Toex!(temp,U,temp2,A,isodd) #Teo
    #Tx!(temp,U,temp2,A) #Toe

    Tx!(temp, U, x, A) #Toe

    add_fermion!(xout, 1, x, -1, temp)
    #add_fermion!(xout,1,x,-1,temp,iseven)

    iseven = true
    set_wing_fermion!(xout, A.boundarycondition)

    return
end

function WWdagx!(
    xout::T,
    U::Array{G,1},
    x::T,
    A;
    boundarycondition=boundarycondition_default
) where {T<:WilsonFermion_4D,G<:AbstractGaugefields} #(1 - K^2 Teo Toe) xe
    iseven = true
    isodd = false
    temp = A._temporary_fermi[7]#temps[4]
    temp2 = A._temporary_fermi[6]#temps[4]

    clear_fermion!(xout)

    #Tx!(temp,U,x,A) 
    Tdagoex!(temp, U, x, A, iseven; boundarycondition) #Toe
    #Tx!(temp2,U,temp,A) 
    Tdagoex!(temp2, U, temp, A, isodd; boundarycondition) #Teo

    #set_wing_fermion!(temp,A.boundarycondition)
    #add_fermion!(xout,1,x,-1,temp2)
    add_fermion!(xout, 1, x, -1, temp2, iseven)
    set_wing_fermion!(xout, A.boundarycondition)


    return
end


include("./WilsonFermion_4D_wing_mpi.jl")
include("./WilsonFermion_4D_nowing_mpi.jl")
