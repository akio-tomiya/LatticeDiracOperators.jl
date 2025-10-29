
abstract type Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion} <:
              AbstractFermionfields_5D{NC} end

"""
Struct for MobiusDomainwallFermion
"""
struct MobiusDomainwallFermion_5D{NC,WilsonFermion} <:
       Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion}
    w::Array{WilsonFermion,1}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    L5::Int64
    Dirac_operator::String
    NWilson::Int64
    nowing::Bool


    function MobiusDomainwallFermion_5D(
        L5,
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T;
        nowing=false,
    ) where {T<:Integer}
        if nowing
            x = WilsonFermion_4D_nowing(NC, NX, NY, NZ, NT)
        else
            x = WilsonFermion_4D_wing(NC, NX, NY, NZ, NT)
        end
        xtype = typeof(x)
        w = Array{xtype,1}(undef, L5)
        w[1] = x
        for i = 2:L5
            w[i] = similar(x)
        end
        #println(w[2][1,1,1,1,1,1])
        NWilson = length(x)
        Dirac_operator = "MobiusDomainwall"
        return new{NC,xtype}(w, NC, NX, NY, NZ, NT, L5, Dirac_operator, NWilson, nowing)
    end

    function MobiusDomainwallFermion_5D(
        u::AbstractGaugefields{NC,4},
        L5::T,
        ; nowing=false, kwargs...
    ) where {T<:Integer,NC}

        x = Initialize_WilsonFermion(u; nowing, kwargs...)
        NX = u.NX
        NY = u.NY
        NZ = u.NZ
        NT = u.NT
        #if nowing
        #    x = WilsonFermion_4D_nowing(NC, NX, NY, NZ, NT)
        #else
        #    x = WilsonFermion_4D_wing(NC, NX, NY, NZ, NT)
        #end
        xtype = typeof(x)
        w = Array{xtype,1}(undef, L5)
        w[1] = x
        for i = 2:L5
            w[i] = similar(x)
        end
        #println(w[2][1,1,1,1,1,1])
        NWilson = length(x)
        Dirac_operator = "MobiusDomainwall"
        return new{NC,xtype}(w, NC, NX, NY, NZ, NT, L5, Dirac_operator, NWilson, nowing)
    end

    function MobiusDomainwallFermion_5D(
        w::Array{WilsonFermion,1},
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
        L5::T,
        Dirac_operator::String,
        NWilson::Int64,
        nowing::Bool
    ) where {T<:Integer,WilsonFermion}
        x = similar(w[1])
        w1 = Array{WilsonFermion,1}(undef, L5)
        w1[1] = x
        for i = 2:L5
            w1[i] = similar(x)
        end

        return new{NC,WilsonFermion}(w1, NC, NX, NY, NZ, NT, L5, Dirac_operator, NWilson, nowing)
    end
end

#=
function Base.similar(
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
) where {NC,WilsonFermion}
    return MobiusDomainwallFermion_5D(x.L5, NC, x.NX, x.NY, x.NZ, x.NT, nowing=x.nowing)
end
=#

function Base.similar(
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
) where {NC,WilsonFermion}
    return MobiusDomainwallFermion_5D(x.w, NC, x.NX, x.NY, x.NZ, x.NT, x.L5, x.Dirac_operator, x.NWilson, x.nowing)
end


function apply_R!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
) where {NC,WilsonFermion}
    clear_fermion!(xout)

    L5 = xout.L5
    for i5 = 1:L5
        j5 = L5 - i5 + 1
        substitute_fermion!(xout.w[i5], x.w[j5])
    end
end


function apply_P!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
) where {NC,WilsonFermion}
    L5 = xout.L5
    clear_fermion!(xout)

    for i5 = 1:L5
        j5 = i5
        #P_- -> P_+ in this definition
        mul_1plusγ5x_add!(xout.w[i5], x.w[j5], 1)
        set_wing_fermion!(xout.w[i5])

        #P_+ -> P_- in this definition
        j5 = i5 + 1
        j5 += ifelse(j5 > L5, -L5, 0)
        mul_1minusγ5x_add!(xout.w[i5], x.w[j5], 1)
        set_wing_fermion!(xout.w[i5])
    end
end

function apply_Pdag!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
) where {NC,WilsonFermion}
    L5 = xout.L5
    clear_fermion!(xout)

    for i5 = 1:L5
        j5 = i5
        #P_- -> P_+ in this definition
        mul_1plusγ5x_add!(xout.w[i5], x.w[j5], 1)
        set_wing_fermion!(xout.w[i5])

        #P_+ -> P_- in this definition
        j5 = i5 - 1
        j5 += ifelse(j5 < 1, L5, 0)
        mul_1minusγ5x_add!(xout.w[i5], x.w[j5], 1)
        set_wing_fermion!(xout.w[i5])
    end
end

function apply_11tensor!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},) where {NC,WilsonFermion}

    clear_fermion!(xout)

    i5 = 1
    substitute_fermion!(xout.w[i5], x.w[i5])

end

function apply_P_edge!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},) where {NC,WilsonFermion}

    clear_fermion!(xout)
    ratio = 1.0

    i5 = 1
    # LTK Definition P_- -> P_+
    mul_1plusγ5x_add!(xout.w[i5], x.w[i5], ratio)

    i5 = xout.L5
    # LTK Definition P_+ -> P_-
    mul_1minusγ5x_add!(xout.w[i5], x.w[i5], ratio)
end

function Chiral_Condensate_Operator!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    L5,
    A,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    temp1, temp2) where {NC,WilsonFermion}

    clear_fermion!(xout)

    apply_P!(temp1, x)
    apply_R!(temp2, temp1)
    solve_DinvX!(temp1, A, temp2)
    apply_Pdag!(xout, temp1)

end

function calc_Δ5x!(Δ5x, D, U,
    X1::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    X2::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion}) where {NC,WilsonFermion}

    D5_PV = D.D5DW_PV(U)
    Q_PV = MobiusD5DWdagD5DW_Wilson_operator(D5_PV)
    D5_M0 = D.D5DW_M0(U)

    temp1 = similar(X1)
    temp2 = similar(X2)

    Y1 = similar(X1)
    Y2 = similar(X2)
    Z1 = similar(X1)
    Z2 = similar(X2)

    apply_P!(Y1, X1.L5, X1, temp1)
    apply_P!(Y2, X2.L5, X2, temp1)
    mul!(Z1, D5_M0, Y1)

    solve_DinvX!(temp1, Q_PV, Z1)
    third_term = dot(Z2, temp1)

    mul!(temp1, D5_M0, Y1)
    solve_DinvX!(temp2, D5_PV, temp1)
    first_term = dot(Y2, temp2)

    mul!(temp1, D5_M0, Y2)
    solve_DinvX!(temp2, D5_PV, temp1)
    second_term = dot(Y1, temp2)

    Δ5x = 0.5 * (first_term + conj(second_term) + third_term)
end

function expand_4D_FermionField_to_5D!(
    x5::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x4::WilsonFermion) where {NC,WilsonFermion}

    clear_fermion!(x5)
    # L5 = x5.L5

    i5 = 1
    substitute_fermion!(x5.w[i5], x4)

end

# function expand_11_FermionField_to_5D!(
#     xout::Abstract_MobiusDomainwallFermion_5D{NC, WilsonFermion},
#     x::Abstract_MobiusDomainwallFermion_5D{NC, WilsonFermion}) where {NC, WilsonFermion}
#     clear_fermion!(xout)
#     # L5 = xout.L5

#     i5 = 1
#     # xout[i5] = x.w[i5]
#     expand_4D_FermionField_to_5D!(xout, x[i5])

# end

# function apply_P!(
#     xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
#     x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
# ) where {NC,WilsonFermion}
#     L5 = xout.L5
#     clear_fermion!(xout)

#     for i5 = 1:L5
#         j5 = i5
#         #P_- -> P_+ in this definition
#         mul_1plusγ5x_add!(xout.w[i5], x.w[j5], 1)
#         set_wing_fermion!(xout.w[i5])

#         #P_+ -> P_- in this definition
#         j5 = i5 + 1
#         j5 += ifelse(j5 > L5, -L5, 0)
#         mul_1minusγ5x_add!(xout.w[i5], x.w[j5], 1)
#         set_wing_fermion!(xout.w[i5])
#     end
# end

# function apply_Pdag!(
#     xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
#     x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
# ) where {NC,WilsonFermion}
#     L5 = xout.L5
#     clear_fermion!(xout)

#     for i5 = 1:L5
#         j5 = i5
#         #P_- -> P_+ in this definition
#         mul_1plusγ5x_add!(xout.w[i5], x.w[j5], 1)
#         set_wing_fermion!(xout.w[i5])

#         #P_+ -> P_- in this definition
#         j5 = i5 - 1
#         j5 += ifelse(j5 < 1, L5, 0)
#         mul_1minusγ5x_add!(xout.w[i5], x.w[j5], 1)
#         set_wing_fermion!(xout.w[i5])
#     end
# end

function apply_1pD!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    L5,
    U::Array{G,1},
    A,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    factor,
) where {NC,WilsonFermion,G<:AbstractGaugefields}
    #println("clear fermion")
    clear_fermion!(xout)

    # if factor == 0
    #     for i5 = 1:xout.L5
    #         massfactor = -(1 / (2 * A.κ) + 1)
    #         axpy!(massfactor, x.w[i5], xout.w[i5])
    #     end
    #     return
    # end

    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5
        for i5 = 1:xout.L5
            if i5 <= div(L5, 2) || i5 >= xout.L5 - div(L5, 2) + 1
                push!(irange, i5)
            else
                push!(irange_out, i5)
            end

        end
        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5
    end
    ratio = 1.0

    #Dwilson = A(U)
    for i5 in irange
        j5 = i5
        #println("D4x!")
        #@code_warntype D4x!(xout.w[i5], U, x.w[j5], A, 4)
        D4x!(xout.w[i5], U, x.w[j5], A, 4) #Dw*x
        #@time mul!(xout.w[i5], Dwilson, x.w[j5])
        #mul!(xout.w[i5], Dwilson, x.w[j5])
        #Dx!(xout.w[i5],U,x.w[j5],A) #Dw*x
        #Wx!(xout.w[i5],U,x.w[j5],temps) #Dw*x
        #1/(2*A.κ)
        massfactor = -(factor / (2 * A.κ) + 1)
        #println(massfactor)

        # if factor == 0 
        #     massfactor = 1.0
        # end

        #println("set_wing_fermion!")
        #@time set_wing_fermion!(xout.w[i5])
        #add!(ratio,xout.w[i5],ratio,x.w[j5]) #D = x + Ddagw*x
        #println("add!")
        #println(factor * ratio)
        #add!(factor * ratio, xout.w[i5], 0, x.w[j5]) #D = x + Dw*x
        add!(factor * ratio, xout.w[i5], ratio * massfactor, x.w[j5]) #D = x + Dw*x
        # COMMENT: Factorの位置修正
        # COMMENT : xout = x + factor * Dw * x
        #println("set_wing_fermion!")
        set_wing_fermion!(xout.w[i5])
    end

    # if L5 != xout.L5
    #     for i5 in irange_out
    #         axpy!(1, x.w[i5], xout.w[i5])
    #     end
    # end

end

function apply_1mD!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    L5,
    U::Array{G,1},
    A,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    factor,
) where {NC,WilsonFermion,G<:AbstractGaugefields}
    clear_fermion!(xout)

    # if factor == 0
    #     for i5 = 1:xout.L5
    #         massfactor = -(1 / (2 * A.κ) + 1)
    #         axpy!(massfactor, x.w[i5], xout.w[i5])
    #     end
    #     return
    # end

    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5

        for i5 = 1:xout.L5
            if i5 <= div(L5, 2) || i5 >= xout.L5 - div(L5, 2) + 1
                push!(irange, i5)
            else
                push!(irange_out, i5)
            end

        end


        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5
    end
    ratio = 1

    for i5 in irange
        j5 = i5
        D4x!(xout.w[i5], U, x.w[j5], A, 4) #Dw*x
        #Dx!(xout.w[i5],U,x.w[j5],A) #Dw*x
        #Wx!(xout.w[i5],U,x.w[j5],temps) #Dw*x
        #1/(2*A.κ)
        massfactor = -(factor / (2 * A.κ) + 1)

        # if factor == 0 
        #     massfactor = 1.0
        # end

        #set_wing_fermion!(xout.w[i5])
        #add!(ratio,xout.w[i5],ratio,x.w[j5]) #D = x + Ddagw*x
        add!(factor * ratio, xout.w[i5], ratio * massfactor, x.w[j5]) #D = x + Dw*x
        # COMMENT: Factorの位置修正
        # COMMENT : xout = x + factor * Dw * x
        set_wing_fermion!(xout.w[i5])
    end

    # if L5 != xout.L5
    #     for i5 in irange_out
    #         axpy!(1, x.w[i5], xout.w[i5])
    #     end
    # end

end

function apply_1pDdag!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    L5,
    U::Array{G,1},
    A,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    factor,
) where {NC,WilsonFermion,G<:AbstractGaugefields}
    clear_fermion!(xout)

    # if factor == 0
    #     for i5 = 1:xout.L5
    #         massfactor = -(1 / (2 * A.κ) + 1)
    #         axpy!(massfactor, x.w[i5], xout.w[i5])
    #     end
    #     return
    # end

    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5

        for i5 = 1:xout.L5
            if i5 <= div(L5, 2) || i5 >= xout.L5 - div(L5, 2) + 1
                push!(irange, i5)
            else
                push!(irange_out, i5)
            end

        end


        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5
    end
    ratio = 1

    for i5 in irange
        j5 = i5
        D4dagx!(xout.w[i5], U, x.w[j5], A, 4) #Dw*x

        #Dx!(xout.w[i5],U,x.w[j5],A) #Dw*x
        #Wx!(xout.w[i5],U,x.w[j5],temps) #Dw*x
        #1/(2*A.κ)
        massfactor = -(factor / (2 * A.κ) + 1)

        # if factor == 0 
        #     massfactor = 1.0
        # end

        set_wing_fermion!(xout.w[i5])
        #add!(ratio,xout.w[i5],ratio,x.w[j5]) #D = x + Ddagw*x
        add!(factor * ratio, xout.w[i5], ratio * massfactor, x.w[j5]) #D = x + Dw*x
        # COMMENT : Factorの位置修正
        # COMMENT : xout = x + factor * Dw * x
        set_wing_fermion!(xout.w[i5])
    end

    # if L5 != xout.L5
    #     for i5 in irange_out
    #         axpy!(1, x.w[i5], xout.w[i5])
    #     end
    # end

end

function apply_1mDdag!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    L5,
    U::Array{G,1},
    A,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    factor,
) where {NC,WilsonFermion,G<:AbstractGaugefields}
    clear_fermion!(xout)

    # if factor == 0
    #     for i5 = 1:xout.L5
    #         massfactor = -(1 / (2 * A.κ) + 1)
    #         axpy!(massfactor, x.w[i5], xout.w[i5])
    #     end
    #     return
    # end

    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5

        for i5 = 1:xout.L5
            if i5 <= div(L5, 2) || i5 >= xout.L5 - div(L5, 2) + 1
                push!(irange, i5)
            else
                push!(irange_out, i5)
            end

        end


        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5
    end
    ratio = 1

    for i5 in irange
        j5 = i5
        D4dagx!(xout.w[i5], U, x.w[j5], A, 4) #Dw*x
        #Dx!(xout.w[i5],U,x.w[j5],A) #Dw*x
        #Wx!(xout.w[i5],U,x.w[j5],temps) #Dw*x
        #1/(2*A.κ)
        massfactor = -(factor / (2 * A.κ) + 1)

        # if factor == 0 
        #     massfactor = 1.0
        # end

        set_wing_fermion!(xout.w[i5])
        #add!(ratio,xout.w[i5],ratio,x.w[j5]) #D = x + Ddagw*x
        add!(factor * ratio, xout.w[i5], ratio * massfactor, x.w[j5]) #D = x + Dw*x
        # COMMENT : Factorの位置修正
        # COMMENT : xout = x + factor * Dw * x
        set_wing_fermion!(xout.w[i5])
    end

    # if L5 != xout.L5
    #     for i5 in irange_out
    #         axpy!(1, x.w[i5], xout.w[i5])
    #     end
    # end

end

function apply_F!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    L5,
    m,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    temp1,
) where {NC,WilsonFermion}
    clear_fermion!(xout)

    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5

        for i5 = 1:xout.L5
            if i5 <= div(L5, 2) || i5 >= xout.L5 - div(L5, 2) + 1
                push!(irange, i5)
            else
                push!(irange_out, i5)
            end

        end


        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5
    end
    ratio = 1
    ratio2 = 1

    for i5 in irange
        j5 = i5 + 1

        if 1 <= j5 <= xout.L5
            #-P_- -> - P_+ :gamma_5 of LTK definition
            if xout.L5 != 2
                # ((1 + (b-c)/2 DW )*P_-*x)
                mul_1plusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], ratio * ratio2, temp1.w[i5])
                #set_wing_fermion!(xout.w[i5])
            end
        end


        j5 = i5 - 1
        if 1 <= j5 <= xout.L5
            #-P_+ -> - P_- :gamma_5 of LTK definition
            if xout.L5 != 2
                mul_1minusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], ratio * ratio2, temp1.w[i5])
                #set_wing_fermion!(xout.w[i5])
            end
        end




        if xout.L5 != 1
            if i5 == 1
                j5 = xout.L5
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1minusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], -m * ratio * ratio2, temp1.w[i5])
                #set_wing_fermion!(xout.w[i5])
            end

            if i5 == xout.L5
                j5 = 1
                mul_1plusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], -m * ratio * ratio2, temp1.w[i5])
                #println(-m * ratio * ratio2, "\t", dot(temp1.w[i5], temp1.w[i5]))

            end
        end
        set_wing_fermion!(xout.w[i5])
        #println(dot(xout.w[i5], xout.w[i5]))
    end

    #if L5 != xout.L5
    #    for i5 in irange_out
    #        axpy!(1,x.w[i5],xout.w[i5])
    #    end
    #end

end

function apply_δF!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    L5,
    m,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    temp1,
) where {NC,WilsonFermion}
    clear_fermion!(xout)

    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5

        for i5 = 1:xout.L5
            if i5 <= div(L5, 2) || i5 >= xout.L5 - div(L5, 2) + 1
                push!(irange, i5)
            else
                push!(irange_out, i5)
            end

        end


        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5
    end
    ratio = 1
    ratio2 = 1

    for i5 in irange

        if xout.L5 != 1
            if i5 == 1
                j5 = xout.L5
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1minusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], -m * ratio * ratio2, temp1.w[i5])
                set_wing_fermion!(xout.w[i5])
            end

            if i5 == xout.L5
                j5 = 1
                mul_1plusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], -m * ratio * ratio2, temp1.w[i5])
                set_wing_fermion!(xout.w[i5])
            end
        end
    end

    #if L5 != xout.L5
    #    for i5 in irange_out
    #        axpy!(1,x.w[i5],xout.w[i5])
    #    end
    #end

end


function apply_Fdag!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    L5,
    m,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    temp1,
) where {NC,WilsonFermion}
    clear_fermion!(xout)

    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5

        for i5 = 1:xout.L5
            if i5 <= div(L5, 2) || i5 >= xout.L5 - div(L5, 2) + 1
                push!(irange, i5)
            else
                push!(irange_out, i5)
            end

        end


        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5
    end
    ratio = 1
    ratio2 = 1

    for i5 in irange
        j5 = i5 + 1
        if 1 <= j5 <= xout.L5
            #-P_- -> - P_+ :gamma_5 of LTK definition
            if xout.L5 != 2
                # ((1 + (b-c)/2 DW )*P_-*x)
                mul_1minusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], ratio * ratio2, temp1.w[i5])
                set_wing_fermion!(xout.w[i5])
            end
        end


        j5 = i5 - 1
        if 1 <= j5 <= xout.L5
            #-P_+ -> - P_- :gamma_5 of LTK definition
            if xout.L5 != 2
                mul_1plusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], ratio * ratio2, temp1.w[i5])
                set_wing_fermion!(xout.w[i5])
            end
        end

        if xout.L5 != 1
            if i5 == 1
                j5 = xout.L5
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1plusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], -m * ratio * ratio2, temp1.w[i5])
                #add!(ratio, xout.w[i5], 1 * ratio * ratio2, temp1.w[i5])
                set_wing_fermion!(xout.w[i5])
            end

            if i5 == xout.L5
                j5 = 1
                mul_1minusγ5x!(temp1.w[i5], x.w[j5])
                add!(ratio, xout.w[i5], -m * ratio * ratio2, temp1.w[i5])
                #add!(ratio, xout.w[i5], 1 * ratio * ratio2, temp1.w[i5])
                set_wing_fermion!(xout.w[i5])
            end
        end
    end

    #if L5 != xout.L5
    #    for i5 in irange_out
    #        axpy!(1,x.w[i5],xout.w[i5])
    #    end
    #end

end


function D5DWx!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    U::Array{G,1},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    m,
    A,
    L5,
    b,
    c,
    temp1,
    temp2,
) where {NC,WilsonFermion,G<:AbstractGaugefields}


    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    coeff_plus = (b + c) / 2
    # coeff_plus = 1
    coeff_minus = -(b - c) / 2
    # coeff_minus  = 0
    clear_fermion!(xout)
    ratio = 1

    factor = coeff_plus
    #xout = (1 + factor*D)*x
    #println("apply_1pD!")
    apply_1pD!(xout, L5, U, A, x, factor)



    #temp2 = F*x
    #println("apply_F!")
    apply_F!(temp2, L5, m, x, temp1)


    factor = coeff_minus
    #xout = (1 + factor*D)*F*x
    #println("apply_1pD!")
    apply_1pD!(temp1, L5, U, A, temp2, factor)
    for i5 = 1:L5
        # axpy!(-1, temp1.w[i5], xout.w[i5])
        add!(-1, xout.w[i5], 1, temp1.w[i5])
        #add!(0, xout.w[i5], 1, temp1.w[i5])
    end

    set_wing_fermion!(xout)


    return
end

function D5DWdagx!(
    xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    U::Array{G,1},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    m,
    A,
    L5,
    b,
    c,
    temp1,
    temp2,
) where {NC,WilsonFermion,G<:AbstractGaugefields}

    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    clear_fermion!(xout)
    ratio = 1
    coeff_plus = (b + c) / 2
    coeff_minus = -(b - c) / 2
    # coeff_plus = 1
    # coeff_minus  = 0
    #ratio = xout.L5/L5


    factor = coeff_minus
    #xout = Fdag*(1 + factor*Ddag)*x
    apply_1pDdag!(temp2, L5, U, A, x, factor)
    apply_Fdag!(xout, L5, m, temp2, temp1)

    factor = coeff_plus
    #xout = (1 + factor*Ddag)*x
    apply_1pDdag!(temp1, L5, U, A, x, factor)

    for i5 = 1:L5
        # axpy!(1, temp1.w[i5], xout.w[i5])
        add!(1, xout.w[i5], -1, temp1.w[i5])
    end

    set_wing_fermion!(xout)

    return
end


"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
with σ^2 = 1/2
c-------------------------------------------------c
"""
function gauss_distribution_fermion!(
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
) where {NC,WilsonFermion}
    L5 = length(x.w)
    for iL = 1:L5
        gauss_distribution_fermion!(x.w[iL])
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
function gauss_distribution_fermion!(
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    randomfunc,
    σ,
) where {NC,WilsonFermion}
    L5 = length(x.w)
    for iL = 1:L5
        gauss_distribution_fermion!(x.w[iL], randomfunc, σ)
    end
    return

end

function Z4_distribution_fermi!(x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion}) where {NC,WilsonFermion}
    # (x::AbstractFermionfields_5D{NC}) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.w[1].f)[6]
    θ = 0.0
    N::Int32 = 4
    Ninv = Float64(1 / N)
    clear_fermion!(x)
    for ialpha = 1:n6
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        @inbounds @simd for ic = 1:NC
                            θ = Float64(rand(0:N-1)) * π * Ninv # r \in [0,π/4,2π/4,3π/4]
                            x.w[1][ic, ix, iy, iz, it, ialpha] = cos(θ) + im * sin(θ)
                        end
                    end
                end
            end
        end
    end
    set_wing_fermion!(x)
    return
end