using Wilsonloop
import Wilsonloop: DwDU, get_direction, get_position#,Adjoint_GLink
import Gaugefields: evaluate_gaugelinks!, clear_U!, set_wing_U!



g1,g2,g3 = mk_gamma(1)
const γmatrices = g1 
#const γ5 = γmatrices[:,:,5]

#const rplusγ1 = g2
#const rminusγ1 = g3


function shift_wilsonloop(w::Wilsonline{Dim}, shift) where {Dim}
    wshifted = Wilsonline(Dim = Dim)
    #show(w)
    for i = 1:length(w)
        link = w[i]
        position = collect(get_position(link)) .- shift
        #println("link: ", get_position(link))
        #println("position, ",position)
        link_shifted = Wilsonloop.set_position(link, Tuple(position))
        push!(wshifted, link_shifted)
    end
    return wshifted
end

struct Linkinfo{Dim}
    positions::Vector{NTuple{Dim,Int64}}
    directions::Vector{Int8}
    isdagvectors::Vector{Bool}

    function Linkinfo(line_for_gaugefields::Wilsonline{Dim}) where {Dim}
        numlinks = length(line_for_gaugefields)
        positions = Array{NTuple{Dim,Int64},1}(undef, numlinks)
        directions = Array{Int8,1}(undef, numlinks)
        isdagvectors = Array{Bool,1}(undef, numlinks)
        for i = 1:numlinks
            glink = line_for_gaugefields[i]
            positions[i] = get_position(glink)
            directions[i] = get_direction(glink)
            isdagvectors[i] = isdag(glink)
            #isdagvectors[i] = ifelse(typeof(glink ) <: Adjoint_GLink,true,false)
        end
        return new{Dim}(positions, directions, isdagvectors)
    end
end

struct General_Dirac_term{Dim}
    coefficient::Float64
    matrix_for_spinors::Matrix{ComplexF64}
    matrix_for_spinors_dagger::Matrix{ComplexF64}
    matrix_for_spinors_γ5M::Matrix{ComplexF64}
    line_for_gaugefields::Wilsonline{Dim}
    linkinfo::Linkinfo{Dim}
    #positions::Vector{NTuple{Dim,Int64}}
    #directions::Vector{Int8}
    #isdagvectors::Vector{Bool}
    derivatives::Vector{Vector{DwDU{Dim}}}
    linkinfo_derivatives_right::Vector{Vector{Linkinfo{Dim}}}
    linkinfo_derivatives_leftdag::Vector{Vector{Linkinfo{Dim}}}
    derivatives_dag::Vector{Vector{DwDU{Dim}}}
    linkinfo_derivatives_dag_right::Vector{Vector{Linkinfo{Dim}}}
    linkinfo_derivatives_dag_leftdag::Vector{Vector{Linkinfo{Dim}}}
    startingpoint::NTuple{Dim,Int64}
    endingpoint::NTuple{Dim,Int64}

    function General_Dirac_term(
        coefficient,
        matrix_for_spinors,
        line_for_gaugefields::Wilsonline{Dim},
        startingpoint,
        endingpoint,
    ) where {Dim}

        matrix_for_spinors_dagger = γ5 * matrix_for_spinors * γ5
        matrix_for_spinors_γ5M = γ5 * matrix_for_spinors

        linkinfo = Linkinfo(line_for_gaugefields)

        derivatives = Array{Vector{DwDU{Dim}},1}(undef, Dim)
        linkinfo_derivatives_right = Vector{Vector{Linkinfo{Dim}}}(undef, Dim)
        linkinfo_derivatives_leftdag = Vector{Vector{Linkinfo{Dim}}}(undef, Dim)

        derivatives_dag = Array{Vector{DwDU{Dim}},1}(undef, Dim)
        linkinfo_derivatives_dag_right = Vector{Vector{Linkinfo{Dim}}}(undef, Dim)
        linkinfo_derivatives_dag_leftdag = Vector{Vector{Linkinfo{Dim}}}(undef, Dim)


        for μ = 1:Dim
            derivatives[μ] = derive_U(line_for_gaugefields, μ)
            numderivatives = length(derivatives[μ])
            linkinfo_derivatives_right[μ] = Vector{Linkinfo{Dim}}(undef, numderivatives)
            linkinfo_derivatives_leftdag[μ] = Vector{Linkinfo{Dim}}(undef, numderivatives)

            println("$μ ----link")
            show(line_for_gaugefields)
            println("--- derivatives")
            show(derivatives[μ])
            println("\n derivatives")


            for i = 1:numderivatives
                rightlinks = get_rightlinks(derivatives[μ][i])
                #shiftedposition = -2*collect(derivatives[μ][i].position)
                #rightlinks = shift_wilsonloop(rightlinks,Tuple(shiftedposition))
                leftlinks = get_leftlinks(derivatives[μ][i])
                #leftlinks = shift_wilsonloop(leftlinks,Tuple(shiftedposition))
                linkinfo_derivatives_right[μ][i] = Linkinfo(rightlinks)
                linkinfo_derivatives_leftdag[μ][i] = Linkinfo(leftlinks')

                println("position")
                println(derivatives[μ][i].position)
                println("")
                println("right")
                show(rightlinks)
                println("")
                println("left")
                show(leftlinks')
                println("")
            end



            println("μ = $μ")

            show(line_for_gaugefields')
            dagline = shift_wilsonloop(line_for_gaugefields', endingpoint)
            show(dagline)
            derivatives_dag[μ] = derive_U(dagline, μ)
            show(derivatives_dag[μ])

            numderivatives = length(derivatives_dag[μ])
            linkinfo_derivatives_dag_right[μ] = Vector{Linkinfo{Dim}}(undef, numderivatives)
            linkinfo_derivatives_dag_leftdag[μ] =
                Vector{Linkinfo{Dim}}(undef, numderivatives)
            for i = 1:numderivatives
                println("position = ", derivatives_dag[μ][i].position)
                linkinfo_derivatives_dag_right[μ][i] =
                    Linkinfo(get_rightlinks(derivatives_dag[μ][i]))
                linkinfo_derivatives_dag_leftdag[μ][i] =
                    Linkinfo(get_leftlinks(derivatives_dag[μ][i])')
            end
        end

        return new{Dim}(
            coefficient,
            matrix_for_spinors,
            matrix_for_spinors_dagger,
            matrix_for_spinors_γ5M,
            line_for_gaugefields,
            linkinfo,
            derivatives,
            linkinfo_derivatives_right,
            linkinfo_derivatives_leftdag,
            derivatives_dag,
            linkinfo_derivatives_dag_right,
            linkinfo_derivatives_dag_leftdag,
            startingpoint,
            endingpoint,
        )
    end

end

function get_positions(d::General_Dirac_term)
    return d.positions
end

function get_directions(d::General_Dirac_term)
    return d.directions
end

function get_isdagvectors(d::General_Dirac_term)
    return d.isdagvectors
end

function get_fermionposition(d::General_Dirac_term)
    return d.endingpoint
end


function get_spinormatrix(d::General_Dirac_term)
    return d.matrix_for_spinors
end


function get_spinormatrix_γ5M(d::General_Dirac_term)
    return d.matrix_for_spinors_γ5M
end



function get_spinormatrix_dagger(d::General_Dirac_term)
    return d.matrix_for_spinors_dagger
end


function get_wilsonline(d::General_Dirac_term)
    return d.line_for_gaugefields
end

function get_coefficient(d::General_Dirac_term)
    return d.coefficient
end

function get_linkinfo(d::General_Dirac_term)
    return d.linkinfo
end

function get_derivatives(d::General_Dirac_term, μ)
    return d.derivatives[μ]
end

function get_linkinfo_derivatives_leftdag(d::General_Dirac_term, μ)
    return d.linkinfo_derivatives_leftdag[μ]
end

function get_linkinfo_derivatives_right(d::General_Dirac_term, μ)
    return d.linkinfo_derivatives_right[μ]
end

function get_derivatives_dag(d::General_Dirac_term, μ)
    return d.derivatives_dag[μ]
end

function get_linkinfo_derivatives_dag_leftdag(d::General_Dirac_term, μ)
    return d.linkinfo_derivatives_dag_leftdag[μ]
end

function get_linkinfo_derivatives_dag_right(d::General_Dirac_term, μ)
    return d.linkinfo_derivatives_dag_right[μ]
end



function Base.show(g::General_Dirac_term{Dim}) where {Dim}
    println("coefficient: ", g.coefficient)

    println("details: ")
    println("  gauge links: ")
    show(g.line_for_gaugefields)
    println("  matrix for spinors: ")
    display(g.matrix_for_spinors)
    println("\t")
    println("  hermitian-conjugate matrix for spinors: ")
    display(g.matrix_for_spinors_dagger)
    println("\t")
    for μ = 1:Dim
        println("dD/dU$μ: ")
        show(g.derivatives[μ])
    end

end


struct Wilson_GeneralDirac_operator{Dim,T,fermion} <:
       Dirac_operator{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    boundarycondition::Vector{Int8}
    _temporary_fermi::Vector{fermion}
    dirac_terms::Vector{General_Dirac_term{Dim}}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    _temporary_gaugefields::Vector{T}
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Vector{fermion}
end


function Wilson_GeneralDirac_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    numg = 3
    _temporary_gaugefields = Array{eltype(U),1}(undef, numg)
    for i = 1:numg
        _temporary_gaugefields[i] = similar(U[1])
    end

    xtype = typeof(x)
    num = 7
    _temporary_fermi = Array{xtype,1}(undef, num)

    boundarycondition = check_parameters(parameters, "boundarycondition", [1, 1, 1, -1])

    inputmode = check_parameters(parameters, "inputmode", "easy")

    if inputmode == "expart"
        println("expart mode for constructing Dirac terms")
        dirac_terms = expartmode_constructing_diraclines(parameters, Dim)
    elseif inputmode == "easy"
        println("easy mode for constructing Dirac terms")
        dirac_terms = easymode_constructing_diraclines(parameters, Dim)
    else
        error("inputmode $inputmode is not supported")
    end


    for i = 1:num
        _temporary_fermi[i] = similar(x)
    end

    eps_CG = check_parameters(parameters, "eps_CG", default_eps_CG)
    MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)
    verbose_level = check_parameters(parameters, "verbose_level", 2)
    method_CG = check_parameters(parameters, "method_CG", "bicg")


    verbose_print = Verbose_print(verbose_level)


    for i = 1:num
        _temporary_fermi[i] = similar(x)
    end

    numcg = check_parameters(parameters, "numtempvec_CG", 7)
    #numcg = 7
    _temporary_fermion_forCG = Array{xtype,1}(undef, numcg)
    for i = 1:numcg
        _temporary_fermion_forCG[i] = similar(x)
    end




    return Wilson_GeneralDirac_operator{Dim,eltype(U),xtype}(
        U,
        boundarycondition,
        _temporary_fermi,
        dirac_terms,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        _temporary_gaugefields,
        verbose_print,
        _temporary_fermion_forCG,
    )
end

function (D::Wilson_GeneralDirac_operator{Dim,T,fermion})(U) where {Dim,T,fermion}
    return Wilson_GeneralDirac_operator{Dim,T,fermion}(
        U,
        D.boundarycondition,
        D._temporary_fermi,
        D.dirac_terms,
        D.eps_CG,
        D.MaxCGstep,
        D.verbose_level,
        D.method_CG,
        D._temporary_gaugefields,
        D.verbose_print,
        D._temporary_fermion_forCG,
    )
end

struct DdagD_Wilson_GeneralDirac_operator{Dim,T,fermion} <: DdagD_operator
    dirac::Wilson_GeneralDirac_operator{Dim,T,fermion}
    function DdagD_Wilson_GeneralDirac_operator(
        U::Array{<:AbstractGaugefields{NC,Dim},1},
        x,
        parameters,
    ) where {NC,Dim}
        return new{Dim,eltype(U),typeof(x)}(Wilson_GeneralDirac_operator(U, x, parameters))
    end

    function DdagD_Wilson_GeneralDirac_operator(
        D::Wilson_GeneralDirac_operator{Dim,T,fermion},
    ) where {Dim,T,fermion}
        return new{Dim,T,fermion}(D)
    end
end



function get_numterms(w::Wilson_GeneralDirac_operator)
    return length(w.dirac_terms)
end



function Base.getindex(w::Wilson_GeneralDirac_operator, i)
    return w.dirac_terms[i]
end

struct Adjoint_Wilson_GeneralDirac_operator{T} <: Adjoint_Dirac_operator
    parent::T
end

function Base.adjoint(A::T) where {T<:Wilson_GeneralDirac_operator}
    Adjoint_Wilson_GeneralDirac_operator{typeof(A)}(A)
end


function Base.show(D::Wilson_GeneralDirac_operator{Dim,T,fermion}) where {Dim,T,fermion}
    println("----------------------------------------------")
    println("Wilson Dirac operator: general type")
    println("Method for solving y = D^inv x: ", D.method_CG)
    println("eps for CG: ", D.eps_CG)
    println("Mac steps for CG: ", D.MaxCGstep)
    println("verbose level: ", D.verbose_level)
    numdirac = length(D.dirac_terms)
    println("num. of Dirac terms: ", numdirac)
    for i = 1:numdirac
        if i == 1
            st = "st"
        elseif i == 2
            st = "nd"
        elseif i == 3
            st = "rd"
        else
            st = "th"
        end
        println("---------------")
        println("$i-$st term: ")
        show(D.dirac_terms[i])
        println("---------------")
    end
    println("----------------------------------------------")

end



function onelink_diracterm(μ, κ, Dim)
    dirac_terms = General_Dirac_term{Dim}[]

    startingpoint = zeros(Int64, Dim)
    endingpoint = zeros(Int64, Dim)
    endingpoint[μ] = 1

    loop_forward = [(μ, 1)]
    line_for_gaugefields = Wilsonline(loop_forward)
    matrix_for_spinors = rminusγ1[:, :, μ]
    dirac_term = General_Dirac_term(
        κ,
        matrix_for_spinors,
        line_for_gaugefields,
        Tuple(startingpoint),
        Tuple(endingpoint),
    )

    push!(dirac_terms, dirac_term)

    loop_backward = [(μ, -1)]
    line_for_gaugefields = Wilsonline(loop_backward)
    matrix_for_spinors = rplusγ1[:, :, μ]
    endingpoint[μ] = -1
    dirac_term = General_Dirac_term(
        κ,
        matrix_for_spinors,
        line_for_gaugefields,
        Tuple(startingpoint),
        Tuple(endingpoint),
    )
    push!(dirac_terms, dirac_term)
    return dirac_terms

end

function cloverlink_diracterm_debug(μ, ν, cSW, Dim)
    dirac_terms = General_Dirac_term{Dim}[]
    c = -cSW / (16)

    startingpoint = zeros(Int64, Dim)
    endingpoint = zeros(Int64, Dim)

    γμ = γmatrices[:, :, μ]
    γν = γmatrices[:, :, ν]

    σμν = (γμ * γν .- γν * γμ) * (1 / 2)

    loop_righttop = Wilsonline([(μ, 1), (ν, 1), (μ, -1), (ν, -1)])
    loop_lefttop = Wilsonline([(ν, 1), (μ, -1), (ν, -1), (μ, 1)])
    loop_rightbottom = Wilsonline([(ν, -1), (μ, 1), (ν, 1), (μ, -1)])
    loop_leftbottom = Wilsonline([(μ, -1), (ν, -1), (μ, 1), (ν, 1)])

    #loopterm = loop_righttop #correct
    #loopterm = loop_lefttop #not correct
    #loopterm = loop_rightbottom #not correct
    loopterm = loop_leftbottom #not correct
    #loopterm = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)])

    dirac_term =
        General_Dirac_term(c, σμν, loopterm, Tuple(startingpoint), Tuple(endingpoint))

    push!(dirac_terms, dirac_term)

    dirac_term =
        General_Dirac_term(-c, σμν, loopterm', Tuple(startingpoint), Tuple(endingpoint))

    push!(dirac_terms, dirac_term)


    return dirac_terms

end

function cloverlink_diracterm(μ, ν, cSW, Dim)
    dirac_terms = General_Dirac_term{Dim}[]
    c = -cSW / (16)

    startingpoint = zeros(Int64, Dim)
    endingpoint = zeros(Int64, Dim)

    γμ = γmatrices[:, :, μ]
    γν = γmatrices[:, :, ν]

    σμν = (γμ * γν .- γν * γμ) * (1 / 2)

    loop_righttop = Wilsonline([(μ, 1), (ν, 1), (μ, -1), (ν, -1)])
    loop_lefttop = Wilsonline([(ν, 1), (μ, -1), (ν, -1), (μ, 1)])
    loop_rightbottom = Wilsonline([(ν, -1), (μ, 1), (ν, 1), (μ, -1)])
    loop_leftbottom = Wilsonline([(μ, -1), (ν, -1), (μ, 1), (ν, 1)])

    dirac_term =
        General_Dirac_term(c, σμν, loop_righttop, Tuple(startingpoint), Tuple(endingpoint))

    push!(dirac_terms, dirac_term)

    dirac_term =
        General_Dirac_term(c, σμν, loop_lefttop, Tuple(startingpoint), Tuple(endingpoint))

    push!(dirac_terms, dirac_term)

    dirac_term = General_Dirac_term(
        c,
        σμν,
        loop_rightbottom,
        Tuple(startingpoint),
        Tuple(endingpoint),
    )

    push!(dirac_terms, dirac_term)

    dirac_term = General_Dirac_term(
        c,
        σμν,
        loop_leftbottom,
        Tuple(startingpoint),
        Tuple(endingpoint),
    )

    push!(dirac_terms, dirac_term)


    dirac_term = General_Dirac_term(
        -c,
        σμν,
        loop_righttop',
        Tuple(startingpoint),
        Tuple(endingpoint),
    )

    push!(dirac_terms, dirac_term)

    dirac_term =
        General_Dirac_term(-c, σμν, loop_lefttop', Tuple(startingpoint), Tuple(endingpoint))

    push!(dirac_terms, dirac_term)

    dirac_term = General_Dirac_term(
        -c,
        σμν,
        loop_rightbottom',
        Tuple(startingpoint),
        Tuple(endingpoint),
    )

    push!(dirac_terms, dirac_term)

    dirac_term = General_Dirac_term(
        -c,
        σμν,
        loop_leftbottom',
        Tuple(startingpoint),
        Tuple(endingpoint),
    )

    push!(dirac_terms, dirac_term)

    return dirac_terms

end


function easymode_constructing_diraclines(parameters, Dim)
    nameofterms = check_important_parameters(parameters, "nameofterms", "[\"one-link\"]")
    @assert typeof(nameofterms) <: Vector "nameofterms is not a vector. Now $nameofterms"
    println(nameofterms)
    numterms = length(nameofterms)

    dirac_terms = General_Dirac_term{Dim}[]

    for i = 1:numterms
        term = nameofterms[i]
        if term == "one-link"
            println("Conventional Wilson terms (1+γ) and (1-γ) are added")
            κ = check_important_parameters(parameters, "κ")
            for μ = 1:Dim
                diracterms = onelink_diracterm(μ, κ, Dim)
                append!(dirac_terms, diracterms)
            end

        elseif term == "clover"
            c = check_important_parameters(parameters, "cSW")
            for μ = 1:Dim
                for ν = μ:Dim
                    if ν == μ
                        continue
                    end
                    diracterms = cloverlink_diracterm(μ, ν, c, Dim)
                    append!(dirac_terms, diracterms)
                end
            end
            #println("dd ",length(dirac_terms))
        elseif term == "cloverdebug"
            c = check_important_parameters(parameters, "cSW")
            μ = 1
            ν = 2
            diracterms = cloverlink_diracterm_debug(μ, ν, c, Dim)
            append!(dirac_terms, diracterms)
            #println("dd ",length(dirac_terms))
        else
            error("term $term is not supported")
        end
    end
    #println("dd ",length(dirac_terms))
    return dirac_terms

end

function expartmode_constructing_diraclines(parameters, Dim)
    dirac_terms = check_important_parameters(parameters, "dirac_terms")
    return dirac_terms
end





function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {
    T1<:AbstractFermionfields,
    T2<:Wilson_GeneralDirac_operator,
    T3<:AbstractFermionfields,
}
    Wx_general!(y, A.U, x, A)
end

function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {
    T1<:AbstractFermionfields,
    T2<:Adjoint_Wilson_GeneralDirac_operator,
    T3<:AbstractFermionfields,
}
    Wdagx_general!(y, A.parent.U, x, A.parent)
end




function Wx_general!(
    xout::T,
    U::Array{G,1},
    x::T,
    A::Wilson_GeneralDirac_operator,
) where {T<:WilsonFermion_4D_wing,G<:AbstractGaugefields}
    #temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_4D_wing,G <: AbstractGaugefields}
    temp = A._temporary_fermi[4]#temps[4]
    temp1 = A._temporary_fermi[1] #temps[1]
    temp2 = A._temporary_fermi[2] #temps[2]

    tempg = A._temporary_gaugefields
    V = tempg[1]
    tempg1 = tempg[2]
    tempg2 = tempg[3]

    clear_fermion!(xout)
    clear_fermion!(temp2)
    numterms = get_numterms(A)

    for i = 1:numterms
        diracterm = A[i]
        position = get_fermionposition(diracterm)
        xshifted = shift_fermion(x, position)
        wilsonline = get_wilsonline(diracterm)

        linkinfo = get_linkinfo(diracterm)

        gaugefield_fermion_mul!(temp1, linkinfo, U, xshifted, A._temporary_fermi)

        #@time gaugefield_fermion_mul!(temp1,wilsonline,U,xshifted,[temp])

        #println(position)

        #wilsonline = get_wilsonline(diracterm)
        #show(wilsonline)

        #evaluate_gaugelinks!(V,wilsonline,U,[tempg1,tempg2]) 
        #set_wing_U!(V)

        #Vshift = shift_U(V,position)
        #mul!(temp1,V,xshifted)

        S = get_spinormatrix(diracterm)

        mul!(temp1, S)

        coeff = get_coefficient(diracterm)

        add_fermion!(temp2, coeff, temp1)

        #println(xout[1,1,1,1,1,1])
    end
    #error("d")

    add_fermion!(xout, 1, x, -1, temp2)
    set_wing_fermion!(xout, A.boundarycondition)

    #display(xout)
    #    exit()
    return
end

function Wdagx_general!(
    xout::T,
    U::Array{G,1},
    x::T,
    A::Wilson_GeneralDirac_operator,
) where {T<:WilsonFermion_4D_wing,G<:AbstractGaugefields}
    #temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_4D_wing,G <: AbstractGaugefields}
    temp = A._temporary_fermi[4]#temps[4]
    temp1 = A._temporary_fermi[1] #temps[1]
    temp2 = A._temporary_fermi[2] #temps[2]

    tempg = A._temporary_gaugefields
    V = tempg[1]
    tempg1 = tempg[2]
    tempg2 = tempg[3]

    clear_fermion!(temp2)
    clear_fermion!(xout)
    numterms = get_numterms(A)

    for i = 1:numterms
        diracterm = A[i]
        position = get_fermionposition(diracterm)
        xshifted = shift_fermion(x, position)

        #wilsonline = get_wilsonline(diracterm)
        linkinfo = get_linkinfo(diracterm)
        #positions = get_positions(diracterm)
        #directions = get_directions(diracterm)
        #isdagvectors = get_isdagvectors(diracterm)
        gaugefield_fermion_mul!(temp1, linkinfo, U, xshifted, A._temporary_fermi)

        #clear_U!(V)
        #evaluate_gaugelinks!(V,wilsonline,U,[tempg1,tempg2]) 
        #set_wing_U!(V)

        #Vshift = shift_U(V,position)
        #mul!(temp1,V,xshifted)

        S = get_spinormatrix_dagger(diracterm)

        mul!(temp1, S)

        coeff = get_coefficient(diracterm)

        add_fermion!(temp2, coeff, temp1)
    end

    add_fermion!(xout, 1, x, -1, temp2)
    set_wing_fermion!(xout, A.boundarycondition)
    #set_wing_fermion!(xout,A.boundarycondition)
    #error("rr")
    #display(xout)
    #    exit()
    return
end

const zeroposition = [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]

function gaugefield_fermion_mul!(
    xout,
    linkinfo,
    U::Array{T,1},
    x,
    temps_fermion::Array{<:AbstractFermionfields{NC,Dim},1},
) where {T<:AbstractGaugefields,Dim,NC}
    positions = linkinfo.positions
    directions = linkinfo.directions
    isdagvectors = linkinfo.isdagvectors
    temp1 = temps_fermion[end]
    temp2 = temps_fermion[end-1]

    #glinks = w
    numlinks = length(positions)
    if numlinks == 0
        substitute_fermion!(xout, x)
        #error("no gaugefield found")
        return
    end


    #println(typeof(position))

    if numlinks == 1
        position = positions[numlinks]
        direction = directions[numlinks]
        isUenddag = isdagvectors[numlinks]

        if position == zeroposition[Dim]
            if isUenddag
                mul!(xout, U[direction]', x)
            else
                mul!(xout, U[direction], x)
            end
        else
            Ushift = shift_U(U[direction], position)
            if isUenddag
                mul!(xout, Ushift', x)
            else
                mul!(xout, Ushift, x)
            end
        end
        return
    else
        position = positions[numlinks]
        direction = directions[numlinks]
        isUenddag = isdagvectors[numlinks]

        if position == zeroposition[Dim]
            if isUenddag
                mul!(temp1, U[direction]', x)
            else
                mul!(temp1, U[direction], x)
            end
        else
            Ushift = shift_U(U[direction], position)
            if isUenddag
                mul!(temp1, Ushift', x)
            else
                mul!(temp1, Ushift, x)
            end
        end
    end


    for j = numlinks-1:-1:2
        position = positions[j]
        direction = directions[j]
        isUdag = isdagvectors[j]

        if position == zeroposition[Dim]
            if isUdag
                mul!(temp2, U[direction]', temp1)
            else
                mul!(temp2, U[direction], temp1)
            end
        else
            Ushift = shift_U(U[direction], position)
            if isUdag
                mul!(temp2, Ushift', temp1)
            else
                mul!(temp2, Ushift, temp1)
            end
        end

        temp2, temp1 = temp1, temp2
    end

    position = positions[1]
    direction = directions[1]
    isU1dag = isdagvectors[1]

    if position == zeroposition[Dim]
        if isU1dag
            mul!(xout, U[direction]', temp1)
        else
            mul!(xout, U[direction], temp1)
        end
    else
        Ushift = shift_U(U[direction], position)
        if isU1dag
            mul!(xout, Ushift', temp1)
        else
            mul!(xout, Ushift, temp1)
        end
    end






end
