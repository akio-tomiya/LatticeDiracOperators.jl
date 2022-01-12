using Wilsonloop

struct General_Dirac_term{Dim}
    matrix_for_spinors::Matrix{ComplexF64}
    lines_for_gaugefields::Tuple{Wilsonline{Dim},Wilsonline{Dim}}
    startingpoints:::Tuple{NTuple{Int64,Dim},NTuple{Int64,Dim}}
    endingpoints::Tuple{NTuple{Int64,Dim},NTuple{Int64,Dim}}
end

struct Wilson_GeneralDirac_operator{Dim,T,fermion} <: Dirac_operator{Dim}  where T <: AbstractGaugefields
    U::Array{T,1}
    boundarycondition::Vector{Int8}
    _temporary_fermi::Vector{fermion}
    dirac_terms::Vector{General_Dirac_term{Dim}}
    coefficients::Vector{Float64}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
end

function onelink_diracterm(μ,κ,Dim)
    startingpoint = zeros(Int64,Dim)
    endingpoint = zeros(Int64,Dim)
    endingpoint[μ] = 1

    loop_forward = [(μ,1)]
    loop_backward = [(μ,-1)]

    lines_for_gaugefields = (Wilsonline(loop_forward),Wilsonline(loop_backward))
    startingpoints = (Tuple(startingpoint),Tuple(startingpoint))
    endingpoints = (Tuple(endingpoint),Tuple(-endingpoint))
    r = 1

    γ,rplusγ,rminusγ = mk_gamma(r)


end

function easymode_constructing_diraclines(parameters,Dim)
    nameofterms = check_important_parameters(parameters,"nameofterms")
    numterms = length(nameofterms) 

    dirac_terms =General_Dirac_line{Dim}[]

    for i=1:numterms
        term = nameofterms[i]
        if term == "1-link"
            κ = check_important_parameters(parameters,"κ")
            for μ=1:Dim
                diracterm = onelink_diracterm(μ,κ,Dim)
                push!(dirac_terms,diracterm)
            end
        elseif term == "clover"
        end
    end

end

function expartmode_constructing_diraclines(parameters,Dim)
    dirac_terms = check_important_parameters(parameters,"dirac_terms")
    return dirac_terms
end



function Wilson_GeneralDirac_operator(U::Array{<: AbstractGaugefields{NC,Dim},1},x,parameters) where {NC,Dim}
    xtype = typeof(x)
    num = 7
    _temporary_fermi = Array{xtype,1}(undef,num)

    boundarycondition = check_parameters(parameters,"boundarycondition",[1,1,1,-1])

    inputmode = check_parameters(parameters,"inputmode","easy")
    coefficients = check_important_parameters(parameters,"coefficients")

    if inputmode == "expart"
        println("expart mode for constructing Dirac terms")
        dirac_terms = expartmode_constructing_diraclines(parameters,Dim)
    elseif inputmode == "easy"
        println("easy mode for constructing Dirac terms")
        dirac_terms = easymode_constructing_diraclines(parameters,Dim)
    else
        error("inputmode $inputmode is not supported")
    end


    for i=1:num
        _temporary_fermi[i] = similar(x)
    end

    eps_CG = check_parameters(parameters,"eps_CG",default_eps_CG)
    MaxCGstep = check_parameters(parameters,"MaxCGstep",default_MaxCGstep)
    verbose_level = check_parameters(parameters,"verbose_level",2)
    method_CG = check_parameters(parameters,"method_CG","bicg")


    for i=1:num
        _temporary_fermi[i] = similar(x)
    end

    if verbose_level == 1 
        verbose = Verbose_1()
    elseif verbose_level == 2
        verbose = Verbose_2()
    elseif verbose_level == 3
        verbose = Verbose_3()
    else
        error("verbose_level = $verbose_level is not supported")
    end 

    return Wilson_GeneralDirac_operator{Dim,eltype(U),xtype}(U,boundarycondition,_temporary_fermi,
        dirac_terms,
        coefficients,
        eps_CG,MaxCGstep,
        verbose_level,method_CG
        )
end



