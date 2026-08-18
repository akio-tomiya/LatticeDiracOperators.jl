
using Wilsonloop
import Wilsonloop: make_cloverloops
import Gaugefields: AbstractGaugefields_module.Antihermitian!

struct WilsonClover{Dim,TG}
    cSW::Float64
    cloverloops::Matrix{Vector{Wilsonline{Dim}}}
    internal_flags::Array{Bool,1}
    inn_table::Array{Int64,3}
    _ftmp_vectors::Array{Array{ComplexF64,3},1}
    _is1::Array{Int64,1}
    _is2::Array{Int64,1}
    CloverFμν::Vector{TG}
    #CloverFμν::Array{ComplexF64,4}
    temp_gaugefields::Vector{TG}
    factor::Float64
    #dcoverloopsdU::Matrix{Vector{Vector{Wilsonloop.DwDU{Dim}}}}
    #dcoverloopsdagdU::Matrix{Vector{Vector{Wilsonloop.DwDU{Dim}}}}
    #σ::Array{Array{ComplexF64,2},2}
end


function WilsonClover(cSW, Dim, NV, U, hop)
    #σ = make_σμν()
    cloverloops = Matrix{Vector{Wilsonline{Dim}}}(undef, Dim, Dim)

    #dcloverloopsdU = Matrix{Vector{Vector{DwDU{Dim}}}}(undef,Dim,Dim)
    #dcloverloopsdagdU = Matrix{Vector{Vector{DwDU{Dim}}}}(undef,Dim,Dim)


    for μ = 1:Dim
        for ν = 1:Dim
            #loops = Wilsonline{Dim}[]
            #loop_righttop = Wilsonline([(μ, 1), (ν, 1), (μ, -1), (ν, -1)])
            #push!(loops, loop_righttop)


            cloverloops[μ, ν] = make_cloverloops(μ, ν; Dim=Dim)
            #dcloverloopsdU[μ, ν] = Vector{Vector{DwDU{Dim}}}(undef,Dim) 
            #dcloverloopsdagdU[μ, ν] = Vector{Vector{DwDU{Dim}}}(undef,Dim) 
            #=
            for μd=1:Dim
                dcloverloopsdU[μ, ν][μd] =  DwDU{Dim}[]
                dcloverloopsdagdU[μ, ν][μd] =  DwDU{Dim}[]
            end

            for μd=1:Dim
                for loop in cloverloops[μ, ν]
                    dU = derive_U(loop,μd)
                    for dUi in dU
                        #show(dUi)
                        push!(dcloverloopsdU[μ, ν][μd] ,dUi)
                    end
                end
                for loop in cloverloops[μ, ν]
                    dU = derive_U(loop',μd)
                    for dUi in dU
                        #show(dUi)
                        push!(dcloverloopsdagdU[μ, ν][μd] ,dUi)
                    end
                end
            end
            =#
        end
    end


    #error("dd")


    numtemp = 5
    TG = eltype(U)
    temp_gaugefields = Vector{TG}(undef, numtemp)
    for i = 1:numtemp
        temp_gaugefields[i] = similar(U[1])
    end


    inn_table = zeros(Int64, NV, 4, 2)
    internal_flags = zeros(Bool, 2)
    _ftmp_vectors = Array{Array{ComplexF64,3},1}(undef, 6)
    _is1 = zeros(Int64, NV)
    _is2 = zeros(Int64, NV)

    factor = cSW * hop


    CloverFμν = Make_CloverFμν(U, temp_gaugefields, cloverloops, factor)
    #asum = 0.0
    #=
    asum2 = 0.0
    temps = eltype(U)[]
    ni = 4
    for i=1:ni
        push!(temps,similar(U[1]))
    end
    CloverFμν2 =  Gaugefields.make_Cloverloopterms(U,temps)

    for i=1:length(CloverFμν)
        asum += sum(abs.(CloverFμν[i].U))
        asum2 += sum(abs.(CloverFμν2[i].U))*factor*0.125
    end
    println("sum(abs.(CloverFμν )) $asum $asum2")
    =#


    return WilsonClover{Dim,TG}(
        cSW, cloverloops, internal_flags, inn_table, _ftmp_vectors, _is1, _is2, CloverFμν,
        temp_gaugefields, factor)#,dcloverloopsdU,dcoverloopsdagdU)
    #return new(cSW, σ)
end

function (D::WilsonClover{Dim,TG})(U) where {Dim,TG}
    Make_CloverFμν!(D.CloverFμν, U, D.temp_gaugefields, D.cloverloops, D.factor)
    #println("clover $(sum(abs.(D.CloverFμν[1].U)))")
    #println("U ",sum(abs.(U[1].U)))
    return WilsonClover{Dim,TG}(
        D.cSW, D.cloverloops, D.internal_flags, D.inn_table, D._ftmp_vectors, D._is1, D._is2, D.CloverFμν,
        D.temp_gaugefields, D.factor)
end




function Make_CloverFμν(U::Array{<:AbstractGaugefields{NC,Dim},1}, temps::Vector{T}, cloverloops, factor) where {T,NC,Dim}
    NV = temps[1].NV
    CloverFμν = Vector{T}(undef, 6)
    for μν = 1:6
        CloverFμν[μν] = similar(U[1])
    end
    #CloverFμν = zeros(ComplexF64,NC,NC,NV,6)
    Make_CloverFμν!(CloverFμν, U, temps, cloverloops, factor)
    return CloverFμν
end

function Make_CloverFμν!(CloverFμν, U::Array{<:AbstractGaugefields{NC,Dim},1}, temps, cloverloops, factor) where {NC,Dim}
    @assert Dim == 4 "Only Dim = 4 case is supported. Now Dim = $Dim"
    work1 = temps[4]
    work2 = temps[5]

    coe = im * 0.125 * factor

    # ... Calculation of 4 leaves under the counter clock order.
    μν = 0
    for μ = 1:3
        for ν = μ+1:4
            μν += 1
            if μν > 6
                error("μν > 6 ?")
            end

            loops = cloverloops[μ, ν]

            evaluate_gaugelinks!(work1, loops, U, temps)
            #println(coe)
            #for i=1:4
            #    println("work1 ",work1[:,:,i,1,1,1],"\n")
            #end
            #Antihermitian!(CloverFμν[μν],work1,factor=coe) #work - work^+

            Antihermitian!(work2, work1) #work - work^+
            mul!(CloverFμν[μν], coe, work2)
            #for i=1:4
            #    println("c1 ",CloverFμν[μν][:,:,i,1,1,1],"\n")
            #end
            #substitute_U!(CloverFμν[μν],work1)

            #loopset = Loops(U,fparam._cloverloops[μ,ν],[work2,work3,work4])
            #evaluate_loops!(work1,loopset,U)
            #setFμν!(CloverFμν,μν,work1)

        end
    end
    #error("clover")



end



function make_σμν()
    σ = Array{Array{ComplexF64,2},2}(undef, 4, 4)
    for μ = 1:4
        γμ = γ_all[:, :, μ]
        for ν = 1:4
            γν = γ_all[:, :, ν]
            σ[μ, ν] = (γμ * γν .- γν * γμ) * (1 / 2)
        end
    end
    return σ
end



"""
    cloverterm_σμν!(vec,cloverterm,x,temp1,temp2)

TBW
"""
function cloverterm_σμν!(vec, cloverterm, x, temp1, temp2)
    μν = 0
    clear_fermion!(temp1)
    clear_fermion!(temp2)
    #println("x ",sum(abs.(x.f)))
    #println("vec ",sum(abs.(vec.f)))
    for μ = 1:3
        for ν = μ+1:4
            μν += 1
            #println("$μν $(sum(abs.(cloverterm.CloverFμν[μν].U)))")

            #println("clovr  ",sum(abs.(cloverterm.CloverFμν[μν].U)))
            mul!(temp1, cloverterm.CloverFμν[μν], x)
            #println("ff1 ",sum(abs.(temp1.f)))
            apply_σμν!(temp2, μ, ν, temp1)
            #println("ff2 ",sum(abs.(temp2.f)))
            #apply_σμν!(temp1,μ,ν,x)
            #println(sum(abs.(cloverterm.CloverFμν[μν].U)))
            #mul!(temp2,cloverterm.CloverFμν[μν],temp1)
            #println("ffd ",sum(abs.(vec.f)))
            add_fermion!(vec, 1, temp2)
            #println("ffc ",sum(abs.(vec.f)))
        end
    end
    #println("ff ",sum(abs.(vec.f)))
    set_wing_fermion!(vec)
    #println("ffvv ",sum(abs.(vec.f)))
end


function cloverterm!(vec, cloverterm, x)
    NT = x.NT
    NZ = x.NZ
    NY = x.NY
    NX = x.NX
    NC = x.NC
    CloverFμν = cloverterm.CloverFμν



    i = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    i += 1
                    for k1 = 1:NC
                        for k2 = 1:NC

                            c1 = x[k2, ix, iy, iz, it, 1]
                            c2 = x[k2, ix, iy, iz, it, 2]
                            c3 = x[k2, ix, iy, iz, it, 3]
                            c4 = x[k2, ix, iy, iz, it, 4]

                            vec[k1, ix, iy, iz, it, 1] += CloverFμν[1][k1, k2, ix, iy, iz, it] * (-c1) +
                                                          +CloverFμν[2][k1, k2, ix, iy, iz, it] * (-im * c2) +
                                                          +CloverFμν[3][k1, k2, ix, iy, iz, it] * (-c2) +
                                                          +CloverFμν[4][k1, k2, ix, iy, iz, it] * (-c2) +
                                                          +CloverFμν[5][k1, k2, ix, iy, iz, it] * (im * c2) +
                                                          +CloverFμν[6][k1, k2, ix, iy, iz, it] * (-c1)



                            vec[k1, ix, iy, iz, it, 2] += CloverFμν[1][k1, k2, ix, iy, iz, it] * (c2) +
                                                          +CloverFμν[2][k1, k2, ix, iy, iz, it] * (im * c1) +
                                                          +CloverFμν[3][k1, k2, ix, iy, iz, it] * (-c1) +
                                                          +CloverFμν[4][k1, k2, ix, iy, iz, it] * (-c1) +
                                                          +CloverFμν[5][k1, k2, ix, iy, iz, it] * (-im * c1) +
                                                          +CloverFμν[6][k1, k2, ix, iy, iz, it] * (c2)

                            vec[k1, ix, iy, iz, it, 3] += CloverFμν[1][k1, k2, ix, iy, iz, it] * (-c3) +
                                                          +CloverFμν[2][k1, k2, ix, iy, iz, it] * (-im * c4) +
                                                          +CloverFμν[3][k1, k2, ix, iy, iz, it] * (c4) +
                                                          +CloverFμν[4][k1, k2, ix, iy, iz, it] * (-c4) +
                                                          +CloverFμν[5][k1, k2, ix, iy, iz, it] * (-im * c4) +
                                                          +CloverFμν[6][k1, k2, ix, iy, iz, it] * (c3)

                            vec[k1, ix, iy, iz, it, 4] += CloverFμν[1][k1, k2, ix, iy, iz, it] * (c4) +
                                                          +CloverFμν[2][k1, k2, ix, iy, iz, it] * (im * c3) +
                                                          +CloverFμν[3][k1, k2, ix, iy, iz, it] * (c3) +
                                                          +CloverFμν[4][k1, k2, ix, iy, iz, it] * (-c3) +
                                                          +CloverFμν[5][k1, k2, ix, iy, iz, it] * (im * c3) +
                                                          +CloverFμν[6][k1, k2, ix, iy, iz, it] * (-c4)


                        end
                    end
                end
            end

        end
    end

    #println("vec = ",vec*vec)

end




struct WilsonClover_misc
    clover_coefficient::Float64
    internal_flags::Array{Bool,1}
    inn_table::Array{Int64,3}
    _ftmp_vectors::Array{Array{ComplexF64,3},1}
    _is1::Array{Int64,1}
    _is2::Array{Int64,1}

    function WilsonClover(
        U::Array{<:AbstractGaugefields{NC,Dim},1},
        x,
        clover_coefficient,
    ) where {NC,Dim}
        _, _, NN... = size(U[1])
        NV = prod(NN)
        inn_table = zeros(Int64, NV, 4, 2)
        internal_flags = zeros(Bool, 2)
        _ftmp_vectors = Array{Array{ComplexF64,3},1}(undef, 6)
        for i = 1:6
            _ftmp_vectors[i] = zeros(ComplexF64, NC, NV, 4)
        end

        _is1 = zeros(Int64, NV)
        _is2 = zeros(Int64, NV)

        return new(clover_coefficient, internal_flags, inn_table, _ftmp_vectors, _is1, _is2)
    end
end
