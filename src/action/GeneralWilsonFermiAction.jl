struct GeneralWilsonFermiAction{Dim,Dirac,fermion,gauge} <: FermiAction{Dim,Dirac,fermion,gauge}
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
    _temporary_fermionfields::Vector{fermion}
    _temporary_gaugefields::Vector{gauge}



    function GeneralWilsonFermiAction(D::Dirac_operator{Dim},hascovnet,covneuralnet,parameters_action) where Dim
        num = 6

        x = D._temporary_fermi[1]
        xtype = typeof(x)
        _temporary_fermionfields = Array{xtype,1}(undef,num)
        for i=1:num
            _temporary_fermionfields[i] = similar(x)
        end

        Utemp = D.U[1]
        Utype = typeof(Utemp)
        numU = 2
        _temporary_gaugefields = Array{Utype,1}(undef,numU)
        for i=1:numU
            _temporary_gaugefields[i] = similar(Utemp)
        end

        return new{Dim,typeof(D),xtype,Utype}(hascovnet,
                        covneuralnet,D,
                        _temporary_fermionfields,
                        _temporary_gaugefields)

    end
end

function evaluate_FermiAction(fermi_action::GeneralWilsonFermiAction{Dim,Dirac,fermion,gauge},U,ϕ::AbstractFermionfields) where {Dim,Dirac,fermion,gauge} 
    W = fermi_action.diracoperator(U)
    η = fermi_action._temporary_fermionfields[1]
    solve_DinvX!(η,W',ϕ)
    Sf = dot(η,η)
    return real(Sf)
end

function gauss_sampling_in_action!(η::AbstractFermionfields,U,fermi_action::GeneralWilsonFermiAction{Dim,Dirac,fermion,gauge} ) where {Dim,Dirac,fermion,gauge} 
    #gauss_distribution_fermion!(η)
    gauss_distribution_fermion!(η,rand)
end


function sample_pseudofermions!(ϕ::AbstractFermionfields,U,fermi_action::GeneralWilsonFermiAction{Dim,Dirac,fermion,gauge} ,ξ::AbstractFermionfields) where {Dim,Dirac,fermion,gauge} 
    W = fermi_action.diracoperator(U)
    mul!(ϕ,W',ξ)
    set_wing_fermion!(ϕ)
end