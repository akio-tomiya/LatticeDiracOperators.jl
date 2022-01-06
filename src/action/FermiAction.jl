struct FermiAction{Dim,Dirac}
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
end

function FermiAction(D::Dirac_operator{Dim};covneuralnet = nothing) where {NC,Dim}
    diractype = typeof(D)
    if covneuralnet ==  nothing
        hascovnet = false
    else
        hascovnet = true
    end
    return FermiAction{Dim,diractype}(hascovnet,covneuralnet,D)
end