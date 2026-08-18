# Domain-wall compatibility entry point.
#
# The standard LatticeMatrices-backed five-dimensional field is loaded by the
# Möbius family entry point after the legacy Möbius abstract types are
# available.  Keep the historical implementation included so existing field
# and function names remain source compatible.
include("./deprecated/DomainwallFermion_legacy.jl")
