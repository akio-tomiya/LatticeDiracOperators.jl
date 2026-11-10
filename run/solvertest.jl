# Solve D y = b for the Wilson and Shamir domain-wall operators on a cold SU(3)
# configuration, check D^{-1}(D x) = x, and time repeated solves.
# Serial: julia -t 4 --project=. run/solvertest.jl
# MPI (requires MPI.jl in the environment): mpiexec -n 2 julia --project=. run/solvertest.jl 1 1 1 2

const PROCESS_GRID = isempty(ARGS) ? (1, 1, 1, 1) : Tuple(parse.(Int, ARGS[1:4]))
if prod(PROCESS_GRID) > 1
    using MPI
end

import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

const LATTICE = (8, 8, 8, 16) # original samples: (16, 16, 16, 32)
const NC = 3
const NUM_SOLVES = 10

relative_difference(y, x) = sqrt(real(dot(y, y)) / real(dot(x, x)))

function check_solver(name, U, x, parameters; root=true)
    D = Dirac_operator(U, x, parameters)
    b = similar(x)
    y = similar(x)

    Random.seed!(123)
    gauss_distribution_fermion!(x)
    mul!(b, D, x)
    solve_DinvX!(y, D, b)
    axpby!(-1, x, 1, y) # y <- D^{-1}(D x) - x
    deviation = relative_difference(y, x)

    elapsed = 0.0
    for _ in 1:NUM_SOLVES
        gauss_distribution_fermion!(b)
        elapsed += @elapsed solve_DinvX!(y, D, b)
    end
    if root
        println(name, ": |D^{-1}(D x) - x| / |x| = ", deviation,
            ", average solve time = ", elapsed / NUM_SOLVES, " s")
    end
    return deviation
end

function main()
    U = gauge_configuration(LATTICE; colors=NC, halo=1, start=:cold, process_grid=PROCESS_GRID)
    root = get_myrank(U) == 0
    root && println("lattice = $LATTICE, process grid = $PROCESS_GRID")

    x = Initialize_pseudofermion_fields(U[1], "Wilson")
    wilson = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.141139,
        "eps_CG" => 1e-16,
        "verbose_level" => 0,
    )
    check_solver("Wilson (κ = 0.141139)", U, x, wilson; root)

    L5 = 4
    x5 = Initialize_pseudofermion_fields(U[1], "Domainwall"; L5)
    domainwall = Dict{String,Any}(
        "Dirac_operator" => "Domainwall",
        "mass" => 0.1,
        "L5" => L5,
        "eps_CG" => 1e-16,
        "verbose_level" => 0,
    )
    check_solver("Shamir domain wall (L5 = $L5, mass = 0.1)", U, x5, domainwall; root)
end

main()
