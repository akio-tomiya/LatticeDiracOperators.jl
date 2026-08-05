NX = 2
NY = 2
NZ = 2
NT = 2
NC = 3

U_diagnostics =
    Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT; condition="cold")
source_diagnostics =
    Initialize_pseudofermion_fields(U_diagnostics[1], "Wilson")
clear_fermion!(source_diagnostics)
setindex_global!(source_diagnostics, 1, 1, 1, 1, 1, 1, 1)

parameters_diagnostics = Dict{String,Any}(
    "Dirac_operator" => "Wilson",
    "κ" => 0.10,
    "r" => 1.0,
    "boundarycondition" => [1, 1, 1, -1],
    "faster version" => false,
    "method_CG" => "bicgstab",
    "eps_CG" => 1.0e-20,
    "MaxCGstep" => 10_000,
    "verbose_level" => 0,
)
operator_diagnostics = Dirac_operator(
    U_diagnostics,
    source_diagnostics,
    parameters_diagnostics,
)
solution_diagnostics = similar(source_diagnostics)
clear_fermion!(solution_diagnostics)
diagnostics = solve_DinvX!(
    solution_diagnostics,
    operator_diagnostics,
    source_diagnostics,
)

@test diagnostics isa SolverDiagnostics
@test diagnostics.method === :bicgstab
@test 0 < diagnostics.iterations < diagnostics.maximum_iterations
@test diagnostics.recursive_residual_squared <
      diagnostics.target_residual_squared
@test diagnostics.target_residual_squared == 1.0e-20
@test diagnostics.restart_count >= 0
@test diagnostics.convergence_branch in
      (:intermediate_residual, :updated_residual)

action_diagnostics = similar(source_diagnostics)
clear_fermion!(action_diagnostics)
mul!(action_diagnostics, operator_diagnostics, solution_diagnostics)
true_relative_residual = norm(
    convert_to_normalvector(action_diagnostics) -
    convert_to_normalvector(source_diagnostics),
) / norm(convert_to_normalvector(source_diagnostics))
@test true_relative_residual < 1.0e-10

# A converged initial guess and a failed solve must both release every
# preallocated temporary field.
zero_source = similar(source_diagnostics)
clear_fermion!(zero_source)
zero_solution = similar(source_diagnostics)
clear_fermion!(zero_solution)
for _ in 1:3
    zero_diagnostics =
        solve_DinvX!(zero_solution, operator_diagnostics, zero_source)
    @test zero_diagnostics.iterations == 0
    @test zero_diagnostics.recursive_residual_squared == 0
    @test zero_diagnostics.restart_count == 0
    @test zero_diagnostics.convergence_branch === :initial_residual
end

failed_solution = similar(source_diagnostics)
clear_fermion!(failed_solution)
@test_throws ErrorException bicgstab(
    failed_solution,
    operator_diagnostics,
    source_diagnostics;
    eps=1.0e-40,
    maxsteps=1,
    verbose=Verbose_print(0),
)

clear_fermion!(solution_diagnostics)
diagnostics_after_failure = solve_DinvX!(
    solution_diagnostics,
    operator_diagnostics,
    source_diagnostics,
)
@test diagnostics_after_failure isa SolverDiagnostics

parameters_bicg = copy(parameters_diagnostics)
parameters_bicg["method_CG"] = "bicg"
operator_bicg =
    Dirac_operator(U_diagnostics, source_diagnostics, parameters_bicg)
solution_bicg = similar(source_diagnostics)
clear_fermion!(solution_bicg)
diagnostics_bicg =
    solve_DinvX!(solution_bicg, operator_bicg, source_diagnostics)
@test diagnostics_bicg isa SolverDiagnostics
@test diagnostics_bicg.method === :bicg
@test 0 < diagnostics_bicg.iterations < diagnostics_bicg.maximum_iterations
@test diagnostics_bicg.recursive_residual_squared <
      diagnostics_bicg.target_residual_squared
@test diagnostics_bicg.restart_count == 0
@test diagnostics_bicg.convergence_branch === :updated_residual
action_bicg = similar(source_diagnostics)
clear_fermion!(action_bicg)
mul!(action_bicg, operator_bicg, solution_bicg)
true_relative_residual_bicg = norm(
    convert_to_normalvector(action_bicg) -
    convert_to_normalvector(source_diagnostics),
) / norm(convert_to_normalvector(source_diagnostics))
@test true_relative_residual_bicg < 1.0e-10

failed_bicg_solution = similar(source_diagnostics)
clear_fermion!(failed_bicg_solution)
@test_throws ErrorException bicg(
    failed_bicg_solution,
    operator_bicg,
    source_diagnostics;
    eps=1.0e-40,
    maxsteps=1,
    verbose=Verbose_print(0),
)
clear_fermion!(solution_bicg)
@test solve_DinvX!(
    solution_bicg,
    operator_bicg,
    source_diagnostics,
) isa SolverDiagnostics

parameters_preconditioned = copy(parameters_diagnostics)
parameters_preconditioned["method_CG"] = "preconditiond_bicgstab"
operator_preconditioned = Dirac_operator(
    U_diagnostics,
    source_diagnostics,
    parameters_preconditioned,
)
solution_preconditioned = similar(source_diagnostics)
clear_fermion!(solution_preconditioned)
diagnostics_preconditioned = solve_DinvX!(
    solution_preconditioned,
    operator_preconditioned,
    source_diagnostics,
)
@test diagnostics_preconditioned isa SolverDiagnostics
@test diagnostics_preconditioned.method === :preconditiond_bicgstab
@test 0 < diagnostics_preconditioned.iterations <
          diagnostics_preconditioned.maximum_iterations
@test diagnostics_preconditioned.recursive_residual_squared <
      diagnostics_preconditioned.target_residual_squared
@test diagnostics_preconditioned.restart_count >= 0
@test diagnostics_preconditioned.convergence_branch in
      (:intermediate_residual, :updated_residual)
action_preconditioned = similar(source_diagnostics)
clear_fermion!(action_preconditioned)
mul!(
    action_preconditioned,
    operator_preconditioned,
    solution_preconditioned,
)
true_relative_residual_preconditioned = norm(
    convert_to_normalvector(action_preconditioned) -
    convert_to_normalvector(source_diagnostics),
) / norm(convert_to_normalvector(source_diagnostics))
@test true_relative_residual_preconditioned < 1.0e-10
for _ in 1:3
    clear_fermion!(solution_preconditioned)
    @test solve_DinvX!(
        solution_preconditioned,
        operator_preconditioned,
        source_diagnostics,
    ) isa SolverDiagnostics
end

@info "solver diagnostic metrics" bicgstab_iterations =
    diagnostics.iterations bicgstab_recursive_residual_squared =
    diagnostics.recursive_residual_squared bicgstab_true_relative_residual =
    true_relative_residual bicg_iterations =
    diagnostics_bicg.iterations bicg_recursive_residual_squared =
    diagnostics_bicg.recursive_residual_squared bicg_true_relative_residual =
    true_relative_residual_bicg preconditioned_iterations =
    diagnostics_preconditioned.iterations preconditioned_recursive_residual_squared =
    diagnostics_preconditioned.recursive_residual_squared preconditioned_true_relative_residual =
    true_relative_residual_preconditioned
