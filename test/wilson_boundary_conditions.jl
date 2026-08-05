function temporal_slice(field, it)
    return [
        field[ic, ix, iy, iz, it, ialpha] for
        ialpha = 1:field.NG, iz = 1:field.NZ, iy = 1:field.NY,
        ix = 1:field.NX, ic = 1:field.NC
    ]
end

@testset "non-wing Wilson temporal boundary condition" begin
    NX, NY, NZ, NT = 2, 2, 2, 3
    NC = 3
    U = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT; condition="cold")
    source = Initialize_pseudofermion_fields(U[1], "Wilson"; nowing=true)
    clear_fermion!(source)
    setindex_global!(source, 1, 1, 1, 1, 1, 1, 1)

    base_parameters = Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.1,
        "r" => 1.0,
        "faster version" => false,
        "verbose_level" => 0,
    )
    periodic_parameters = copy(base_parameters)
    periodic_parameters["boundarycondition"] = [1, 1, 1, 1]
    antiperiodic_parameters = copy(base_parameters)
    antiperiodic_parameters["boundarycondition"] = [1, 1, 1, -1]

    periodic_operator = Dirac_operator(U, source, periodic_parameters)
    antiperiodic_operator = Dirac_operator(U, source, antiperiodic_parameters)

    periodic_result = similar(source)
    antiperiodic_result = similar(source)
    clear_fermion!(periodic_result)
    clear_fermion!(antiperiodic_result)
    mul!(periodic_result, periodic_operator, source)
    mul!(antiperiodic_result, antiperiodic_operator, source)

    periodic_boundary_slice = temporal_slice(periodic_result, NT)
    antiperiodic_boundary_slice = temporal_slice(antiperiodic_result, NT)
    @test norm(periodic_boundary_slice) > 0
    @test periodic_boundary_slice ≈ -antiperiodic_boundary_slice

    periodic_adjoint_result = similar(source)
    antiperiodic_adjoint_result = similar(source)
    clear_fermion!(periodic_adjoint_result)
    clear_fermion!(antiperiodic_adjoint_result)
    mul!(periodic_adjoint_result, periodic_operator', source)
    mul!(antiperiodic_adjoint_result, antiperiodic_operator', source)

    periodic_adjoint_boundary_slice = temporal_slice(periodic_adjoint_result, NT)
    antiperiodic_adjoint_boundary_slice =
        temporal_slice(antiperiodic_adjoint_result, NT)
    @test norm(periodic_adjoint_boundary_slice) > 0
    @test periodic_adjoint_boundary_slice ≈
          -antiperiodic_adjoint_boundary_slice
end

function check_lower_z_evenodd_halo!(field, target_is_even)
    clear_fermion!(field)
    for ialpha = 1:field.NG, it = 1:field.NT, iy = 1:field.NY,
        ix = 1:field.NX
        field[1, ix, iy, field.NZ, it, ialpha] = 1
    end

    set_wing_fermion!(field, [1, 1, 1, 1], target_is_even)
    for ialpha = 1:field.NG, it = 1:field.NT, iy = 1:field.NY,
        ix = 1:field.NX
        source_is_even = iseven((ix + iy + field.NZ + it) % 2)
        expected = source_is_even == target_is_even ? 1 : 0
        @test field[1, ix, iy, 0, it, ialpha] == expected
    end
end

@testset "non-cubic even-odd lower-z halo parity" begin
    NX, NY, NZ, NT = 2, 2, 3, 2
    fundamental = WilsonFermion_4D_wing{3}(NX, NY, NZ, NT)
    check_lower_z_evenodd_halo!(fundamental, true)
    check_lower_z_evenodd_halo!(fundamental, false)

    adjoint = initialize_Adjoint_fermion(fundamental)
    check_lower_z_evenodd_halo!(adjoint, true)
    check_lower_z_evenodd_halo!(adjoint, false)
end
