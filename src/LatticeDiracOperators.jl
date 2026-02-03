module LatticeDiracOperators
using Wilsonloop
using Gaugefields
using Requires
using AlgRemez_jll
using MPI
using JACC
using LatticeMatrices

import Gaugefields: add_U!
import Gaugefields: Abstractfields, clear_U!
import Gaugefields:
    println_verbose_level1, println_verbose_level2, println_verbose_level3, Verbose_print

import Gaugefields.Temporalfields_module: get_temp
using PreallocatedArrays
Gaugefields.Temporalfields_module.get_temp(a::PreallocatedArray) = get_block(a)
Gaugefields.Temporalfields_module.get_temp(a::PreallocatedArray, i) = get_block(a, i)


# Write your package code here.
include("./rhmc/AlgRemez.jl")
include("./rhmc/rhmc.jl")
#include("./cgmethods.jl")


import LatticeMatrices: γ1, γ2, γ3, γ4, mul_AshiftB!, mul_shiftAshiftB!,
    Enzyme_derivative!
export γ1, γ2, γ3, γ4
export mul_AshiftB!, mul_shiftAshiftB!
export Enzyme_derivative!



include("Diracoperators.jl")
include("./SakuraiSugiura/SSmodule.jl")


import .Dirac_operators:
    Initialize_pseudofermion_fields,
    Dirac_operator,
    gauss_distribution_fermion!,
    Initialize_WilsonFermion,
    Initialize_4DWilsonFermion,
    DdagD_operator,
    solve_DinvX!,
    FermiAction,
    shift_fermion,
    bicg,
    bicgstab,
    sample_pseudofermions!,
    gauss_sampling_in_action!,
    evaluate_FermiAction,
    calc_UdSfdU,
    calc_UdSfdU!,
    bicgstab,
    gmres,
    #pregmres,
    Z2_distribution_fermion!,
    Wilson_Dirac_operator_evenodd,
    calc_p_UdSfdU!,
    Wilson_GeneralDirac_operator,
    set_wing_fermion!,
    eigensystem,
    eigensystem_old,
    construct_sparsematrix,
    initialize_Adjoint_fermion,
    calc_dSfdU!,
    Wilson_Dirac_operator_faster,
    Dx!,
    Ddagx!,
    setvalue_fermion!,
    setindex_global!,
    uniform_distribution_fermion!,
    γ5D,
    convert_to_normalvector,
    save_fermionfield,
    load_fermionfield!,
    substitute_fermion!,
    apply_F_5D!, apply_δF_5D!, D4x_5D!,
    Z4_distribution_fermi!,
    clear_fermion!,
    add_fermion!,
    dSFdU!,
    GeneralFermion,
    DdagDgeneral,
    GeneralFermionAction

export GeneralFermion, DdagDgeneral
export dSFdU!
export apply_F_5D!, apply_δF_5D!, D4x_5D!, Z4_distribution_fermi!

export substitute_fermion!

export Initialize_pseudofermion_fields,
    Dirac_operator, gauss_distribution_fermion!, cg, bicg
export Initialize_WilsonFermion, Initialize_4DWilsonFermion
export DdagD_operator, solve_DinvX!, FermiAction, GeneralFermionAction
export shift_fermion
export WilsonFermion_4D_wing
export sample_pseudofermions!,
    gauss_sampling_in_action!,
    evaluate_FermiAction,
    calc_UdSfdU,
    calc_UdSfdU!,
    bicgstab,
    calc_p_UdSfdU!
export Wilson_Dirac_operator_evenodd, Wilson_GeneralDirac_operator, set_wing_fermion!
export println_verbose_level1, println_verbose_level2, println_verbose_level3, Verbose_print
export bicg, bicgstab
export eigensystem, calc_dSfdU!, eigensystem_old
export construct_sparsematrix, initialize_Adjoint_fermion, Z2_distribution_fermion!
export Wilson_Dirac_operator_faster
export setvalue_fermion!
export setindex_global!
export uniform_distribution_fermion!, γ5D
export convert_to_normalvector
export save_fermionfield, load_fermionfield!
export clear_fermion!, add_fermion!

end
