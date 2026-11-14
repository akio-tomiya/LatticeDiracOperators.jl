module WilsonForceFactorTests
using JACC
JACC.@init_backend
using Gaugefields, LatticeDiracOperators, LatticeMatrices, LinearAlgebra, Random, Test
const DO = LatticeDiracOperators.Dirac_operators
const distributed = get(ENV, "CHECK_MPI", "0") == "1"
if distributed
    using MPI
    MPI.Initialized() || MPI.Init()
end
const nr = distributed ? MPI.Comm_size(MPI.COMM_WORLD) : 1
const rank = distributed ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
reduce_sum(x) = distributed ? MPI.Allreduce(x, +, MPI.COMM_WORLD) : x
const dims = (4,2,2,2)
const pes = (nr,1,1,1)
function put_source!(x, values, legacy)
    if legacy
        for I in CartesianIndices(values)
            c,s,a,b,d,t = Tuple(I)
            setindex_global!(x, values[I], c,a,b,d,t,s)
        end
    else
        v = LatticeMatrix(values,4,pes; nw=1,phases=x.f.phases,comm0=x.f.comm)
        substitute!(x.f,v)
    end
    set_wing_fermion!(x)
end
function site_matrix(u, legacy)
    legacy && return copy(u.U[:,:,1,1,1,1])
    a = Array(u.U.A); n=u.U.nw+1
    return copy(a[:,:,n,n,n,n])
end
function put_matrix!(u, v, legacy)
    if legacy
        u.U[:,:,1,1,1,1] .= v
    else
        a=Array(u.U.A); n=u.U.nw+1
        if rank==0; a[:,:,n,n,n,n] .= v; end
        copyto!(u.U.A,a); mark_halo_dirty!(u.U)
    end
    set_wing_U!(u)
end
function force_check(kind, factor)
    legacy=kind in (:legacy,:faster)
    U = legacy ? Initialize_4DGaugefields(3,0,dims...;condition="cold",verbose_level=0) :
        Initialize_Gaugefields(3,1,dims...;condition="cold",isMPILattice=true,PEs=pes,verbose_level=0)
    # Deterministic nontrivial links, identical global field on every backend.
    rng=MersenneTwister(461)
    for mu in 1:4
        values=zeros(ComplexF64,3,3,dims...)
        for site in CartesianIndices(dims)
            a=randn(rng,ComplexF64,3,3); h=(a-a')/2; h-=tr(h)*I/3
            values[:,:,Tuple(site)...] .= exp(0.2h)
        end
        if legacy
            U[mu].U .= values
        else
            substitute!(U[mu].U,LatticeMatrix(values,4,pes;nw=1,comm0=U[mu].U.comm))
        end
    end
    set_wing_U!(U)
    phi=Initialize_pseudofermion_fields(U[1],"Wilson";nowing=true)
    put_source!(phi,randn(MersenneTwister(462),ComplexF64,3,4,dims...),legacy)
    p=Dict{String,Any}("Dirac_operator"=>kind==:clover ? "WilsonClover" : "Wilson",
        "κ"=>0.08,"cSW"=>1.17,"factor"=>factor,"faster version"=>kind==:faster,
        "method_CG"=>"cg","eps_CG"=>1e-24,"MaxCGstep"=>2000,"verbose_level"=>0)
    W=Dirac_operator(U,phi,p); action=FermiAction(W,Dict("Nf"=>2))
    effective_factor=hasproperty(W,:factor) ? W.factor : 1.0
    F=[similar(u) for u in U]
    calc_UdSfdU!(F,action,U,phi)
    if get(ENV,"CHECK_MOMENTUM","1")=="1"
        momentum=initialize_TA_Gaugefields(U); projected=initialize_TA_Gaugefields(U)
        clear_U!(momentum); clear_U!(projected)
        DO.calc_p_UdSfdU!(momentum,action,U,phi)
        maxerr=0.0
        for mu in 1:4
            Gaugefields.Traceless_antihermitian_add!(projected[mu],1.0,F[mu])
            a=legacy ? momentum[mu].a : Array(momentum[mu].a.A)
            b=legacy ? projected[mu].a : Array(projected[mu].a.A)
            maxerr=max(maxerr,maximum(abs.(a-b)))
        end
        rank==0 && println("MOMENTUM kind=$kind factor=$factor max_projected_force_error=$maxerr")
        @test maxerr<1e-10
    end
    H=Matrix(im*Diagonal([1.,-1.,0.])); H/=sqrt(2)
    raw=site_matrix(F[1],legacy)
    analytic=reduce_sum(rank==0 ? -2real(tr(raw*H)) : 0.)
    baseline=site_matrix(U[1],legacy)
    pert=similar(U)
    for eps in (1e-4,1e-5)
        substitute_U!(pert,U); put_matrix!(pert[1],exp(eps*H)*baseline,legacy)
        plus=evaluate_FermiAction(action,pert,phi)
        substitute_U!(pert,U); put_matrix!(pert[1],exp(-eps*H)*baseline,legacy)
        minus=evaluate_FermiAction(action,pert,phi)
        fd=(plus-minus)/(2eps)
        rank==0 && println("FORCE kind=$kind factor=$factor effective_factor=$effective_factor eps=$eps analytic=$analytic fd=$fd ratio=$(analytic/fd) relative_error=$(abs(analytic-fd)/abs(fd)) operator=$(nameof(typeof(W)))")
        @test isapprox(analytic,fd;rtol=2e-5,atol=3e-6)
    end
end
@testset "Issue 13 force finite differences" begin
    kinds=get(ENV,"STANDARD_ONLY","0")=="1" ? (:standard,:clover) : (:legacy,:faster,:standard,:clover)
    if get(ENV,"TOLERANCE_ONLY","0")!="1"
        for kind in kinds, factor in (1.,2.,0.5,-1.); force_check(kind,factor); end
    end
end
@testset "Issue 13 staggered tolerance" begin
    for legacy in (get(ENV,"STANDARD_ONLY","0")=="1" ? (false,) : (true,false))
        U=Initialize_Gaugefields(3,1,dims...;condition="cold",isMPILattice=!legacy,PEs=pes,verbose_level=0)
        x=Initialize_pseudofermion_fields(U[1],"staggered")
        for params in (Dict("eps_CG"=>1e-8),Dict("eps"=>1e-7),Dict("eps_CG"=>1e-8,"eps"=>1e-7))
            p=merge(Dict{String,Any}("Dirac_operator"=>"staggered","mass"=>0.2,"verbose_level"=>0),params)
            D=Dirac_operator(U,x,p)
            rank==0 && println("STAGGERED legacy=$legacy params=$params actual=$(D.eps_CG) gauge=$(typeof(U[1]))")
            @test D.eps_CG == get(params,"eps_CG",get(params,"eps",1e-19))
        end
    end
end

end
