using ParallelDataTransfer
using Distributed
length(workers()) == 1 && @show addprocs(3)
# TODO: scaled jacobian penalty
# TODO: sample slower to move eigvals from 1
# TODO: System instead of DiffSystem
# DID set fixed normalizer updated every 50 steps

# @everywhere using Revise
cd(@__DIR__)
@isdefined(simulate_pendcart) || (@everywhere include(joinpath("/home/fredrikb/.julia/v0.6/GuidedPolicySearch/src/system_pendcart.jl")))
@everywhere using Main.PendCart
@everywhere using Parameters, JacProp, OrdinaryDiffEq, LTVModels, LTVModelsBase, LinearAlgebra, Statistics, Random
@everywhere using Flux: params, jacobian
@everywhere using Flux.Optimise: Param, optimiser, expdecay
@everywhere begin
    BLAS.set_num_threads(1)
    @with_kw struct PendcartSys
        N  = 1000
        nu  = 1
        nx = 4
        h = 0.02
        σ0 = 0
        sind = 1:nx
        uind = nx+1:(nx+nu)
        s1ind = (nx+nu+1):(nx+nu+nx)
    end

    let h = 0.01, g = 9.82, l = 0.35, d = 1
        global generate_data, true_jacobian
        function fsys(xd,x,u)
            xd[1] = x[2]
            xd[2] = -g/l * sin(x[1]) + u[]/l * cos(x[1]) - d*x[2]
            xd[3] = x[4]
            xd[4] = u[]
            xd
        end

        function generate_data(sys::PendcartSys, seed, validation=false; ufun=u->filt(ones(200),[200], 20u')')
            @unpack N, nu, h, σ0 = sys
            srand(seed)
            done = false
            local x,u

            fs = logspace(-3,-1.5,50)[randperm(50)[1:5]]
            # u    = ufun(randn(nu,N+2))
            u = sum(f->sin.(2π*f.* (1:N+2) .+ 2π*rand()), fs)'
            t    = 0:h:N*h
            x0   = [0.8π,0,0,0]
            prob = OrdinaryDiffEq.ODEProblem((xd,x,p,t)->fsys(xd,x, u[:,floor(Int,t/h)+1]),x0,(t[[1,end]]...))
            sol  = solve(prob,Tsit5(),reltol=1e-4,abstol=1e-4)
            x    = hcat(sol(t)...)

            u = u[:,1:N]
            validation || (x .+= σ0 * randn(size(x)))
            @assert all(isfinite, x)
            @assert all(isfinite, u)
            x,u
        end

        function true_jacobianc(sys::PendcartSys, x::AbstractVector, u::AbstractVector)
            u[isnan.(u)] = 0
            nx = size(x,1)
            nu = size(u,1)
            fx = Array{Float64}(nx,nx)
            fu = Array{Float64}(nx)
            fx[:,:] = [0 1 0 0;
            -g/l*cos(x[1])-u/l*sin(x[1]) -d 0 0;
            0 0 0 1;
            0 0 0 0]
            fu[:,:] = [0, cos(x[1])/l, 0, 1]
            [fx  fu]
        end

        function true_jacobian(sys::PendcartSys, x::AbstractVector, u::AbstractVector)
            J = true_jacobianc(sys,x,u)
            ABd = expm([J*sys.h; zeros(nu, nx + nu)])[1:4,:]# ZoH sampling
        end
    end

    function callbacker(epoch, loss,d,trace,model,mt)
        i = length(trace) + epoch - 1
        function ()
            l = sum(d->Flux.data(loss(d...)),d)
            increment!(trace,epoch,l)
            if i % 500 == 0
                @printf("Loss: %.4f", l)
                # jacplot(model, trajs[1], true_jacobian, ds=5,show=true,reuse=true)
            end
        end
    end


    num_params = 40
    wdecay     = 0.1
    stepsize   = 0.01
    const sys  = PendcartSys(N=200, h=0.01, σ0 = 0.3)
    true_jacobian(x,u)  = true_jacobian(sys,x,u)
    true_jacobianc(x,u) = true_jacobianc(sys,x,u)
    nu         = sys.nu
    nx         = sys.nx

end


## Generate validation data
function valdata()
    vx,vu,vy = Vector{Float64}[],Vector{Float64}[],Vector{Float64}[]
    for i = 20:60
        x,u = generate_data(sys,i, true)
        for j in 10:5:(sys.N-1)
            push!(vx, x[:,j])
            push!(vy, x[:,j+1])
            push!(vu, u[:,j])

        end
    end
    hcat(vx...),hcat(vu...),hcat(vy...)
end
vx,vu,vy = valdata()
vt = Trajectory(vx,vu,vy)

trajs = [Trajectory(generate_data(sys, i)...) for i = 1:3]

sendto(workers(), trajs=trajs, vt=vt)

## Monte-Carlo evaluation
num_montecarlo = 10
it = 1
res = map(1:num_montecarlo) do it
    r2 = @spawn begin
        λ = exp10.(range(-3, stop=1, length=1000))[rand(1:1000)]
        srand(it)
        cb(model) = ()#(jacplot(model, trajs[1], true_jacobian, ds=5,show=true,reuse=true);gui())
        model     = JacProp.ADDiffSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
        opt       = LTVModels.ADAMOptimizer.(model.w, α = stepsize)
        trainerad = ADModelTrainer(;model=model, opt=opt, λ=λ, testdata = vt)
        for i = 1:1
            trainerad(trajs[i], epochs=0)
        end
        train!(trainerad, epochs=400,cb=cb)
        trainerad
    end
    r1 = @spawn begin
        wdecay = exp10.(range(-3, stop=1, length=1000))[rand(1:1000)]
        srand(it)
        # models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
        models     = [DiffSystem(nx,nu,num_params, tanh)]
        opts       = [[ADAM.(params.(models), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
        trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = wdecay)
        for i = 1:2
            trainer(trajs[i], epochs=0, jacprop=0, useprior=false)
            # traceplot(trainer)
        end
        trainer(epochs=400)
        trainer
    end

    println("Done with montecarlo run $it")
    r1,r2
end


res = [(fetch.(rs)...) for rs in res]
resdiff = getindex.(res,1)
resad = getindex.(res,2)

# serialize("results", res)
# res = deserialize("results")
using Plots
for i = 1:10
    rd = resdiff[i]; rad = resad[i]
    eigvalplot(rd.models, vt, true_jacobian;layout=(2,1),subplot=1,cont=false,title="Standard wd = $(rd.σdivider)", ylims=(-0.2,0.2))
    eigvalplot!(rad.model, vt, true_jacobian;subplot=2,cont=false,title="AD Jacprop λ = $(rad.λ)", link = :both, ylims=(-0.2,0.2)); display(current())
end
plot(resad[1].trace.iterations,[res.trace.values for res in resad], c=:blue)
plot!(resad[1].trace.iterations,[res.tracev.values for res in resad], c=:orange)
plot!(resdiff[1].trace.iterations,[res.trace.values for res in resdiff], c=:red, xscale=:log10, yscale=:log10, legend=false)



nr = length(res[1])÷2
labelvec = ["f" "g"]
infostring = @sprintf("Num hidden: %d, sigma: %2.2f, Montecarlo: %d", num_params, sys.σ0, num_montecarlo)
simvals = [Trajectory(generate_data(sys, i, true)...) for i = 4:6]
pred  = [JacProp.eval_pred.(resdiff, vt) JacProp.eval_pred.(resad, vt)]
sim   = vcat([[JacProp.eval_sim.(resdiff, simval) JacProp.eval_sim.(resad, simval)] for simval in simvals]...)
jac   = [JacProp.eval_jac.(resdiff, vt, true_jacobian,3) JacProp.eval_jac.(resad, vt, true_jacobian,3)]

##
using StatPlots
vio1 = boxplot(pred, lab=["Standard" "Jacobian propagation"], ylabel="Prediction RMS", reuse=false, yscale=:identity)
vio2 = boxplot(sim, lab=["Standard" "Jacobian propagation"], ylabel="Simulation RMS", yscale=:identity)
vio3 = boxplot(jac, lab=["Standard" "Jacobian propagation"], ylabel="Jacobian Error", yscale=:identity)
plot(vio1,vio2,vio3,title=infostring)
##
# savefig2("/local/home/fredrikb/papers/nn_prior/figs/valerr.tex")

plot(trajs[1])
predsimplot(resdiff[1].models, simvals[2], title="Standard")
predsimplot(resad[1].model, simvals[2], reuse=false, title="Jacprop")
# simulate(resad[1].model, trajs[1])' |> plot
##
i = 1
j = 1
plot(simvals[i], control=false, layout=4, lab="Traj")
simplot!(resdiff[j].models, simvals[i], lab="Standard")
simplot!(resad[j].model, simvals[i], lab="AD JAcprop")
##


jacplot(resdiff[1].models, simvals[1], true_jacobian, ds=2, reuse=false)
jacplot!(resad[1].model, simvals[1], ds=2)









num_montecarlo = 40
it = 1
res = pmap(1:num_montecarlo) do it
    λ = logspace(-2,2,100)[rand(1:100)]
    num_params = rand(10:100)
    srand(it)
    model     = JacProp.ADDiffSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
    opt       = LTVModels.ADAMOptimizer(model.w, α = stepsize)
    trainerad = ADModelTrainer(;model=model, opt=opt, λ=λ, testdata = vt)
    for i = 1:2
        trainerad(trajs[i], epochs=0)
    end
    trainerad(epochs=2000)
    println(it)
    trainerad
end
# serialize("res", res)

res2 = map(res) do r
    JacProp.eval_jac.(r, vt, true_jacobian,3)
end
im = indmin(res2)
trainerad = res[im]
s = map(res) do r
    r.model.sizes[1][1]
end
λ = map(res) do r
    r.λ
end

jacplot(trainerad.model, trajs[1], true_jacobian, ds=2, reuse=false)

scatter([s λ], res2, layout=2)
plot!(xscale=:log10, yscale=:log10, subplot=2)




# Generate ss vs tf param plot ==========================================
# using ControlSystems
# t = Trajectory(generate_data(sys, 1)...)
#
# function jacs(t)
#     N = length(t)
#     J = map(1:N) do evalpoint
#         Jd = true_jacobian(t.x[:,evalpoint],t.u[:,evalpoint])
#         Jc = true_jacobianc(t.x[:,evalpoint],t.u[:,evalpoint])
#         Gss = c2d(ss(Jc[1:end,1:nx], Jc[1:end,nx+1:end],[1 0 0 0;0 0 1 0],0), sys.h)[1]
#         Gtf = tf(Gss)
#         # @show numpoly(Gtf)
#         hcat(getfield.(denpoly(Gtf),:a)...)
#         Jd[:], [hcat(getfield.(numpoly(Gtf),:a)...)' hcat(getfield.(denpoly(Gtf),:a)...)']
#     end
#     JacProp.smartcat2(getindex.(J, 1)), JacProp.smartcat3(getindex.(J, 2))
# end
# Gss, Gtf = jacs(t)
# Gtf = reshape(Gtf, 2*9,100)
#
# plot((Gss./maximum(abs.(Gss),2))', layout=2, subplot=1, c=:blue, legend=false)
# plot!((Gtf./maximum(abs.(Gtf),2))', subplot=2, c=:blue, legend=false)
