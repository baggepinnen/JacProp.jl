cd(@__DIR__)
length(workers()) == 1 && @show addprocs(4)

# @everywhere using Revise
using ParallelDataTransfer
@everywhere include(Pkg.dir("DynamicMovementPrimitives","src","two_link.jl"))
@everywhere using TwoLink, Parameters, JacProp, OrdinaryDiffEq, LTVModels, LTVModelsBase
@everywhere using Flux: params, jacobian
@everywhere using Flux.Optimise: Param, optimiser, expdecay
@everywhere begin
    @with_kw struct TwoLinkSys
        N  = 1000
        nu  = 2
        nx = 4
        h = 0.02
        σ0 = 0
        sind = 1:nx
        uind = nx+1:(nx+nu)
        s1ind = (nx+nu+1):(nx+nu+nx)
    end

    function generate_data(sys::TwoLinkSys, seed, validation=false; ufun=u->filt(ones(100),[100], 10u')')
        @unpack N, nu, nx, h, σ0, sind, uind, s1ind = sys
        srand(seed)
        done = false
        local x,u
        while !done
            u    = ufun(randn(nu,N+2))
            t    = 0:h:N*h
            x0   = [-0.4,0,0,0]
            prob = OrdinaryDiffEq.ODEProblem((x,p,t)->time_derivative(x, u[:,floor(Int,t/h)+1]),x0,(t[[1,end]]...))
            sol  = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8)
            x    = hcat(sol(t)...)
            done = all(abs.(x[1:2,:]) .< 0.9π)
        end
        u = u[:,1:N]
        validation || (x .+= σ0 * randn(size(x)))
        @assert all(isfinite, x)
        @assert all(isfinite, u)
        x,u
    end

    function true_jacobian(sys::TwoLinkSys, x::AbstractVector, u::AbstractVector)
        Jtrue = ReverseDiff.jacobian(x->time_derivative(x, u), x)[:,1:4]
        Jtrue = [Jtrue ReverseDiff.jacobian(u->time_derivative(x, u), u)]
        Jtrue = expm([sys.h*Jtrue;zeros(2,6)])[1:4,:] # Discretize
    end

    function callbacker(epoch, loss,d,trace,model,mt)
        i = length(trace) + epoch - 1
        function ()
            l = sum(d->Flux.data(loss(d...)),d)
            increment!(trace,epoch,l)
            i % 500 == 0 && println(@sprintf("Loss: %.4f", l))
        end
    end


    num_params = 30
    wdecay     = 0
    stepsize   = 0.02
    const sys  = TwoLinkSys(N=200, h=0.02, σ0 = 0.01)
    true_jacobian(x,u) = true_jacobian(sys,x,u)
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
#
# ## Without jacprop
# f2 = @spawnat 2 begin
#     srand(1)
#     models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
#     opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
#     trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
#     for i = 1:3
#         trainer(trajs[i], epochs=20000, jacprop=0, useprior=false)
#         # traceplot(trainer)
#     end
#     trainer
# end
#
#
# ## With jacprop and prior
# f3 = @spawnat 3 begin
#     srand(1)
#     models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
#     opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
#     trainerj  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 1000I, σdivider = 20)
#     for i = 1:3
#         trainerj(trajs[i], epochs=10000, jacprop=1, useprior=true)
#         # traceplot(trainerj)
#     end
#     trainerj
# end
#
# ## With jacprop no prior
# f4 = @spawnat 4 begin
#     srand(1)
#     models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
#     opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
#     trainerjn  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 1000I, σdivider = 200)
#     for i = 1:3
#         trainerjn(trajs[i], epochs=100, jacprop=1, useprior=false)
#         # traceplot(trainerjn)
#     end
#     trainerjn(epochs=10000)
#     trainerjn
# end
# # predsimplot(trainerjn.models, trajs[1], title=["\$q_1\$" "\$\\dotq_1\$" "\$q_2\$" "\$\\dotq_2\$"])
# # predsimplot(trainerjn.models, Trajectory(generate_data(sys, 6)...), title=["\$q_1\$" "\$\\dotq_1\$" "\$q_2\$" "\$\\dotq_2\$"])
#
# pyplot(reuse=false)
# trainer,trainerj,trainerjn = fetch(f2), fetch(f3),fetch(f4)
#
# mutregplot(trainer, vt, true_jacobian, title="Witout jacprop", subplot=1, layout=(2,3), reuse=false, useprior=false)
# mutregplot!(trainerj, vt, true_jacobian, title="With jacprop and prior", subplot=2, link=:y, useprior=true, show=false)
# mutregplot!(trainerjn, vt, true_jacobian, title="With jacprop, no prior", subplot=3, link=:y, useprior=false)
# traceplot!(trainer, subplot=4)
# traceplot!(trainerj, subplot=5, show=false)
# traceplot!(trainerjn, subplot=6)
# gui()
# ##
#
#
# jacplot(trainer.models, vt, true_jacobian, label="Without", c=:red, reuse=false, fillalpha=0.2)
# # jacplot!(KalmanModel(trainer, vt), vt, label="Without", c=:pink)
# jacplot!(trainerj.models, vt, label="With", c=:blue, fillalpha=0.2)
# # jacplot!(KalmanModel(trainerj, vt), vt, label="With", c=:cyan)
# jacplot!(trainerjn.models, vt, label="With no", c=:green, fillalpha=0.2)
# # jacplot!(KalmanModel(trainerjn, vt), vt, label="With no", c=:green)
# gui()
#
#
# ui = display_modeltrainer(trainer, size=(800,600))
# jacplot(trainer.models, trainer.trajs[3], true_jacobian, ds=20)
# @gif for i = 1:length(t)
#     eigvalplot(trainer.models, trainer.trajs[1], true_jacobian, ds=20, onlyat=i)
# end
#
# # TODO: Validate vt in callback

## Monte-Carlo evaluation
num_montecarlo = 2

res = map(1:num_montecarlo) do it
    r1 = @spawn begin
        srand(it)
        models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
        opts       = ADAM.(params.(models), stepsize, decay=0.0005)#;
        trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
        for i = 1:3
            trainer(trajs[i], epochs=0, jacprop=0, useprior=false)
            # traceplot(trainer)
        end
        trainer(epochs=200)
        trainer
    end

    r2 = @spawn begin
        srand(it)
        model     = JacProp.ADDiffSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
        opt       = LTVModels.ADAMOptimizer(model.w, α = 0.2stepsize)
        trainerad = ADModelTrainer(;model=model, opt=opt, λ=10, testdata = vt)
        for i = 1:3
            trainerad(trajs[i], epochs=0)
        end
        trainerad(epochs=150)
        trainerad
    end

    println("Done with montecarlo run $it")
    r1,r2
end
res = [(fetch.(rs)...) for rs in res]

# serializesave("results", res)
# res = deserialize("results")
eigvalplot(res[1][1].models, vt, true_jacobian;  title="Standard", ylims=[-0.1,0.1], xlims=[0.5,1.1], layout=2)
eigvalplot!(res[1][2].model, vt, true_jacobian;  title="AD Jacprop", ylims=[-0.1,0.1], xlims=[0.5,1.1], subplot=2);gui()
plot(res[1][2].trace.iterations,[res[2].trace.values for res in res], c=:blue)
plot!(res[1][2].trace.iterations,[res[2].tracev.values for res in res], c=:orange)
plot!(res[1][1].trace.iterations,[res[1].trace.values for res in res]./4, c=:red, xscale=:log10, yscale=:log10, legend=false)


resdiff = getindex.(res,1)
resad = getindex.(res,2)

nr = length(res[1])÷2
labelvec = ["f" "g"]
infostring = @sprintf("Num hidden: %d, sigma: %2.2f, Montecarlo: %d", num_params, sys.σ0, num_montecarlo)
simvals = [Trajectory(generate_data(sys, i)...) for i = 4:6]
pred  = [JacProp.eval_pred.(resdiff, vt) JacProp.eval_pred.(resad, vt)]
sim   = vcat([[JacProp.eval_sim.(resdiff, simval) JacProp.eval_sim.(resad, simval)] for simval in simvals]...)
jac   = [JacProp.eval_jac.(resdiff, vt, true_jacobian,3) JacProp.eval_jac.(resad, vt, true_jacobian,3)]

##
using StatPlots
vio1 = boxplot(pred, lab=["Standard" "Jacobian propagation"], ylabel="Prediction RMS", reuse=false, yscale=:log10)
vio2 = boxplot(min.((sim),2), lab=["Standard" "Jacobian propagation"], ylabel="Simulation RMS", yscale=:log10)
vio3 = boxplot(jac, lab=["Standard" "Jacobian propagation"], ylabel="Jacobian Error", yscale=:identity)
plot(vio1,vio2,vio3,title=infostring); gui()
##
# savefig2("/local/home/fredrikb/papers/nn_prior/figs/valerr.tex")

plot(trajs[1])
predsimplot(resdiff[1].models, trajs[1]); gui()
predsimplot(resad[1].model, trajs[1], reuse=false); gui()
# simulate(resad[1].model, trajs[1])' |> plot
##
i = 1
j = 3
plot(simvals[i], control=false, layout=4, lab="Traj")
simplot!(resdiff[j].models, simvals[i], lab="Standard")
simplot!(resad[j].model, simvals[i], lab="AD JAcprop")
##



JacProp.eval_jac(resad[1], vt, true_jacobian)

JacProp.models(resad[1])

jacobian(resad[1].model,vt.xu[:,1])
