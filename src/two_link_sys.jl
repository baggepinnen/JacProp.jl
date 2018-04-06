if length(workers()) == 1
    addprocs(4)
end
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

    function callbacker(epoch, loss,d,trace)
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

sendto(collect(2:5), trajs=trajs, vt=vt)

## Without jacprop
f2 = @spawnat 2 begin
    srand(1)
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
    for i = 1:3
        trainer(trajs[i], epochs=20000, jacprop=0, useprior=false)
        # traceplot(trainer)
    end
    trainer
end


## With jacprop and prior
f3 = @spawnat 3 begin
    srand(1)
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainerj  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 1000I, σdivider = 20)
    for i = 1:3
        trainerj(trajs[i], epochs=10000, jacprop=1, useprior=true)
        # traceplot(trainerj)
    end
    trainerj
end

## With jacprop no prior
f4 = @spawnat 4 begin
    srand(1)
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainerjn  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 1000I, σdivider = 200)
    for i = 1:3
        trainerjn(trajs[i], epochs=10000, jacprop=1, useprior=false)
        # traceplot(trainerjn)
    end
    trainerjn
end

pyplot(reuse=false)
trainer,trainerj,trainerjn = fetch(f2), fetch(f3),fetch(f4)

mutregplot(trainer, vt, true_jacobian, title="Witout jacprop", subplot=1, layout=(2,3), reuse=false, useprior=false)
mutregplot!(trainerj, vt, true_jacobian, title="With jacprop and prior", subplot=2, link=:y, useprior=true, show=false)
mutregplot!(trainerjn, vt, true_jacobian, title="With jacprop, no prior", subplot=3, link=:y, useprior=false)
traceplot!(trainer, subplot=4)
traceplot!(trainerj, subplot=5, show=false)
traceplot!(trainerjn, subplot=6)
gui()
##


jacplot(trainer.models, vt, true_jacobian, label="Without", c=:red, reuse=false, fillalpha=0.2)
# jacplot!(KalmanModel(trainer, vt), vt, label="Without", c=:pink)
jacplot!(trainerj.models, vt, label="With", c=:blue, fillalpha=0.2)
# jacplot!(KalmanModel(trainerj, vt), vt, label="With", c=:cyan)
jacplot!(trainerjn.models, vt, label="With no", c=:green, fillalpha=0.2)
# jacplot!(KalmanModel(trainerjn, vt), vt, label="With no", c=:green)
gui()


ui = display_modeltrainer(trainer, size=(800,600))
jacplot(trainer.models, trainer.trajs[3], true_jacobian, ds=20)
@gif for i = 1:length(t)
    eigvalplot(trainer.models, trainer.trajs[1], true_jacobian, ds=20, onlyat=i)
end


# TODO: Sample points all over state-space and use as validation
# TODO: see if error during first two iterations, when number of trajs is small, is smaller using jacprop
# TODO: make jacprop magnitude an option std()/10
# TODO: Validate vt in callback
# TODO: why can a network perform better but train with higher final loss?
