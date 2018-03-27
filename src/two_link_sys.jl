include(Pkg.dir("DynamicMovementPrimitives","src","two_link.jl"))

using TwoLink, Parameters, JacProp, OrdinaryDiffEq
using Flux: params, jacobian
using Flux.Optimise: Param, optimiser, expdecay

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

function generate_data(sys::TwoLinkSys, seed, validation=false; ufun=u->filt(ones(50),[50], 10u')')
    @unpack N, nu, nx, h, σ0, sind, uind, s1ind = sys
    srand(seed)

    done = false
    local x,u
    while !done
        u = ufun(randn(nu,N+2))
        t  = 0:h:N*h
        x0 = [-0.4,0,0,0]
        prob = OrdinaryDiffEq.ODEProblem((x,p,t)->time_derivative(x, u[:,floor(Int,t/h)+1]),x0,(t[[1,end]]...))
        sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8)
        x = hcat(sol(t)...)
        done = all(abs.(x[1:2,:]) .< 0.9π)
    end
    # y = hcat([time_derivative(x[:,t], u[:,t])[3:4] for t in 1:N]...)
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

function callbacker(loss,d)
    i = 0
    function ()
        i % 100 == 0 && println(@sprintf("Loss: %.4f", sum(d->Flux.data(loss(d...)),d)))
        i += 1
    end
end



num_params = 10
wdecay     = 0
stepsize   = 0.05
sys        = TwoLinkSys(N=1000, h=0.02, σ0 = 0.1)
true_jacobian(x,u) = true_jacobian(sys,x,u)
nu         = sys.nu
nx         = sys.nx
models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]

trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
x,u = generate_data(sys, 1)

t = Trajectory(x[:,1:end-1],u, x[:,2:end])
push!(trainer,t)
train!(trainer, epochs=500, jacprop=0)

x,u = generate_data(sys, 2)
trainer(x[:,1:end-1],u, x[:,2:end], epochs=100, jacprop=1)

x,u = generate_data(sys, 3)
trainer(x[:,1:end-1],u, x[:,2:end], epochs=100, jacprop=1)

x,u = generate_data(sys, 4)
trainer(x[:,1:end-1],u, x[:,2:end], epochs=100, jacprop=1)

trainer(epochs=50, jacprop=1)
trainer(epochs=500, jacprop=1)
# trainer(epochs=500, jacprop=1)

ui = display_modeltrainer(trainer, size=(800,600))
jacplot(trainer.models, trainer.trajs[3], true_jacobian, ds=20)
@gif for i = 1:length(t)
    eigvalplot(trainer.models, trainer.trajs[1], true_jacobian, ds=20, onlyat=i)
end

# TODO: Sample points all over state-space and use as validation
# TODO: regularize LTV with NN
