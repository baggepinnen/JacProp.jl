include(Pkg.dir("DynamicMovementPrimitives","src","two_link.jl"))

using TwoLink, Parameters, JacProp, OrdinaryDiffEq
using Flux: params, jacobian
using Flux.Optimise: Param, optimiser, expdecay

@with_kw struct TwoLinkSys
    N  = 1000
    n  = 2
    ns = 2n
    h = 0.02
    σ0 = 0
    sind = 1:ns
    uind = ns+1:(ns+n)
    s1ind = (ns+n+1):(ns+n+ns)
end

function generate_data(sys::TwoLinkSys, seed, validation=false; ufun=u->filt(ones(50),[50], 10u')')
    @unpack N, n, ns, h, σ0, sind, uind, s1ind = sys
    srand(seed)

    done = false
    local x,u
    while !done
        u = ufun(randn(n,N+2))
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

function true_jacobian(sys::TwoLinkSys, evalpoint, x, u)
    Jtrue = ReverseDiff.jacobian(x->time_derivative(x, u[:,evalpoint]), x[:,evalpoint])[:,1:4]
    Jtrue = [Jtrue ReverseDiff.jacobian(u->time_derivative(x[:,evalpoint], u), u[:,evalpoint])]
    Jtrue = expm([sys.h*Jtrue;zeros(2,6)])[1:4,:] # Discretize
end


num_params = 10
wdecay     = 0
stepsize   = 0.05
sys        = TwoLinkSys(N=1000, h=0.02, σ0 = 0.1)
n          = sys.n
ns         = sys.ns
models     = [System(n,ns,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]

trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models))
x,u = generate_data(sys, 1)

t = Trajectory(x[:,1:end-1],u, x[:,2:end])
push!(trainer,t)
JacProp.train!(trainer, 100, 1)

x,u = generate_data(sys, 2)
trainer(x[:,1:end-1],u, x[:,2:end])

x,u = generate_data(sys, 3)
trainer(x[:,1:end-1],u, x[:,2:end])

x,u = generate_data(sys, 4)
trainer(x[:,1:end-1],u, x[:,2:end])

trainer(epochs=100)
