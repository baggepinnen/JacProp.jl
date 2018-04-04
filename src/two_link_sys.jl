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



num_params = 20
wdecay     = 0
stepsize   = 0.05
sys        = TwoLinkSys(N=200, h=0.02, σ0 = 0.01)
true_jacobian(x,u) = true_jacobian(sys,x,u)
nu         = sys.nu
nx         = sys.nx

## Generate validation data
function valdata()
    vx,vu,vy = Vector{Float64}[],Vector{Float64}[],Vector{Float64}[]
    for i = 20:100
        x,u = generate_data(sys,i, true)
        for j in 1:100:(sys.N-1)
            push!(vx, x[:,j])
            push!(vy, x[:,j+1])
            push!(vu, u[:,j])
        end
    end
    hcat(vx...),hcat(vu...),hcat(vy...)
end
vx,vu,vy = valdata()
vt = Trajectory(vx,vu,vy)

## Without jacprop
srand(1)
models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]

trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 2)


for i = 1:3
    t = Trajectory(generate_data(sys, i)...)
    trainer(t, epochs=1600, jacprop=0)
end

# trainer(epochs=500, jacprop=1)
inspectdr()
mutregplot(trainer, vt, true_jacobian, title="Witout jacprop");gui()


## With jacprop
srand(1)
models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]

trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 2)


for i = 1:3
    t = Trajectory(generate_data(sys, i)...)
    trainer(t, epochs=1600, jacprop=1)
end

# trainer(epochs=500, jacprop=1)
mutregplot(trainer, vt, true_jacobian, title="With jacprop");gui()
##



ui = display_modeltrainer(trainer, size=(800,600))
jacplot(trainer.models, trainer.trajs[3], true_jacobian, ds=20)
@gif for i = 1:length(t)
    eigvalplot(trainer.models, trainer.trajs[1], true_jacobian, ds=20, onlyat=i)
end


# TODO: Sample points all over state-space and use as validation
# TODO: see if error during first two iterations, when number of trajs is small, is smaller using jacprop
# TODO: make jacprop magnitude an option std()/10
# TODO: See if working better with different P ≢ 10
