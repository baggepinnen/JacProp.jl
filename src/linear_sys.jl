if length(workers()) == 1 #src
    addprocs(4) #src
end #src
default(grid=false) #src
using ParallelDataTransfer #src
@everywhere using Parameters, JacProp, LTVModels, LTVModelsBase #src
@everywhere using Flux: params, jacobian #src
@everywhere using Flux.Optimise: Param, optimiser, expdecay #src
@everywhere begin #src

@with_kw struct LinearSys <: AbstractSystem
    A
    B
    N     = 1000
    nx    = size(A,1)
    nu    = size(B,2)
    h     = 0.02
    σ0    = 0
    sind  = 1:nx
    uind  = nx+1:(nx+nu)
    s1ind = (nx+nu+1):(nx+nu+nx)
end

function LinearSys(seed; nx = 10, nu = nx, h=0.02, kwargs...)
    srand(seed)
    A = randn(nx,nx)
    A = A-A'        # skew-symmetric = pure imaginary eigenvalues
    A = A - h*I     # Make 'slightly' stable
    A = expm(h*A)   # discrete time
    B = h*randn(nx,nu)
    LinearSys(;A=A, B=B, nx=nx, nu=nu, h=h, kwargs...)
end

function generate_data(sys::LinearSys, seed, validation=false)
    Parameters.@unpack A,B,N, nx, nu, h, σ0 = sys
    srand(seed)
    u      = filt(ones(5),[5], 10randn(N+2,nu))'
    t      = h:h:N*h+h
    x0     = randn(nx)
    x      = zeros(nx,N+1)
    x[:,1] = x0
    for i = 1:N-1
        x[:,i+1] = A*x[:,i] + B*u[:,i]
    end

    validation || (x .+= σ0 * randn(size(x)))
    u = u[:,1:N]
    @assert all(isfinite, u)
    x,u
end

function true_jacobian(sys::LinearSys, x, u)
    [sys.A sys.B]
end

function callbacker(epoch, loss,d,trace,model)
    printed = false
    info("New callbacker, epoch $epoch")
    function ()
        info("callback, epoch $epoch")
        l = sum(d->Flux.data(loss(d...)),d)
        increment!(trace,epoch,l)
        if !printed && epoch % 500 == 0
            println(@sprintf("Loss: %.4f", l))
            printed = true
        end
    end
end

num_params = 30
wdecay     = 0
stepsize   = 0.02
const sys  = LinearSys(1, N=200, h=0.02, σ0 = 0.01)
true_jacobian(x,u) = true_jacobian(sys,x,u)
nu         = sys.nu
nx         = sys.nx

end #src


# Generate validation data
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

# Generate training trajectories
trajs = [Trajectory(generate_data(sys, i)...) for i = 1:3]

sendto(collect(2:5), trajs=trajs, vt=vt) #src

# # Without jacprop
f2 = @spawnat 2 begin #src
    srand(1)
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
    for i = 1:3
        trainer(trajs[i], epochs=2000, jacprop=0, useprior=false)
    end
    trainer
end #src


# # With jacprop and prior
f3 = @spawnat 3 begin #src
    srand(1)
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainerj  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 1000I, σdivider = 20)
    for i = 1:3
        trainerj(trajs[i], epochs=1000, jacprop=1, useprior=true)
    end
    trainerj
end #src

# # With jacprop no prior
f4 = @spawnat 4 begin #src
    srand(1)
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainerjn  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 1000I, σdivider = 200)
    for i = 1:3
        trainerjn(trajs[i], epochs=1000, jacprop=1, useprior=false)
    end
    trainerjn
end #src


# # Visualize result
pyplot(reuse=false)
trainer,trainerj,trainerjn = fetch(f2), fetch(f3),fetch(f4)

## mutregplot(trainer, vt, true_jacobian, title="Witout jacprop", subplot=1, layout=(2,3), reuse=false, useprior=false, showltv=false)
## mutregplot!(trainerj, vt, true_jacobian, title="With jacprop and prior", subplot=2, link=:y, useprior=true, show=false, showltv=false)
## mutregplot!(trainerjn, vt, true_jacobian, title="With jacprop, no prior", subplot=3, link=:y, useprior=false, showltv=false)
## traceplot!(trainer, subplot=3)
## traceplot!(trainerj, subplot=5, show=false)
## traceplot!(trainerjn, subplot=4)

mutregplot(trainer, vt, true_jacobian, title="Witout jacprop", subplot=1, layout=(2,2), reuse=false, useprior=false, showltv=false, legend=false, xaxis=(1:3), xaxis=(1:3))
mutregplot!(trainerjn, vt, true_jacobian, title="With jacprop", subplot=2, link=:y, useprior=false, showltv=false, legend=false, xaxis=(1:3))
traceplot!(trainer, subplot=3, title="Training error", xlabel="Epoch", legend=false)
traceplot!(trainerjn, subplot=4, title="Training error", xlabel="Epoch", legend=false)

gui()
##


jacplot(trainer.models, vt, true_jacobian, label="Without", c=:red, reuse=false, fillalpha=0.2)
## jacplot!(KalmanModel(trainer, vt), vt, label="Without", c=:pink)
jacplot!(trainerj.models, vt, label="With", c=:blue, fillalpha=0.2)
## jacplot!(KalmanModel(trainerj, vt), vt, label="With", c=:cyan)
jacplot!(trainerjn.models, vt, label="With no", c=:green, fillalpha=0.2)
## jacplot!(KalmanModel(trainerjn, vt), vt, label="With no", c=:green)
gui()


## eigvalplot(trainer.models, vt, subplot=1, layout=3, title="g, Witout jacprop, wdecay=$wdecay")
## eigvalplot!(trainerj.models, vt, subplot=2, title="g, With jacprop and prior, wdecay=$wdecay")
## eigvalplot!(trainerjn.models, vt, subplot=3, title="g, With jacprop, no prior, wdecay=$wdecay")

eigvalplot(trainer.models, vt, true_jacobian, subplot=1, layout=2, title="g, Witout jacprop", markeralpha=0.05, link=:both, xlims=(-0.2, 1.2), ds=1, markerstrokealpha=0)
eigvalplot!(trainerjn.models, vt, true_jacobian, subplot=2, title="g, With jacprop", markeralpha=0.05, link=:both, xlims=(-0.2, 1.2), ds=1, markerstrokealpha=0)
plot!(ylims=(-0.8, 0.8))
gui()

ui = display_modeltrainer(trainerjn, size=(800,600))
jacplot(trainer.models, trainer.trajs[3], true_jacobian, ds=20)
@gif for i = 1:length(t)
    eigvalplot(trainer.models, trainer.trajs[1], true_jacobian, ds=20, onlyat=i)
end


## TODO: produce eigvalplots for system and diffsystem with and without wdecay
## TODO: make jacprop magnitude an option std()/10
## TODO: Validate vt in callback
## TODO: why can a network perform better but train with higher final loss? Overfitting?



wdecay = 0.0
f1 = @spawnat 1 begin #src
    srand(1)
    models     = [System(nx,nu,num_params, a) for a in default_activations]
    opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainers  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
    for i = 1:3
        trainers(trajs[i], epochs=2000, jacprop=1, useprior=false)
    end
    trainers
end #src


f2 = @spawnat 2 begin #src
    srand(1)
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainerds  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
    for i = 1:3
        trainerds(trajs[i], epochs=2000, jacprop=1, useprior=false)
    end
    trainerds
end #src

wdecay = 0.1
f3 = @spawnat 3 begin #src
    srand(1)
    models     = [System(nx,nu,num_params, a) for a in default_activations]
    opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainers  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
    for i = 1:3
        trainers(trajs[i], epochs=2000, jacprop=1, useprior=false)
    end
    trainers
end #src


f4 = @spawnat 4 begin #src
    srand(1)
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainerdswd  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
    for i = 1:3
        trainerdswd(trajs[i], epochs=2000, jacprop=1, useprior=false)
    end
    trainerdswd
end #src

trainers,trainerds = fetch(f1), fetch(f2)
trainerswd,trainerdswd = fetch(f3), fetch(f4)
eigvalplot(trainers.models, vt, true_jacobian, title="f, wdecay=0", layout=4, subplot=1, markeralpha=0.05, ds=1, markerstrokealpha=0)
eigvalplot!(trainerds.models, vt, true_jacobian,  title="g, wdecay=0", subplot=2, markeralpha=0.05, ds=1, markerstrokealpha=0)
eigvalplot!(trainerswd.models, vt, true_jacobian,  title="f, wdecay=$wdecay", subplot=3, markeralpha=0.05, ds=1, markerstrokealpha=0)
eigvalplot!(trainerdswd.models, vt, true_jacobian,  title="g, wdecay=$wdecay", subplot=4, markeralpha=0.05, ds=1, markerstrokealpha=0)
plot!(link=:both, xlims=(0,1.1), ylims=(-0.5, 0.5))
gui()


ui = display_modeltrainer(trainerdswd, size=(800,600))

for acti in 1:4
    plot(layout=4, link=:both)
    actstr = string(JacProp.default_activations[acti])
    actstr = actstr[1:2] == "NN" ? actstr[7:end] : actstr
    for tri in 1:4
        tr = [trainers, trainerds, trainerswd, trainerdswd][tri]
        eigvalplot!([tr.models[acti]], vt, true_jacobian, title="$actstr, $(tri%2==0 ? "g" : "f"), wdecay=$(tri>2 ? wdecay : 0)", subplot=tri, markeralpha=0.05, ds=3, markerstrokealpha=0)
    end
    gui()
end
