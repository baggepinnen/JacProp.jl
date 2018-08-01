# weave("linear_sys_tmp.jl", doctype="github", out_path="build")
using Plots
default(grid=false)
using Parameters, JacProp, OrdinaryDiffEq, LTVModels, LTVModelsBase
using Flux: params, jacobian
using Flux.Optimise: Param, optimiser, expdecay

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
const sys  = LinearSys(1, nx = 10, N=200, h=0.02, σ0 = 0.01)
true_jacobian(x,u) = true_jacobian(sys,x,u)
nu         = sys.nu
nx         = sys.nx
kalmanopts = [:P => 10, :R2 => 1000I, :σdivider => 20]


#' Generate validation data
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

#' Generate training trajectories
trajs = [Trajectory(generate_data(sys, i)...) for i = 1:3]

serializesave(filename, data) = open(f->serialize(f, data), filename, "w")
Base.deserialize(filename) = open(f->deserialize(f), filename)



#' ## Without jacprop

srand(1)
models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)
trainer  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainer(trajs[i], epochs=0, jacprop=0, useprior=false)
end
trainer(epochs=2000)
serializesave("trainer", trainer)
trainer = deserialize("trainer")
#' ## With jacprop

srand(1)
models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)
trainerjn  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainerjn(trajs[i], epochs=0, jacprop=1, useprior=false)
end
trainerjn(epochs=1000, jacprop=1)
serializesave("trainerjn", trainerjn)
trainerjn = deserialize("trainerjn")

#' ## With AD-jacprop
srand(1)
model     = JacProp.ADDiffSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
opt       = LTVModels.ADAMOptimizer(model.w, α = 0.1stepsize)
trainerad = ADModelTrainer(;model=model, opt=opt, λ=1, testdata = vt)
for i = 1:3
    trainerad(trajs[i], epochs=0)
end
trainerad(epochs=200)
serializesave("trainerad", trainerad)
# trainerad = deserialize("trainerad")

#' ## Visualize result
using Plots.PlotMeasures
F = font(18, "times")
fontopts = [(:xticks, 0:0.5:1), (:yticks, -0.3:0.3:0.3), (:xlims, (0,1.4)), (:ylims, (-0.5, 0.5)), (:grid, false), (:markeralpha, 0.2), (:ds, 1), (:markerstrokealpha, 0), (:titlefont, F), (:tickfont, F), (:xtickfont, F), (:ytickfont, F), (:guidefont, F), (:xguidefont, F), (:yguidefont, F)]

# mutregplot(trainer, vt, true_jacobian, title="Witout jacprop", subplot=1, layout=(2,2), reuse=false, useprior=false, showltv=false, legend=false, xaxis=(1:3), xaxis=(1:3))
# mutregplot!(trainerjn, vt, true_jacobian, title="With jacprop", subplot=2, link=:y, useprior=false, showltv=false, legend=false, xaxis=(1:3))
# traceplot!(trainer, subplot=3, title="Training error", xlabel="Epoch", legend=false)
# traceplot!(trainerjn, subplot=4, title="Training error", xlabel="Epoch", legend=false)

#' The top row shows the error (Frobenius norm) in the Jacbians for several points sampled randomly in the state space. The bottow row shows the training errors. The training errors are lower without jacprop, but he greater error in the Jacobians for the validation data indicates overfitting, which is prevented by jacprop.

eigvalplot(trainer.models, vt, true_jacobian; title="Baseline", layout=(1,3), subplot=1, size=(920,380), fontopts...)
eigvalplot!(trainerjn.models, vt, true_jacobian;  title="Jacprop", subplot=2, fontopts...)
eigvalplot!(trainerad.model, vt, true_jacobian;  title="AD Jacprop", subplot=3, fontopts...)
gui()
savefig("/local/home/fredrikb/papers/nn_prior/figs/jacpropeig2.pdf")
#' # Weight decay
#' ## Weight decay off
srand(1)
models     = [System(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)
trainers  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainers(trajs[i], epochs=0, jacprop=0, useprior=false)
end
trainers(epochs=1000, jacprop=1)

srand(1)
models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)
trainerds  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainerds(trajs[i], epochs=0, jacprop=0, useprior=false)
end
trainerds(epochs=1000, jacprop=1)

#' ## Weight decay on
wdecay = 0.1
srand(1)
models     = [System(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainerswd  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainerswd(trajs[i], epochs=0, jacprop=0, useprior=false)
end
trainerswd(epochs=1000, jacprop=1)


srand(1)
models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainerdswd  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainerdswd(trajs[i], epochs=0, jacprop=0, useprior=false)
end
trainerdswd(epochs=1000, jacprop=1)

#' Visualize result
#+ fig_ext="png"

eigvalplot(trainers.models, vt, true_jacobian; title="\$f, \\lambda=0\$", layout=4, subplot=1, size=(920,700), fontopts...)
eigvalplot!(trainerds.models, vt, true_jacobian;  title="\$g, \\lambda=0\$", subplot=2, fontopts...)
eigvalplot!(trainerswd.models, vt, true_jacobian;  title="\$f, \\lambda=$wdecay\$", subplot=3, fontopts...)
eigvalplot!(trainerdswd.models, vt, true_jacobian;  title="\$g, \\lambda=$wdecay\$", subplot=4, fontopts...)
# plot!(link=:both)
gui()
# savefig("/local/home/fredrikb/papers/nn_prior/figs/all_zoom.pdf")
#' # Different activation functions

# ui = display_modeltrainer(trainerdswd, size=(800,600))

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
