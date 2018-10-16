# weave("linear_sys_tmp.jl", doctype="github", out_path="build")
using Plots
default(grid=false)
cd(@__DIR__)
using Parameters, JacProp, OrdinaryDiffEq, LTVModels, LTVModelsBase, Random, LinearAlgebra, Statistics, DSP
using Flux: params, jacobian
using Flux.Optimise: Param, optimiser, expdecay

function simaverage(x, n)
    N,p = size(x)
    res = map(1:p) do i
        mean(reshape(x[:,i], N÷n,:), dims=2)
    end
    hcat(res...)
end

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
    Random.seed!(seed)
    A = randn(nx,nx)
    A = A-A'        # skew-symmetric = pure imaginary eigenvalues
    A = A - h*I     # Make 'slightly' stable
    A = exp(h*A)   # discrete time
    B = h*randn(nx,nu)
    LinearSys(;A=A, B=B, nx=nx, nu=nu, h=h, kwargs...)
end

function generate_data(sys::LinearSys, seed, validation=false)
    Parameters.@unpack A,B,N, nx, nu, h, σ0 = sys
    Random.seed!(seed)
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
        # i % 500 == 0 && println(@sprintf("Loss: %.4f", l))
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


## Monte-Carlo evaluation
num_montecarlo = 12
it = num_montecarlo ÷ 2
res = map(1:num_montecarlo) do it
    trajs = [Trajectory(generate_data(sys, i)...) for i = it .+ (1:2)]

    #' ## Without jacprop
    Random.seed!(1)
    # wdecay = exp10.(range(-6, stop=1, length=num_montecarlo))[it]
    wdecay = 1e-4
    models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    opts       = [[ADAM.(params.(models), stepsize, ϵ=1e-2); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
    trainer  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, σdivider = wdecay)
    for i = 1:2
        trainer(trajs[i], epochs=0, jacprop=0, useprior=false)
    end
    # trainer(epochs=2000)
    # serializesave("trainer", trainer)
    # trainer = deserialize("trainer")
    #' ## With jacprop

    # Random.seed!(1)
    # models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
    # opts       = ADAM.(params.(models), stepsize, decay=0.0005)
    # trainerjn  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
    # for i = 1:2
    #     trainerjn(trajs[i], epochs=0, jacprop=1, useprior=false)
    # end
    # trainerjn(epochs=1000, jacprop=1)
    # serialize("trainerjn", trainerjn)
    # trainerjn = deserialize("trainerjn")

    #' ## With AD-jacprop
    Random.seed!(1)
    # λ = exp10.(range(-4, stop=3, length=num_montecarlo))[it]
    λ = 0.15
    model     = JacProp.ADDiffSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
    opt       = LTVModels.ADAMOptimizer.(model.w, α = 0.1stepsize)
    trainerad = ADModelTrainer(;model=model, opt=opt, λ=λ, testdata = vt)
    for i = 1:2
        trainerad(trajs[i], epochs=0)
    end
    # trainerad(epochs=2000)
    # serializesave("trainerad", trainerad)
    # trainerad = deserialize("trainerad")

    trainer, trainerad
end

# ress = getindex.(res,1)
# resad = getindex.(res,2)
resdiff = getindex.(res,1)
resaddiff = getindex.(res,2)
# λs = getfield.(ress, :σdivider)
# λad = getfield.(resad, :λ)

# Threads.@threads for trainer=ress       train!(trainer, epochs=2500) end
# @info("Done ress")
# Threads.@threads for trainer=resad      train!(trainer, epochs=2500) end
# @info("Done resad")
Threads.@threads for trainer=resdiff    train!(trainer, epochs=300) end
@info("Done resdiff")
Threads.@threads for trainer=resaddiff  train!(trainer, epochs=300) end
@info("Done resaddiff")

# serialize("results_linear_eigvals_final", (resdiff,resaddiff))
# (resdiff,resaddiff) = deserialize("results_linear_eigvals_final")

using JacProp: eval_pred, eval_sim, eval_jac, eval_jac2
λd = getfield.(resdiff, :σdivider)
λadd = getfield.(resaddiff, :λ)
xv = [λd λadd]
labelvec = ["f" "g"]
infostring = @sprintf("Num hidden: %d, sigma: %2.2f, Montecarlo: %d", num_params, sys.σ0, num_montecarlo)
simvals = [Trajectory(generate_data(sys, i, true)...) for i = 4:6]
pred  = [eval_pred.(resdiff, (vt,)) eval_pred.(resaddiff, (vt,))]
sim   = simaverage(vcat([[eval_sim.(resdiff, (simval,)) eval_sim.(resaddiff, (simval,))] for simval in simvals]...), 3)
sim = [x > 10 ? NaN : x for x in sim]
jac = [eval_jac2.(resdiff, (vt,), true_jacobian,3) eval_jac2.(resaddiff, (vt,), true_jacobian,3)]

## Plot error vs λ regularization parameter
plot(xv,pred, layout=(3,1), subplot=1, ylabel="Prediction RMS", lab=["Standard" "Jacobian propagation"], background_color_legend=false, xscale=:log10,yscale=:log10, size=(1200,1100),legend=false)
plot!(xv,min.(sim,10), subplot=2, ylabel="Simulation RMS", lab=["Standard" "Jacobian propagation"], background_color_legend=false, xscale=:log10,yscale=:log10,legend=false)
plot!(xv,jac, subplot=3, ylabel="Jacobian Error", lab=["Standard" "Jacobian propagation"], background_color_legend=false, xscale=:log10,yscale=:log10,legend=false)

plot()
traceplot!.(resdiff, m=(:blue, 1), markerstrokealpha=0)
traceplot!.(resaddiff, m=(:red, 1), markerstrokealpha=0)
plot!(getfield.(resaddiff, :tracev), m=(:green, 1), markerstrokealpha=0)
gui()




using StatPlots
xticks = (1:2, ["Standard" "Jacprop"])
xvals = [1 2].*ones(num_montecarlo)
common = (marker_z=xv, legend=false, xticks=xticks)
vio1 = violin(xvals,pred; title="Prediction RMS",ylims=(0,maximum(pred)), common...)
# vio2 = violin(xvals,sim; title="Simulation RMS",  common...)
vio3 = violin(xvals,(jac); title="Jacobian Error",ylims=(0,maximum(jac)), common...)
plot(vio1,vio3, colorbar=false, layout=(1,2))
# savefig3("/local/home/fredrikb/phdthesis/blackbox/figs/boxplot_linear.tex")


common = (marker_z=xv, legend=false, xticks=false)
vio1 = violin([1],pred[:,1]; ylabel="Prediction RMS",  side=:left, ylims=(0,maximum(pred)), common...)
violin!([1],pred[:,2]; ylabel="Prediction RMS",  side=:right, ylims=(0,maximum(pred)), common...)
vio3 = violin([1],jac[:,1]; ylabel="Jacobian Error", side=:left, ylims=(0,maximum(jac)), common...)
violin!([1],jac[:,2]; ylabel="Jacobian Error", side=:right, ylims=(0,maximum(jac)), common...)
plot(vio1,vio3, colorbar=false, layout=(1,2))




#' ## Visualize result
using Plots.PlotMeasures
F = font(18, "times")
fontopts = [(:xticks, 0:0.5:1), (:yticks, -0.3:0.3:0.3), (:xlims, (0,1.4)), (:ylims, (-0.5, 0.5)), (:grid, false), (:markeralpha, 0.2), (:ds, 5), (:markerstrokealpha, 0), (:titlefont, F), (:tickfont, F), (:xtickfont, F), (:ytickfont, F), (:guidefont, F), (:xguidefont, F), (:yguidefont, F)]

# mutregplot(trainer, vt, true_jacobian, title="Witout jacprop", subplot=1, layout=(2,2), reuse=false, useprior=false, showltv=false, legend=false, xaxis=(1:3), xaxis=(1:3))
# mutregplot!(trainerjn, vt, true_jacobian, title="With jacprop", subplot=2, link=:y, useprior=false, showltv=false, legend=false, xaxis=(1:3))
# traceplot!(trainer, subplot=3, title="Training error", xlabel="Epoch", legend=false)
# traceplot!(trainerjn, subplot=4, title="Training error", xlabel="Epoch", legend=false)

#' The top row shows the error (Frobenius norm) in the Jacbians for several points sampled randomly in the state space. The bottow row shows the training errors. The training errors are lower without jacprop, but he greater error in the Jacobians for the validation data indicates overfitting, which is prevented by jacprop.

eigvalplot(resdiff[5].models, vt, true_jacobian; title="Weight decay", layout=(1,2), subplot=1, size=(920,280), fontopts...)
# eigvalplot!(trainerjn.models, vt, true_jacobian;  title="Jacprop", subplot=2, fontopts...)
eigvalplot!(resaddiff[7].model, vt;  title="Jacprop", subplot=2, fontopts...)
gui()
# savefig("/local/home/fredrikb/papers/nn_prior/figs/jacpropeig2.pdf")
# pdftopng -r 300 -aa yes jacpropeig2.pdf jacpropeig2.png



















error()
#' # Weight decay
#' ## Weight decay off
Random.seed!(1)
models     = [System(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)
trainers  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainers(trajs[i], epochs=0, jacprop=0, useprior=false)
end
trainers(epochs=1000, jacprop=1)

Random.seed!(1)
models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)
trainerds  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainerds(trajs[i], epochs=0, jacprop=0, useprior=false)
end
trainerds(epochs=1000, jacprop=1)

#' ## Weight decay on
wdecay = 0.1
Random.seed!(1)
models     = [System(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainerswd  = ModelTrainer(;models=models, opts=opts, losses=JacProp.loss.(models), cb=callbacker, kalmanopts...)
for i = 1:3
    trainerswd(trajs[i], epochs=0, jacprop=0, useprior=false)
end
trainerswd(epochs=1000, jacprop=1)


Random.seed!(1)
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
eigvalplot(trainers.models, vt, true_jacobian;  title="", axis=false, fontopts...)
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
