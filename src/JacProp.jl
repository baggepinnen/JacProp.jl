module JacProp

using OrdinaryDiffEq, LTVModels, Parameters, ForwardDiff, StatsBase
using Flux, ValueHistories, IterTools, MLDataUtils, OrdinaryDiffEq, Parameters
using Flux: back!, truncate!, treelike, train!, mse, testmode!, params, jacobian
using Flux.Optimise: Param, optimiser, RMSProp, expdecay

include("utilities.jl")
const default_activations = [swish, Flux.sigmoid, tanh, elu]

struct Trajectory
    x
    u
    n
    m
    Trajectory(x,u) = new(x,u,size(x,1), size(u,1))
end
Base.length(t::Trajectory) = size(t.x,2)

@with_kw struct ModelTrainer
    m
    opt
    loss
    R1 = I
    R2 = 10_000I
    P0 = 10_000I
    trajs::Vector{Trajectory} = Trajectory[]
end

function Flux.train!(mt::ModelTrainer, dataset, modeltype)
    @unpack m,opt,loss = mt
    train!(loss, dataset, opt)
end

function sample_jackprop(mt::ModelTrainer, jacprop = 1)
    modeltype = typeof(mt.m)
    for traj in mt.trajs
        @unpack x,u,n,m = traj
        ltvmodel = LTVModels.fit_model(LTVModels.KalmanModel, x,u,R1,R2,P0, extend=true)
        xt,ut,yt = [x;u], u, y
        for i = 1:jacprop
            xa = x .+ std(x,2)/10 .* randn(size(x))
            ua = u .+ std(u,2)/10 .* randn(size(u))
            ya = LTVModels.predict(ltvmodel, xa, ua)
            xt = [xt [xa;ua]]
            ut = [ut ua]
            yt = modeltype ∈ [VelSystem, VelSystemD] ? [yt ya[3:4,:]] : [yt ya]
        end
    end
end

function add_trajectory()

    results = map(1:n_bootstrap) do it
        # global x, y, xv, yv
        opt        = [ADAM(params(models[it]), stepsize, decay=0.005); [expdecay(Param(p), wdecay) for p in params(models[it]) if p isa AbstractMatrix]]
        results    = fit_model(opt, loss(models[it]), models[it], xt, yt, ut, sys,modeltype, iters=iters, doplot=doplot, batch_size = size(yt,2))
        @pack results = x, u, y
        println("Done: ", it)
        results
    end
    Jm, Js, Jtrue = all_jacobians(results)
    Jltv = cat(2,model.At, model.Bt)
    @pack results = Jm, Js, Jtrue, Jltv
    # if doplot
    #     fig_jac = plot_jacobians(Jm, Js, Jtrue)
    #     fig_time = plotresults(results)
    #     fig_eig = plot_eigvals(results)
    #     fig_jac, fig_time, fig_eig
    #     @pack results = fig_jac, fig_time, fig_eig
    # end
    results
end # Outer iter
outer_results[:models] = models
end




end # module
