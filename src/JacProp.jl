module JacProp

export Trajectory, default_activations, ModelTrainer, sample_jacprop, push!, train!, display_modeltrainer

using LTVModelsBase, Parameters, ForwardDiff, StatsBase, Reexport, Lazy, Juno
@reexport using LTVModels, Flux, ValueHistories, InterTools, MLDataUtils
using Flux: back!, truncate!, treelike, train!, mse, testmode!, params, jacobian, throttle
using Flux.Optimise: Param, optimiser, RMSProp, expdecay

using Plots, InteractNext, Observables, DataStructures

const default_activations = [swish, Flux.sigmoid, tanh, elu]




"""
ModelTrainer

- `models` A vector of functions
- `opts` A vector of optimizers
- `losses` A vector of loss functions
- `R1` Parameter drift covariance for Kalman smoother
- `R2` Measurement noise covariance
- `P0` Initial covariance matrix for Kalman smoother
- `P` Covariance magnification of NN prior
- `cb` can be either `()->()` or a a Function that creates a closure around `loss` and `data` (loss,data)->(()->())

See also [`LTVModels`](@ref),  [`LTVModels.KalmanModel`](@ref)
"""
@with_kw mutable struct ModelTrainer{cbT}
    models
    opts
    losses
    R1 = I
    R2 = 10_000I
    P0 = 10_000I
    P  = 10_000
    trajs::Vector{Trajectory} = Trajectory[]
    cb::cbT
    modelhistory = []
end


include("utilities.jl")

to_callback(cb,args...) = methods(cb).mt.max_args > 1 ? cb(args...) : cb

"""
train!(mt::ModelTrainer; epochs=1, jacprop=1)

See also [`ModelTrainer`](@ref)
"""
function Flux.train!(mt::ModelTrainer; epochs=1, jacprop=1)
    @unpack models,opts,losses,trajs = mt
    data1 = todata(mt)
    data = chain(data1, ncycle(todata(sample_jacprop(mt)), jacprop))
    dataset = ncycle(data, epochs)
    for (loss, opt) in zip(losses,opts)
        train!(loss, dataset, opt, cb=to_callback(mt.cb,loss,data1))
    end
    push!(mt.modelhistory, deepcopy(models))
end

function sample_jacprop(mt::ModelTrainer)
    modeltype = typeof(mt.models[1])
    @unpack R1,R2,P0 = mt
    perturbed_trajs = map(mt.trajs) do traj
        @unpack x,u,nx,nu = traj
        ltvmodel = KalmanModel(mt, traj, P = 1000)
        # ltvmodel = KalmanModel(x,u,R1,R2,P0, extend=true)
        xa = x .+ std(x,2)/10 .* randn(size(x))
        ua = u .+ std(u,2)/10 .* randn(size(u))
        ya = LTVModels.predict(ltvmodel, xa, ua)
        Trajectory(xa,ua,ya)
    end
end

function LTVModels.KalmanModel(mt::ModelTrainer, t::Trajectory, ms=mt.models;
    P=mt.P, useprior=!isempty(mt.modelhistory))
    @unpack x,u,nx,nu = t
    @unpack R1,R2,P0  = mt
    T                 = length(t)
    if useprior
        model = KalmanModel(zeros(nx,nx,T),zeros(nx,nu,T),zeros(1,1,T),false)
        Jm, Js = jacobians(ms, t)
        Pt       = cat(3,[diagm(Js[:,i].^2 .+ P) for i=1:T]...) # TODO: magic number
        fx       = cat(3,[reshape(Jm[1:nx^2,i], nx,nx) for i=1:T]...)
        fu       = cat(3,[reshape(Jm[nx^2+1:end,i], nx,nu) for i=1:T]...)
        prior    = KalmanModel(fx,fu,Pt,false)
        ltvmodel = KalmanModel(model, prior, x,u,R1,R2,P0, extend = true)
    else # Don't use prior if it's the first time training
        ltvmodel = KalmanModel(x,u,R1,R2,P0, extend = true)
    end
end

todata(trajs::Vector{Trajectory}) = [(traj.xu,traj.y) for traj in trajs]
todata(mt::ModelTrainer) = todata(mt.trajs)

Base.push!(mt::ModelTrainer, t::Trajectory) = push!(mt.trajs, t)
Base.push!(mt::ModelTrainer, data::Matrix...) = push!(mt.trajs, Trajectory(data...))

(mt::ModelTrainer)(; epochs=1, jacprop=1, kwargs...) = train!(mt; epochs=epochs, jacprop=jacprop, kwargs...)

function (mt::ModelTrainer)(t::Trajectory; kwargs...)
    push!(mt,t)
    train!(mt; kwargs...)
end

(mt::ModelTrainer)(data::Matrix...; kwargs...) = mt(Trajectory(data...); kwargs...)

Lazy.@forward ModelTrainer.models predict, simulate, Flux.jacobian

include("plots.jl")

end # module
