module JacProp

export default_activations, Trajectory, ModelTrainer, sample_jackprop, push!, train!

using Parameters, ForwardDiff, StatsBase, Reexport, Lazy, Juno
@reexport using LTVModels, Flux, ValueHistories, IterTools, MLDataUtils
using Flux: back!, truncate!, treelike, train!, mse, testmode!, params, jacobian, throttle
using Flux.Optimise: Param, optimiser, RMSProp, expdecay

const default_activations = [swish, Flux.sigmoid, tanh, elu]

@with_kw mutable struct Trajectory
    x::Matrix{Float64}
    u::Matrix{Float64}
    y::Matrix{Float64}
    xu::Matrix{Float64}
    n::Int
    m::Int
    function Trajectory(x,u)
        @assert size(x,2) == size(u,2) "The second dimension of x and u (time) must be the same"
        x,u,y = x[:,1:end-1],u[:,1:end-1],y[:,2:end]
        new(x,u,y,[x;u],size(x,1), size(u,1))
    end
    function Trajectory(x,u,y)
        @assert size(x,2) == size(u,2) == size(y,2) "The second dimension of x,u and y (time) must be the same"
        new(x,u,y,[x;u],size(x,1), size(u,1))
    end
end
Base.length(t::Trajectory) = size(t.x,2)

@with_kw struct ModelTrainer
    models
    opts
    losses
    R1 = I
    R2 = 10_000I
    P0 = 10_000I
    trajs::Vector{Trajectory} = Trajectory[]
end


include("utilities.jl")



function Flux.train!(mt::ModelTrainer, epochs, jacprop)
    @unpack models,opts,losses,trajs = mt
    data = todata(mt)
    data = reduce(vcat, data, [todata(sample_jackprop(mt)) for i = 1:jacprop])
    dataset = ncycle(data, epochs)
    @progress for (loss, opt) in zip(losses,opts)
        i = 0
        function cb()
            i % 50 == 0 && println(@sprintf("Loss: %.4f", sum(d->loss(d...).tracker.data[1],data)))
            i += 1
        end
        train!(loss, dataset, opt, cb=cb)
    end
end

function sample_jackprop(mt::ModelTrainer)
    modeltype = typeof(mt.models[1])
    @unpack R1,R2,P0 = mt
    perturbed_trajs = map(mt.trajs) do traj
        @unpack x,u,n,m = traj
        ltvmodel = LTVModels.fit_model(LTVModels.KalmanModel, x,u,R1,R2,P0, extend=true)
        xa = x .+ std(x,2)/10 .* randn(size(x))
        ua = u .+ std(u,2)/10 .* randn(size(u))
        ya = LTVModels.predict(ltvmodel, xa, ua)
        Trajectory(xa,ua,ya)
    end
end

todata(trajs::Vector{Trajectory}) = [(traj.xu,traj.y) for traj in trajs]
todata(mt::ModelTrainer) = todata(mt.trajs)

Base.push!(mt::ModelTrainer, t::Trajectory) = push!(mt.trajs, t)
Base.push!(mt::ModelTrainer, data::Matrix...) = push!(mt.trajs, Trajectory(data...))

(mt::ModelTrainer)(; epochs=1, jacprop=1) = train!(mt, epochs, jacprop)

function (mt::ModelTrainer)(t::Trajectory; epochs=1, jacprop=1)
    push!(mt,t)
    train!(mt, epochs, jacprop)
end

(mt::ModelTrainer)(data::Matrix...; kwargs...) = mt(Trajectory(data...); kwargs...)

Lazy.@forward ModelTrainer.models predict, simulate, Flux.jacobian

end # module
