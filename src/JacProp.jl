module JacProp

export default_activations, Trajectory, ModelTrainer, sample_jackprop, push!, train!, display_modeltrainer

using Parameters, ForwardDiff, StatsBase, Reexport, Lazy, Juno
@reexport using LTVModels, Flux, ValueHistories, IterTools, MLDataUtils
using Flux: back!, truncate!, treelike, train!, mse, testmode!, params, jacobian, throttle
using Flux.Optimise: Param, optimiser, RMSProp, expdecay

using Plots, InteractNext, Observables, DataStructures

const default_activations = [swish, Flux.sigmoid, tanh, elu]

@with_kw mutable struct Trajectory
    x::Matrix{Float64}
    u::Matrix{Float64}
    y::Matrix{Float64}
    xu::Matrix{Float64}
    nx::Int
    nu::Int
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

"""
ModelTrainer

- `models` A vector of functions
- `opts` A vector of optimizers
- `losses` A vector of loss functions
- `R1` Parameter drift covariance for Kalman smoother
- `R2` Measurement noise covariance
- `P0` Initial covariance matrix for Kalman smoother
- `cb` can be either `()->()` or a a Function that creates a closure around `loss` and `data` (loss,data)->(()->())

See also [`LTVModels`](@ref), [`LTVModels.fit_model`](@ref), [`LTVModels.KalmanModel`](@ref)
"""
@with_kw mutable struct ModelTrainer{cbT}
    models
    opts
    losses
    R1 = I
    R2 = 10_000I
    P0 = 10_000I
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
    data = todata(mt)
    data = reduce(vcat, data, [todata(sample_jackprop(mt)) for i = 1:jacprop])
    dataset = ncycle(data, epochs)
    for (loss, opt) in zip(losses,opts)
        train!(loss, dataset, opt, cb=to_callback(mt.cb,loss,data))
    end
    push!(mt.modelhistory, deepcopy(models))
end

function sample_jackprop(mt::ModelTrainer)
    modeltype = typeof(mt.models[1])
    @unpack R1,R2,P0 = mt
    perturbed_trajs = map(mt.trajs) do traj
        @unpack x,u,nx,nu = traj
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

(mt::ModelTrainer)(; epochs=1, jacprop=1, kwargs...) = train!(mt; epochs=epochs, jacprop=jacprop, kwargs...)

function (mt::ModelTrainer)(t::Trajectory; kwargs...)
    push!(mt,t)
    train!(mt; kwargs...)
end

(mt::ModelTrainer)(data::Matrix...; kwargs...) = mt(Trajectory(data...); kwargs...)

Lazy.@forward ModelTrainer.models predict, simulate, Flux.jacobian

@recipe function plot_Trajectory(t::Trajectory; filtering=0)
    layout --> 2
    show --> false
    @series begin
        title --> "States"
        xlabel --> "Time"
        subplot --> 1
        filtering > 0 ? filt(ones(filtering),[filtering], t.x') : t.x'
    end
    @series begin
        title --> "Control signal"
        xlabel --> "Time"
        subplot --> 2
        filtering > 0 ? filt(ones(filtering),[filtering], t.u') : t.u'
    end
    delete!(plotattributes, :filtering)
    nothing
end

@userplot EigvalPlot

@recipe function eigvalplot(h::EigvalPlot; ds=10, cont=false)
    ms = h.args[1]
    t = h.args[2]
    title --> "Eigenvalue spectrum"
    @unpack xu,nu,nx = t
    lim = 1e-3
    cmap = colormap("Blues", length(t))
    show --> false
    for evalpoint = 1:ds:length(t)
        @series begin
            c --> cmap[evalpoint]
            seriestype := :scatter
            J = jacobian(ms, xu[:,evalpoint])[1]
            e = eigvals(J[1:nx,1:nx])
            if cont
                h = 1
                e = log.(Complex.(e))./h
            end
            lim = max(maximum(imag.(e)), lim)
            real.(e), imag.(e)
        end
        # scatter!(real.(e), imag.(e), c=:blue, show=false, subplot=2, legend=false)
    end
    delete!(plotattributes, :ds)
    phi = linspace(0,2π,300)
    if !cont
        ratio --> :equal
        @series begin
            legend --> false; color --> :black; linestyle --> :dash
            real.(exp.(phi.*im)), imag.(exp.(phi.*im))
        end
    else
        @series (legend := false; linestyle := :dash; color := :black; ([0,0],[-lim,lim]))
    end
    delete!(plotattributes, :cont)
    nothing
end

function display_modeltrainer(mt::ModelTrainer; kwargs...)
    ms = mt.models
    modeldict = OrderedDict("All models" => 0, ["Model "*string(i) => i for i = 1:length(mt.models)]...)

    ui = @manipulate for t = togglebuttons(1:length(mt.trajs), selected=1),#, label="Trajectory number", value=1),
                         mn = togglebuttons(modeldict, selected=1),
                         modelversion = slider(1:length(mt.modelhistory), value=length(mt.modelhistory), label="Model state"),
                         f = slider(0:100, label="Filtering", value=0),
                         ds = slider(1:100, value=4, label="Downsampling"),
                        eigvals = [true,false],
                        cont = [true,false]

        ms = mt.modelhistory[modelversion]

        modelinds = (mn <= 0 || mn > length(ms)) ? (1:length(ms)) : (mn:mn)

        if eigvals
            eigvalplot(ms[modelinds], mt.trajs[t], ds=ds, cont=cont, kwargs...)
        else
            plot(mt.trajs[t]; filtering=f, kwargs...)
        end
    end
end

end # module
