module JacProp

export Trajectory, default_activations, ModelTrainer, ADModelTrainer, sample_jacprop, push!, train!, display_modeltrainer, getter

using LTVModelsBase, Parameters, Reexport, Lazy, Juno, FunctionEnsembles
@reexport using LTVModels, Flux, ValueHistories, IterTools, MLDataUtils
using Flux: back!, truncate!, treelike, train!, mse, testmode!, params, jacobian, throttle
using Flux.Optimise: Param, optimiser, RMSProp, expdecay
using Plots, DataStructures, ForwardDiff, ReverseDiff#, InteractNext, Observables
const Diff = ForwardDiff
const RDiff = ReverseDiff
const default_activations = [swish, Flux.sigmoid, tanh, elu]
const IT = IterTools

abstract type AbstractModelTrainer end


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
@with_kw mutable struct ModelTrainer{cbT,tT} <: AbstractModelTrainer
    models
    opts
    losses
    R1                = I
    R2                = 10_000I
    P0                = 10_000I
    σdivider::Float64 = 10.
    P::Float64        = 1.0
    trajs::Vector{Trajectory} = Trajectory[]
    cb::cbT
    modelhistory = []
    trace::tT = History(Float64)
end

@with_kw mutable struct ADModelTrainer{mT,tT} <: AbstractModelTrainer
    model::mT
    opt
    λ::Float64 = 1.
    testdata::Trajectory
    trajs::Vector{Trajectory} = Trajectory[]
    modelhistory = []
    trace::tT = History(Float64)
    tracev::tT = History(Float64)
    normalizer::Matrix{Float64} = ones(1,1)
end


include("utilities.jl")

to_callback(cb,args...) = methods(cb).mt.max_args > 1 ? cb(args...) : cb

"""
train!(mt::ModelTrainer; epochs=1, jacprop=0, useprior=true, trace = mt.trace)

See also [`ModelTrainer`](@ref)
"""
function Flux.train!(mt::ModelTrainer; epochs=1, jacprop=0, useprior=true, trace = mt.trace)
    @assert !isempty(mt.trajs) "No data in ModelTrainer"
    @unpack models,opts,losses = mt
    data1 = todata(mt)
    jacprop > 0 && (ltvmodels = fit_models(mt, useprior))
    startepoch = length(mt.trace)+1
    @progress for epoch = startepoch:startepoch+epochs-1
        data = if jacprop > 0
            data2 = [todata(sample_jacprop(mt, ltvmodels)) for i = 1:jacprop]
            chain(data1, data2...)
        else
            data1
        end
        for (model,loss, opt) in zip(models,losses,opts)
            train!(loss, data, opt, cb=to_callback(mt.cb, epoch,loss,data1, trace, model, mt))
        end
    end
    push!(mt.modelhistory, deepcopy(models))
    trace
end

function Flux.train!(mt::ADModelTrainer; epochs=1, jacprop=0, trace = mt.trace, cb=()->())
    @assert !isempty(mt.trajs) "No data in ModelTrainer"
    @unpack model,opt,testdata,λ,trace,tracev = mt
    # mt.λ = 0.
    if epochs <= 0
        push!(mt.modelhistory, deepcopy(model))
        return
    end
    w          = model.w
    data       = todata(mt)
    datat      = todata([testdata])
    mt.normalizer = ones(size(data[1][2], 1),size(data[1][1], 1))
    losses = map(data) do d
        x,y        = d
        lossfun    = loss(w,x,y,mt)
        # gcfg       = RDiff.GradientConfig(w)
        GC.gc()
        println("Compiling tape ")
        @time tape = RDiff.GradientTape(lossfun, w) #|> RDiff.compile
        println(" Done")
        increment!(trace, 1, lossfun(w...))
        increment!(tracev, 1, cost(w,model.sizes,model.nx,datat))
        lossfun, tape
    end
    g          = similar.(w)
    # epochs > 0 && plot(reuse=false)
    startepoch = last(trace)[1]+1
    @progress for epoch = startepoch:startepoch+epochs-1
        # epoch == 500 && (mt.λ = λ)
        # lossfun = loss(w,x,y,mt)
        for (i,(lossfun,tape)) in enumerate(losses)
            RDiff.gradient!(g,tape,w)
            for (opt,g) in zip(opt,g)
                opt(g, epoch)
            end
            if epoch % 5 == 0
                # mt.normalizer = find_normalizer(w,data[i],mt)
                # increment!(trace, epoch, lossfun(w...))
                # increment!(tracev, epoch, cost(w,model.sizes,model.nx,datat))
                # if i == length(losses) && myid() == 1
                #     cb(model)
                #     # plot(trace, reuse=true)
                #     # plot!(tracev)
                #     # gui()
                #     println("Losses: ", last(trace), last(tracev)[2])
                # end
            end
        end
    end
    push!(mt.modelhistory, deepcopy(model))
    trace, tracev
end

function fit_models(mt::ModelTrainer, useprior)
    map(mt.trajs) do traj
        ltvmodel = KalmanModel(mt, traj, useprior=useprior)
    end
end

function sample_jacprop(mt::ModelTrainer, ltvmodels)
    perturbed_trajs = map(mt.trajs, ltvmodels) do traj, ltvmodel
        @unpack x,u,nx,nu = traj
        xa = x .+ std(x,2)/mt.σdivider .* randn(size(x))
        ua = u .+ std(u,2)/mt.σdivider .* randn(size(u))
        ya = LTVModels.predict(ltvmodel, xa, ua)
        Trajectory(xa,ua,ya)
    end
end

function LTVModels.KalmanModel(mt::ModelTrainer, t::Trajectory, ms=mt.models;
    P=mt.P, useprior=!isempty(mt.modelhistory))
    useprior = useprior && !isempty(mt.modelhistory)
    @unpack x,u,nx,nu = t
    @unpack R1,R2,P0  = mt
    T                 = length(t)
    if useprior
        model    = KalmanModel(zeros(nx,nx,T),zeros(nx,nu,T),zeros(1,1,T),false)
        Jm, Js   = jacobians(ms, t)
        Pt       = cat(3,[diagm(Js[:,i].^2 .+ P) for i=1:T]...) # TODO: magic number
        fx       = cat(3,[reshape(Jm[1:nx^2,i], nx,nx) for i=1:T]...)
        fu       = cat(3,[reshape(Jm[nx^2+1:end,i], nx,nu) for i=1:T]...)
        prior    = KalmanModel(fx,fu,Pt,false)
        ltvmodel = KalmanModel(model, prior, x,u,R1,R2,P0, extend = true, printfit=false)
    else # Don't use prior if it's the first time training
        ltvmodel = KalmanModel(x,u,R1,R2,P0, extend = true, printfit=false)
    end
    ltvmodel
end

todata(trajs::Vector{Trajectory}) = [(traj.xu,traj.y) for traj in trajs]
todata(mt::AbstractModelTrainer) = todata(mt.trajs)
function todata2(trajs::Vector{Trajectory})
    xu = hcat(getfield.(trajs,:xu)...)
    y = hcat(getfield.(trajs,:y)...)
    xu, y
end
todata2(mt::AbstractModelTrainer) = todata2(mt.trajs)

Base.push!(mt::AbstractModelTrainer, t::Trajectory) = push!(mt.trajs, t)
Base.push!(mt::AbstractModelTrainer, data::Matrix...) = push!(mt.trajs, Trajectory(data...))

for T in (ModelTrainer, ADModelTrainer)
    @eval (mt::($T))(; kwargs...) = train!(mt; kwargs...)
    @eval function (mt::($T))(t::Trajectory; kwargs...)
        push!(mt,t)
        train!(mt; kwargs...)
    end
    @eval (mt::($T))(data::Matrix...; kwargs...) = mt(Trajectory(data...); kwargs...)
    @eval Lazy.@forward ($T).models predict, simulate, Flux.jacobian, eigvalplot, jacplot
end

include("plots.jl")

end # module
