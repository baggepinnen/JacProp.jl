export AbstractSys, AbstractSystem, AbstractDiffSystem, AbstractVelSystem
export System, DiffSystem, StabilizedDiffSystem, NominalDiffSystem, VelSystem, RecurrentSystem, RecurrentDiffSystem
export simulate, predict, jacobians

abstract type AbstractSys <: LTVModelsBase.AbstractModel end
abstract type AbstractSystem <: AbstractSys end
abstract type AbstractDiffSystem <: AbstractSys end
abstract type AbstractVelSystem <: AbstractSys end

import LTVModelsBase: simulate, predict

@with_kw struct System{T} <: AbstractSystem
    m::T
    nx::Int
    nu::Int
    h::Float64 = 1.0
end
function System(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    ny = nx
    np = num_params
    m  = Chain(Dense(nx+nu,np, activation), Dense(np, ny))
    System(m, nx, nu, h)
end
(m::System)(x) = m.m(x)


@with_kw struct DiffSystem{T} <: AbstractDiffSystem
    m::T
    nx::Int
    nu::Int
    h::Float64 = 1.0
end
function DiffSystem(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    ny = nx
    np = num_params
    m  = Chain(Dense(nx+nu,np, activation), Dense(np, ny))
    DiffSystem(m, nx, nu, h)
end
(m::DiffSystem)(x) = m.m(x)+x[1:m.nx,:]

@with_kw struct StabilizedDiffSystem{T} <: AbstractDiffSystem
    τ::Float64
    m::T
    nx::Int
    nu::Int
    h::Float64 = 1.0
end
function StabilizedDiffSystem(τ::Float64,nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    ny = nx
    np = num_params
    m  = Chain(Dense(nx+nu,np, activation), Dense(np, ny))
    StabilizedDiffSystem(τ, m, nx, nu, h)
end
(m::StabilizedDiffSystem)(x) = m.m(x)+ m.τ*x[1:m.nx,:]

@with_kw struct NominalDiffSystem{T} <: AbstractDiffSystem
    A::Matrix{Float64}
    B::Matrix{Float64}
    m::T
    nx::Int
    nu::Int
    h::Float64 = 1.0
end
function NominalDiffSystem(A::AbstractMatrix,B::AbstractMatrix,nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    ny = nx
    np = num_params
    m  = Chain(Dense(nx+nu,np, activation), Dense(np, ny))
    NominalDiffSystem(A, B, m, nx, nu, h)
end
(m::NominalDiffSystem)(x) = m.m(x) .+ m.A*x[1:m.nx,:] .+ m.B*x[1+m.nx:end,:]


@with_kw struct VelSystem{T} <: AbstractVelSystem
    m::T
    nx::Int
    nu::Int
    h::Float64 = 1.0
end
function VelSystem(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    ny = nx
    np = num_params
    m = Chain(Dense(nx+nu,np, activation),  Dense(np, ny))
    VelSystem(m, nx, nu, h)
end
(m::VelSystem)(x) = m.m(x)


# Recurrent systems ========================================================================
@with_kw struct RecurrentSystem{T} <: AbstractSystem
    m::T
    nx::Int
    nu::Int
    h::Float64 = 1.0
end
function RecurrentSystem(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    ny = nx
    np = num_params
    m  = Chain(Dense(nx+nu,np, activation), RNN(np,np), Dense(np,np,activation), Dense(np, ny))
    RecurrentSystem(m, nx, nu, h)
end
(m::RecurrentSystem)(x) = m.m(x)


@with_kw struct RecurrentDiffSystem{T} <: AbstractDiffSystem
    m::T
    nx::Int
    nu::Int
    h::Float64 = 1.0
end
function RecurrentDiffSystem(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    ny = nx
    np = num_params
    m  = Chain(Dense(nx+nu,np, activation), RNN(np,np), Dense(np,np,activation), Dense(np, ny))
    RecurrentDiffSystem(m, nx, nu, h)
end
(m::RecurrentDiffSystem)(x) = m.m(x)+x[1:m.nx,:]

# ==========================================================================================

loss(m::AbstractSys) = (x,y) -> sum((m(x).-y).^2)/size(x,2)

const AbstractEnsembleSystem = Vector{T} where T <: AbstractSys
const EnsembleSystem = Vector{T} where T <: AbstractSystem
const EnsembleDiffSystem = Vector{T} where T <: AbstractDiffSystem
const EnsembleVelSystem = Vector{T} where T <: AbstractVelSystem

xy(::Type{<:AbstractSys}, x, N)        = (x[:,1:N], x[:,2:N+1])
xy(::Type{<:AbstractVelSystem}, x, N)  = (x[:,1:N], x[3:4,2:N+1]) # TODO magic numbers


function get_minimum_loss(results, key)
    array = getindex.(get.(results[key]),2)
    return isempty(array) ? Inf : minimum(array)
end

function simulate(ms::AbstractEnsembleSystem,x::AbstractArray, testmode=true)
    Flux.testmode!.(ms, testmode)
    xsim = copy(x)
    ns = ms[1].nx
    for t = 2:size(x,2)
        @ensemble xsimt = ms(xsim[:,t-1])
        xsim[1:ns,t] = mean(xsimt).data
    end
    xsim[1:ns,:]
end

function simulate(ms::EnsembleVelSystem,x::AbstractArray, testmode=true)
    Flux.testmode!.(ms, testmode)
    xsim = copy(x)
    for t = 2:size(x,2)
        @ensemble xsimt = ms(xsim[:,t-1])
        h = ms[1].h
        xsim[1:2,t] = xsim[1:2,t-1] + h*xsim[3:4,t-1] # TODO magic numbers
        xsim[3:4,t] = mean(xsimt).data # TODO magic numbers
    end
    xsim[1:4,:]
end

simulate(ms, t::Trajectory; kwargs...) = simulate(ms, t.xu; kwargs...)

function predict(ms::AbstractEnsembleSystem, x::AbstractMatrix, testmode=true)
    Flux.testmode!.(ms, testmode)
    @ensemble y = ms(x)
    mean(y).data, getfield.(extrema(y), :data)
end

function predict(ms::EnsembleVelSystem, x::AbstractArray, testmode=true)
    Flux.testmode!.(ms, testmode)
    @ensemble y = ms(x)
    yh = mean(y).data
    h = ms[1].h
    yh = [x[1:2,1] .+ h*cumsum(yh,2); yh] # TODO magic numbers
    bounds = getfield.(extrema(y), :data)
    yh, bounds
end

predict(ms, t::Trajectory; kwargs...) = predict(ms, t.xu; kwargs...)

function Flux.jacobian(ms::AbstractEnsembleSystem, x::AbstractArray, testmode=true)
    Flux.testmode!.(ms, testmode)
    # @show Flux.Tracker.grad(params.(ms[1]))
    # Flux.Tracker.zero_grad!.(params.(ms))
    jacs = [Flux.jacobian(m,x) for m in ms]
    # Flux.Tracker.zero_grad!.(params.(ms))
    jacmat = smartcat3(jacs)
    squeeze(mean(jacmat, 3), 3), squeeze(std(jacmat, 3), 3)
end

function Flux.jacobian(ms::EnsembleVelSystem, x::AbstractArray, testmode=true)
    Flux.testmode!.(ms, testmode)
    h = ms[1].h
    jacs = [[[I h*I h^2/2*eye(2)];Flux.jacobian(m,x)] for m in ms] # The h²/2*I in ∇ᵤ is an approximation since there are (very small) cross terms.
    jacmat = smartcat3(jacs)
    squeeze(mean(jacmat, 3), 3), squeeze(std(jacmat, 3), 3)
end

Flux.jacobian(ms, t::Trajectory; kwargs...) = Flux.jacobian(ms, t.xu; kwargs...)

models(mt::ModelTrainer) = getfield.(mt.models, :m)
models(systems::Vector{<:AbstractSys}) = getfield.(systems, :m)
models(results::AbstractVector) = [r[:m] for r in results]
models(results::Associative) = [r[:m]]

function LTVModels.KalmanModel(ms::AbstractEnsembleSystem, t::Trajectory)
    xu = t.xu
    N  = size(xu,2)
    J  = map(1:length(t)) do i
        Jm, Js = jacobian(ms, xu[:,i])
        Jm, Js
    end
    At = cat(3,[J[i][1][:,1:t.nx] for i = 1:length(t)]...)
    Bt = cat(3,[J[i][1][:,t.nx+1:end] for i = 1:length(t)]...)
    Pt = cat(3,[diagm(J[i][2][:]).^2 for i = 1:length(t)]...)
    LTVModels.KalmanModel(At,Bt,Pt,true)
end


function jacobians(ms, t, ds=1)
    msc = deepcopy(ms) # Obs, this is to not fuck up the gradients of the model parameters ??
    xu = t.xu
    N = size(xu,2)
    J = map(1:ds:N) do evalpoint
        Jm, Js = jacobian(msc, xu[:,evalpoint])
        Jm[:], Js[:]
    end
    Jm = smartcat2(getindex.(J,1))
    Js = smartcat2(getindex.(J,2))
    Jm, Js
end

function eval_pred(results, eval=false)
    @unpack x,u,y,modeltype = results[1]
    ms = models(results)
    nx = ms[1].nx
    yh, bounds = predict(ms, x)
    pred = rms(x[1:nx,2:end].-yh[:,1:end-1]) # TODO: check time alignment
    pred
end

function eval_sim(results, eval=false)
    @unpack x,u,y,modeltype = results[1]
    ms = models(results)
    nx = ms[1].nx
    xh = simulate(ms, x)
    sim = rms(x[1:nx,:].-xh) # TODO: check time alignment
    sim
end

function eval_jac(Jm, Jtrue)
    sqrt(mean(abs2, Jm .- Jtrue))
end

function plotresults(results, eval=false)
    @unpack x,u,y,modeltype = results[1]
    ms = models(results)
    nx = ms[1].nx
    fig = plot(x[1:nx,2:end]', lab="True", layout=nx)
    testmode!.(ms, true)
    plot_prediction(fig, results, eval)
    plot_simulation(fig, results, eval)
    testmode!.(ms, false)
    fig
end

get_res(res,n) = getindex.(res,n)

try
foreach(treelike, [System, DiffSystem, StabilizedDiffSystem, NominalDiffSystem, VelSystem, RecurrentSystem, RecurrentDiffSystem])
end

function smartcat2(vv)
    dim2 = length(vv)
    dim1 = length(vv[1])
    A = zeros(dim1,dim2)
    for i in eachindex(vv)
        A[:,i] = vv[i]
    end
    A
end

function smartcat3(vv)
    dim3 = length(vv)
    dim1,dim2 = size(vv[1])
    A = zeros(dim1,dim2,dim3)
    for i in eachindex(vv)
        A[:,:,i] = vv[i]
    end
    A
end
