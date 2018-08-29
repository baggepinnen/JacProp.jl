export AbstractSys, AbstractSystem, AbstractDiffSystem, AbstractVelSystem
export System, DiffSystem, StabilizedDiffSystem, NominalDiffSystem, VelSystem, RecurrentSystem, RecurrentDiffSystem
export simulate, predict, jacobians

abstract type AbstractSys <: LTVModelsBase.AbstractModel end
abstract type AbstractSystem <: AbstractSys end
abstract type AbstractDiffSystem <: AbstractSys end
abstract type AbstractVelSystem <: AbstractSys end

import LTVModelsBase: simulate, predict
import Flux.Tracker.data

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

# Non-flux systems =========================================================================
tovec(w::Vector{<:Matrix}) = vcat([vec(w) for w in w]...)

function pred(w,x,sizes)
    state = copy(x)
    for i=1:2:length(sizes)-2
        state = tanh.(i2m(w,i,sizes)*state .+ i2m(w,i+1,sizes))
    end
    return i2m(w,length(sizes)-1,sizes)*state .+ i2m(w,length(sizes),sizes)
end
function predd(w,x,sizes,nx)
    state = copy(x)
    for i=1:2:length(sizes)-2
        state = tanh.(i2m(w,i,sizes)*state .+ i2m(w,i+1,sizes))
    end
    return i2m(w,length(sizes)-1,sizes)*state .+ i2m(w,length(sizes),sizes)# .+ x[1:nx,:]
end
pred(m,x) = pred(m.w,x,m.sizes)
predd(m,x) = predd(m.w,x,m.sizes,m.nx)

@with_kw struct ADSystem{JT} <: AbstractSystem
    w::Vector{Float64}
    sizes = ((num_params,nx+nu), (num_params,1), (nx,num_params), (nx,1))
    nx::Int
    nu::Int
    h::Float64 = 1.0
    jacobian::JT
end
function ADSystem(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    sizes = ((num_params,nx+nu), (num_params,1), (nx,num_params), (nx,1))
    w = [ Flux.initn(sizes[1]), zeros(Float64,sizes[2]),Flux.initn(sizes[3]),  zeros(Float64,sizes[4]) ]
    wd          = tovec(w)
    model(x)    = pred(m,x)
    jfg         = Diff.JacobianConfig(model, zeros(nx+nu))
    jacobian(x) = Diff.jacobian(model, x)
    ADSystem(wd,sizes,nx,nu,h,jacobian)
end
(m::ADSystem)(x) = pred(m, x)
(m::ADSystem)(w,x) = pred(w, x, m.sizes)

@with_kw struct ADDiffSystem{JT} <: AbstractDiffSystem
    w::Vector{Float64}
    sizes = ((num_params,nx+nu), (num_params,1), (nx,num_params), (nx,1))
    nx::Int
    nu::Int
    h::Float64 = 1.0
    jacobian::JT
end
function ADDiffSystem(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    sizes = ((num_params,nx+nu), (num_params,1), (nx,num_params), (nx,1))
    w = [ Flux.initn(sizes[1]), zeros(Float64,sizes[2]),Flux.initn(sizes[3]),  zeros(Float64,sizes[4]) ]
    wd          = tovec(w)
    model(x)    = predd(wd,x,sizes,nx)
    jfg         = Diff.JacobianConfig(model, zeros(nx+nu))
    jacobian(x) = Diff.jacobian(model, x)
    ADDiffSystem(wd,sizes,nx,nu,h,jacobian)
end
(m::ADDiffSystem)(x) = predd(m, x)
(m::ADDiffSystem)(w,x) = predd(w, x, m.sizes, m.nx)

Flux.testmode!(m::ADSystem, b=false) = nothing
Flux.testmode!(m::ADDiffSystem, b=false) = nothing

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

# cost(w,x,y)  = sum(abs2, pred(w,x,sizes) .- y)/size(y,2)
cost(w,sizes,nx,x,y)  = sum(abs2, predd(w,x,sizes,nx) .- y)/size(y,2) #+ 0.01sum(vecnorm.(w))
# cost(w,data) = sum(cost(w, d...) for d in data)/length(data)
cost(w,sizes,nx,data) = cost(w,sizes,nx, data...)

# WARNING: don't assign to any vector with .= in the inner loss function closure
function loss(w,x,y,mt::ADModelTrainer{<:ADDiffSystem,<:Any})
    chunk = Diff.Chunk(x[:,1])
    model = mt.model
    sizes, nx, nu = model.sizes, model.nx, model.nu
    function lf(w)
        # println("Entering loss function, typeof(w):", typeof(w))
        λ = mt.λ
        f(x)          = predd(w,x,sizes,nx)
        jcfg          = Diff.JacobianConfig(f, x[:,1], chunk)
        l             = cost(w,sizes,nx,x,y)
        jacobian(x) = Diff.jacobian(f, x, jcfg)
        J1 = jacobian(x[:,1])
        md = zeros(J1); sd = zeros(J1)
        for t = 2:size(x,2)
            J2 = jacobian(x[:,t])
            d = abs2.(J1.-J2)
            sd .+= d
            md .= max.(d, md)
            # copy!(J1,J2)
            J1 = J2
        end
        l += λ*sum(sd./md)
        l
    end
end

function i2m(w,i,sizes)
    s = [1; cumsum([prod.(sizes)...]) .+ 1]
    reshape(view(w,s[i]:(s[i+1]-1)), sizes[i])
end


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

function simulate(ms::AbstractEnsembleSystem,xu::AbstractMatrix, testmode=true)
    Flux.testmode!.(ms, testmode)
    xsim = copy(xu)
    nx = _nx(ms)
    for t = 2:size(xu,2)
        @ensemble xsimt = ms(xsim[:,t-1])
        xsim[1:nx,t] = data(mean(xsimt))
    end
    xsim[1:nx,:]
end

function simulate(ms::AbstractSys,xu::AbstractMatrix, testmode=true)
    ms isa ADDiffSystem || Flux.testmode!.(ms, testmode)
    xsim = copy(xu)
    nx = _nx(ms)
    for t = 2:size(xu,2)
        xsimt = ms(xsim[:,t-1])
        xsim[1:nx,t] = xsimt
    end
    xsim[1:nx,:]
end

function simulate(ms::EnsembleVelSystem,xu::AbstractMatrix, testmode=true)
    Flux.testmode!.(ms, testmode)
    xsim = copy(xu)
    for t = 2:size(xu,2)
        @ensemble xsimt = ms(xsim[:,t-1])
        h = ms[1].h
        xsim[1:2,t] = xsim[1:2,t-1] + h*xsim[3:4,t-1] # TODO magic numbers
        xsim[3:4,t] = data(mean(xsimt)) # TODO magic numbers
    end
    xsim[1:4,:]
end

simulate(ms::Union{AbstractSys,AbstractEnsembleSystem}, t::Trajectory; kwargs...) = simulate(ms, t.xu; kwargs...)

function predict(ms::Vector, x::AbstractMatrix, testmode=true)
    Flux.testmode!.(ms, testmode)
    @ensemble y = ms(x)
    data(mean(y)), data.(extrema(y))
end

predict(ms, x::AbstractMatrix, testmode=true) = ms(x), nothing
function predict(ms::EnsembleVelSystem, x::AbstractArray, testmode=true)
    Flux.testmode!.(ms, testmode)
    @ensemble y = ms(x)
    yh = data(mean(y))
    h = ms[1].h
    yh = [x[1:2,1] .+ h*cumsum(yh,2); yh] # TODO magic numbers
    bounds = data.(extrema(y))
    yh, bounds
end

predict(ms::Union{AbstractSys,AbstractEnsembleSystem}, t::Trajectory; kwargs...) = predict(ms, t.xu; kwargs...)

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

function Flux.jacobian(m::Union{ADSystem,ADDiffSystem}, x)
    m.jacobian(x), nothing
end


Flux.jacobian(ms, t::Trajectory; kwargs...) = Flux.jacobian(ms, t.xu; kwargs...)

models(mt::ModelTrainer) = mt.models
models(mt::ADModelTrainer) = mt.model
models(systems::Vector{<:AbstractSys}) = getfield.(systems, :m)
models(results::AbstractVector) = [r[:m] for r in results]
models(results::Associative) = [r[:m]]
_nx(s::AbstractSys) = s.nx
_nx(v::AbstractVector) = _nx(v[1])
_nx(trainer::ModelTrainer) = _nx(trainer.models)
_nx(trainer::ADModelTrainer) = _nx(trainer.model)

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


function jacobians(ms::AbstractVector, t, ds=1)
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

function jacobians(ms, t, ds=1)
    xu = t.xu
    N = size(xu,2)
    J = map(1:ds:N) do evalpoint
        Jm, Js = jacobian(ms, xu[:,evalpoint])
        Jm[:]
    end
    Jm = smartcat2(J)
    Jm, nothing
end


function eval_pred(trainer::AbstractModelTrainer, vt)
    @unpack x,u,y = vt
    ms = models(trainer)
    nx = _nx(trainer)
    yh, bounds = predict(ms, vt.xu)
    pred = LTVModelsBase.rms(x[:,2:end].-yh[:,1:end-1]) # TODO: check time alignment
    pred
end

function eval_sim(trainer::AbstractModelTrainer, vt)
    @unpack x,u,y,xu = vt
    ms = models(trainer)
    nx = _nx(trainer)
    xh = simulate(ms, xu)
    sim = LTVModelsBase.rms(x[:,:].-xh) # TODO: check time alignment
    sim
end

function eval_jac(trainer::AbstractModelTrainer, vt, truejacfun, ds=1)
    m = models(trainer)
    mean(i-> mean(abs2.(jacobian(m,vt.xu[:,i])[1] .- truejacfun(vt.x[:,i],vt.u[:,i]))), 1:ds:length(vt)) |> √
end



function plotresults(trainer::AbstractModelTrainer, vt)
    @unpack x,u,y = vt
    ms = models(trainer)
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
