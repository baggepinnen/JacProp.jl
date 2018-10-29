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
    m  = Chain(Dense(nx+nu,np, activation), Dense(np,np, activation), Dense(np, ny))
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
    m  = Chain(Dense(nx+nu,np, activation), Dense(np,np, activation), Dense(np, ny))
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

function pred(w,x,nx)
    state = x
    for i=1:2:length(w)-2
        state = tanh.(w[i]*state .+ w[i+1])
    end
    return w[end-1]*state .+ w[end]
end
pred(m,x) = pred(m.w,x,m.nx)
predd(m,x) = predd(m.w,x,m.nx)
predd(w,x,nx) = pred(w,x,nx) .+ x[1:nx,:]

function pred_jac(w,x,nx)
    l = x
    J = Matrix{eltype(w[1])}(I,length(x),length(x))
    for i = 1:2:length(w)-2
        W,b = w[i], w[i+1]
        l   = W*l .+ b
        ∇σ  = ∇tanh(l)
        ∇a  = W
        l  .= tanh.(l) # Reused l to save allocations
        J   = ∇σ * ∇a * J
    end
    J = w[end-1] * J # Linear output layer
    # J += [Matrix{Float64}(I, nx, nx) zeros(nx)] # Not needed since we'll only use the difference
    # @assert isapprox(J, hcat(fdjac(x->predd(w,x,nx),x)...), atol=1e-4)
    return w[end-1]*l .+ w[end] , J#.+ x[1:nx,:], J
end
∇tanh(x) = Matrix(Diagonal((sech.(x).^2)[:]))
∇relu(x) = Matrix(Diagonal(vec(ifelse.(x .<= 0, 0., 1.))))
function ∇swish!(x)
    σx = σ.(x)
    x .= x.*σx
    σx .= x .+ σx .* (1 .- x)
    Matrix(Diagonal(vec(σx)))
end


@with_kw struct ADSystem <: AbstractSystem
    w::NTuple{6, Matrix{Float64}}
    sizes = ((num_params,nx+nu), (num_params,1), (nx,num_params), (nx,1))
    nx::Int
    nu::Int
    h::Float64 = 1.0
end

@with_kw struct ADDiffSystem <: AbstractDiffSystem
    w::NTuple{6, Matrix{Float64}}
    sizes = ((num_params,nx+nu), (num_params,1), (nx,num_params), (nx,1))
    nx::Int
    nu::Int
    h::Float64 = 1.0
end
function ADSystem(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    sizes = ((num_params,nx+nu), (num_params,1), (num_params,num_params), (num_params,1), (nx,num_params), (nx,1))
    w = ( Flux.glorot_uniform(sizes[1]...), zeros(Float64,sizes[2]),Flux.glorot_uniform(sizes[3]...),  zeros(Float64,sizes[4]), Flux.glorot_uniform(sizes[5]...),  zeros(Float64,sizes[6]) )
    model(x)    = pred(w,x,nx)
    ADSystem(w,sizes,nx,nu,h)
end
function ADDiffSystem(nx::Int,nu::Int, num_params::Int, activation::Function, h=1)
    sizes = ((num_params,nx+nu), (num_params,1), (num_params,num_params), (num_params,1), (nx,num_params), (nx,1))
    w = ( Flux.glorot_uniform(sizes[1]...), zeros(Float64,sizes[2]),Flux.glorot_uniform(sizes[3]...),  zeros(Float64,sizes[4]), Flux.glorot_uniform(sizes[5]...),  zeros(Float64,sizes[6]) )
    model(x)    = predd(w,x,nx)
    ADDiffSystem(w,sizes,nx,nu,h)
end
(m::ADSystem)(x) = pred(m, x)
(m::ADSystem)(w,x) = pred(w, x, m.nx)
(m::ADDiffSystem)(x) = predd(m, x)
(m::ADDiffSystem)(w,x) = predd(w, x, m.nx)

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

# cost(w,x,y)  = sum(abs2, pred(w,x) .- y)/size(y,2)
function cost(f,x,y)
    yh = f(x)
    sum(abs2, yh .- y)/size(y,2) #+ 0.1sum(norm.(w).^2)
end
# cost(w,data) = sum(cost(w, d...) for d in data)/length(data)
cost(f,data) = cost(f, data...)

# WARNING: don't assign to any vector with .= in the inner loss function closure
function loss(w,x,y,mt::ADModelTrainer)
    model = mt.model
    nx, nu = model.nx, model.nu
    predfun = model isa ADSystem ? pred : predd
    function lf(w...)
        # println("Entering loss function, typeof(w): ", typeof(w), " length(w): ", length(w))
        @unpack λ,normalizer = mt
        f(x)   = predfun(w,x,nx)
        l      = cost(f,x,y)
        jac(x) = fdjac(f,x)
        J1 = jac(x[:,1])
        # sd = fill(typeof(w[1])(0.),size(J1))
        for t = 2:size(x,2)
            J2 = jac(x[:,t])
            for i in eachindex(J1)
                l += λ*sum(abs2.(J1[i].-J2[i]))
            end
            # copy!(J1,J2)
            J1 = J2
        end
        l
    end
end

predjac(mt::ADModelTrainer{<:ADSystem,<:Any}) = pred_jac
predjac(mt::ADModelTrainer{<:ADDiffSystem,<:Any}) = function (w,x,nx)
    y,J = pred_jac(w,x,nx)
    y + x[1:nx,:], J
end

function loss2(w,x,y,mt::ADModelTrainer)
    model = mt.model
    nx, nu = model.nx, model.nu
    pj = predjac(mt)
    function lf(w...)
        @unpack λ,normalizer = mt
        yh, J1 = pj(w,x[:,1],nx)
        l = sum(abs2, yh .- y[:,1])
        for t = 2:size(x,2)
            yh, J2 = pj(w,x[:,t],nx)
            l += sum(abs2, yh .- y[:,t])/size(y,2) + λ*sum(abs2.(J1.-J2))
            # copy!(J1,J2)
            J1 = J2
        end
        # l += λ/2*norm(w[1])^2 + λ/2*norm(w[2])^2
        # l += λ*maximum(abs.(w[3]*w[1]))#λ*norm(w[1])^2 + λ*norm(w[2])^2
        l
    end
end


function fdjac(f::Function,x::AbstractVector,epsilon=sqrt(eps()))
    n = length(x)
    f_x = f(x)
    shifted_x = copy(x)
    map(eachindex(x)) do i
        shifted_x[i] += epsilon
        J = @fastmath (f(shifted_x) .- f_x) ./ epsilon
        shifted_x[i] = x[i]
        J
    end
end


function find_normalizer(w,data,mt::ADModelTrainer)
    x,y = data
    model = mt.model
    sizes, nx, nu = model.sizes, model.nx, model.nu
    f(x)          = predd(w,x,nx)
    jcfg          = Diff.JacobianConfig(f, x[:,1])
    jacobian(x) = Diff.jacobian(f, x, jcfg)
    mJ = zeros(nx,nx+nu)
    for t = 1:size(x,2)
        J = jacobian(x[:,t])
        mJ .= max.(abs.(J), mJ)
    end
    mJ
end

function i2m(w,i,sizes)
    s = [1; cumsum([prod.(sizes)...]) .+ 1]
    (w,i)->reshape(view(w,s[i]:(s[i+1]-1)), sizes[i])
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
    ms isa Union{ADSystem,ADDiffSystem} || Flux.testmode!.(ms, testmode)
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
    data(mean(y)), extrema(extrema.(y))
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
    dropdims(mean(jacmat, dims=3), dims=3), dropdims(std(jacmat, dims=3), dims=3)
end

function Flux.jacobian(ms::EnsembleVelSystem, x::AbstractArray, testmode=true)
    Flux.testmode!.(ms, testmode)
    h = ms[1].h
    jacs = [[[I h*I h^2/2*eye(2)];Flux.jacobian(m,x)] for m in ms] # The h²/2*I in ∇ᵤ is an approximation since there are (very small) cross terms.
    jacmat = smartcat3(jacs)
    squeeze(mean(jacmat, 3), 3), squeeze(std(jacmat, 3), 3)
end

function Flux.jacobian(m::ADSystem, x)
    f(x) = pred(m.w, x, m.nx)
    j(x) = hcat(fdjac(f,x)...)
    j(x), nothing
end
function Flux.jacobian(m::ADDiffSystem, x)
    f(x) = predd(m.w, x, m.nx)
    j(x) = hcat(fdjac(f,x)...)
    j(x), nothing
end


Flux.jacobian(ms, t::Trajectory; kwargs...) = Flux.jacobian(ms, t.xu; kwargs...)

models(mt::ModelTrainer) = mt.models
models(mt::ADModelTrainer) = mt.model
models(systems::Vector{<:AbstractSys}) = getfield.(systems, :m)
models(results::AbstractVector) = [r[:m] for r in results]
models(results::AbstractDict) = [r[:m]]
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
    At = cat([J[i][1][:,1:t.nx] for i = 1:length(t)]..., dims=3)
    Bt = cat([J[i][1][:,t.nx+1:end] for i = 1:length(t)]..., dims=3)
    Pt = cat([diagm(J[i][2][:]).^2 for i = 1:length(t)]..., dims=3)
    LTVModels.KalmanModel(At,Bt,Pt,true)
end

function LTVModels.KalmanModel(ms::AbstractSys, t::Trajectory)
    xu = t.xu
    N  = size(xu,2)
    J  = map(1:length(t)) do i
        jacobian(ms, xu[:,i])
    end
    At = cat([J[i][:,1:t.nx] for i = 1:length(t)]..., dims=3)
    Bt = cat([J[i][:,t.nx+1:end] for i = 1:length(t)]..., dims=3)
    Pt = cat([eye(length(J[1])) for i = 1:length(t)]..., dims=3)
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

function eval_jac2(trainer::AbstractModelTrainer, vt, truejacfun, ds=1)
    m = models(trainer)
    nx = vt.nx
    mean(1:ds:length(vt)) do i
        J1, J2 = jacobian(m,vt.xu[:,i])[1], truejacfun(vt.x[:,i],vt.u[:,i])
        err = 0mean(abs2.(J1[:,nx+1:end] .- J2[:,nx+1:end]))
        e1,e2 = eigvals(J1[1:nx,1:nx]), eigvals(J2[1:nx,1:nx])
        dm = abs2.(e1 .- e2')
        dists = max.(minimum(dm, dims=1), minimum(dm, dims=2))
        err  += sum(dists)
        err
    end |> √
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
    foreach(T->treelike(JacProp, T), [System, DiffSystem, StabilizedDiffSystem, NominalDiffSystem, VelSystem, RecurrentSystem, RecurrentDiffSystem])
catch err
    @error(err)
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
