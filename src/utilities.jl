import StatsBase.predict

export AbstractSys, AbstractSystem, AbstractDiffSystem, AbstractVelSystem
export System, DiffSystem, VelSystem
export simulate, predict, jacobians

abstract type AbstractSys end
abstract type AbstractSystem <: AbstractSys end
abstract type AbstractDiffSystem <: AbstractSys end
abstract type AbstractVelSystem <: AbstractSys end

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

loss(m::AbstractSys) = (x,y) -> sum((m(x).-y).^2)/size(x,2)

const AbstractEnsembleSystem = Vector{<:AbstractSys}
const EnsembleSystem = Vector{<:AbstractSystem}
const EnsembleDiffSystem = Vector{<:AbstractDiffSystem}
const EnsembleVelSystem = Vector{<:AbstractVelSystem}

xy(::Type{<:AbstractSys}, x, N)        = (x[:,1:N], x[:,2:N+1])
xy(::Type{<:AbstractVelSystem}, x, N)  = (x[:,1:N], x[3:4,2:N+1]) # TODO magic numbers


function get_minimum_loss(results, key)
    array = getindex.(get.(results[key]),2)
    return isempty(array) ? Inf : minimum(array)
end

function simulate(ms::AbstractEnsembleSystem,x, testmode=true)
    Flux.testmode!.(ms, testmode)
    xsim = copy(x)
    ns = ms[1].nx
    for t = 2:size(x,2)
        xsimt = map(m->m(xsim[:,t-1]), ms)
        xsim[1:ns,t] = mean(xsimt).data
    end
    xsim[1:ns,:]
end

function simulate(ms::EnsembleVelSystem,x, testmode=true)
    Flux.testmode!.(ms, testmode)
    xsim = copy(x)
    for t = 2:size(x,2)
        xsimt = map(m->m(xsim[:,t-1]), ms)
        h = ms[1].h
        xsim[1:2,t] = xsim[1:2,t-1] + h*xsim[3:4,t-1] # TODO magic numbers
        xsim[3:4,t] = mean(xsimt).data # TODO magic numbers
    end
    xsim[1:4,:]
end

function StatsBase.predict(ms::AbstractEnsembleSystem, x, testmode=true)
    Flux.testmode!.(ms, testmode)
    y = map(m->m(x), ms)
    mean(y).data, getfield.(extrema(y), :data)
end

function StatsBase.predict(ms::EnsembleVelSystem, x, testmode=true)
    Flux.testmode!.(ms, testmode)
    y = map(m->m(x), ms)
    yh = mean(y).data
    h = ms[1].h
    yh = [x[1:2,1] .+ h*cumsum(yh,2); yh] # TODO magic numbers
    bounds = getfield.(extrema(y), :data)
    yh, bounds
end

function Flux.jacobian(ms::AbstractEnsembleSystem, x, testmode=true)
    Flux.testmode!.(ms, testmode)
    jacs = [Flux.jacobian(m,x) for m in ms]
    jacmat = cat(3,jacs...)
    squeeze(mean(jacmat, 3), 3), squeeze(std(jacmat, 3), 3)
end

function Flux.jacobian(ms::EnsembleVelSystem, x, testmode=true)
    Flux.testmode!.(ms, testmode)
    h = ms[1].h
    jacs = [[[I h*I h^2/2*eye(2)];Flux.jacobian(m,x)] for m in ms] # The h²/2*I in ∇ᵤ is an approximation since there are (very small) cross terms.
    jacmat = cat(3,jacs...)
    squeeze(mean(jacmat, 3), 3), squeeze(std(jacmat, 3), 3)
end

models(mt::ModelTrainer) = getindex.(mt.models, :m)
models(systems::Vector{<:AbstractSys}) = getindex.(systems, :m)
models(results::AbstractVector) = [r[:m] for r in results]
models(results::Associative) = [r[:m]]


function jacobians(results)
    @unpack x,u,modeltype = results[1]
    N = size(x,2)
    J = pmap(1:N) do evalpoint
        Jm, Js = jacobian(models(results), x[:,evalpoint])
        Jm[:], Js[:]
    end
    Jm = hcat(getindex.(J,1)...)
    Js = hcat(getindex.(J,2)...)
    Jm, Js, Jtrue
end


function plot_jacobians(Jm, Js, Jtrue)
    N = size(Jm,1)
    colors = [HSV(h,1,0.8) for h in linspace(0,254,N)]
    plot(Jtrue',lab="True", l=:dash, c=colors', layout=N, legend=false)
    for i = 1:N
        plot!(Jm[i,:], ribbon=2Js[i,:], lab="Estimated", c=colors[i], subplot=i, show=false)
    end
    plot!(repmat([1, N],1,N), [0,0], l=:dash, c=:black, linewidth=2)#, title="Jacobian values and confidence bounds", lab="", legend=false, subplot=1:24)
    # scatter(Jm[:], yerror=2Js[:], lab="Estimated", markerstrokecolor=:match, ms=8)
    # scatter!(Jtrue[:], markershape=:xcross, lab="True", markerstrokecolor=:match, ms=8)
    # plot!([1, prod(size(Jm))], [0,0], l=:dash, c=:black, linewidth=3, title="Jacobian values and confidence bounds", lab="")
end

function LTVModels.plot_eigvals(results, eval=false)
    @unpack x,u,modeltype = results[1]
    N = size(x,2)
    ms = models(results)
    @unpack nx,h = ms[1]
    plot(layout=(2,1), ratio=:equal)
    for evalpoint = 1:10:N
        e = eigvals(jacobian(ms, x[:,evalpoint])[1][1:nx,1:nx])
        scatter!(real.(e), imag.(e), c=:blue, show=false, subplot=1)
        e = log.(Complex.(e))/h
        scatter!(real.(e), imag.(e), c=:blue, show=false, subplot=2, legend=false)
        plot!(title="Eigenvalue spectrum")
    end
    phi = linspace(0,2pi,300)
    plot!(real.(exp.(phi.*im)), imag.(exp.(phi.*im)), legend=false, c=:black, l=:dash, subplot=1)
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

function eval_jac(results, eval=false)
    Jm, Js, Jtrue = all_jacobians(results, eval)
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

function plot_prediction(fig, results, eval=false)
    @unpack x,u,y,modeltype = results[1]
    ms = models(results)
    yh, bounds = predict(ms, x)
    for i = 1:size(yh,1)
        if size(bounds,1) >= i
            plot!(fig, yh[i,:], ribbon = 2bounds[i,:], fillalpha=0.3, subplot=i, lab="Prediction")
        else
            plot!(fig, yh[i,:], subplot=i, lab="Prediction")
        end
    end
end

function plot_simulation(fig, results, eval=false)
    @unpack x,u,y,modeltype = results[1]
    ms = models(results)
    yh = simulate( ms, x)
    for i = 1:size(yh,1)
        plot!(fig, yh[i,:], subplot=i, lab="Simulation")
    end
end

get_res(res,n) = getindex.(res,n)




try
foreach(treelike, [System, DiffSystem, VelSystem])
end
