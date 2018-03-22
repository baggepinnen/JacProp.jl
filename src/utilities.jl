import StatsBase.predict

abstract type AbstractSystem end

@with_kw struct System{T} <: AbstractSystem
    m::T
    ns::Int
    n::Int
    h::Float64 = 1.0
end
function System(n,ns, num_params, activation, h=1)
    ny = ns
    np = num_params
    m  = Chain(Dense(ns+n,np, activation), Dense(np, ny))
    System(m, ns, n, h)
end
(m::System)(x) = m.m(x)


@with_kw struct DiffSystem{T} <: AbstractSystem
    m::T
    ns::Int
    n::Int
    h::Float64 = 1.0
end
function DiffSystem(n,ns, num_params, activation, h=1)
    ny = ns
    np = num_params
    m  = Chain(Dense(ns+n,np, activation), Dense(np, ny))
    DiffSystem(m, ns, n, h)
end
(m::DiffSystem)(x) = m.m(x)+x[1:m.ns,:]


@with_kw struct VelSystem{T} <: AbstractSystem
    m::T
    ns::Int
    n::Int
    h::Float64 = 1.0
end
function VelSystem(n,ns, num_params, activation, h=1)
    ny = n
    np = num_params
    m = Chain(Dense(ns+n,np, activation),  Dense(np, ny))
    VelSystem(m, ns, n, h)
end
(m::VelSystem)(x) = m.m(x)

loss(m::AbstractSystem) = (x,y) -> sum((m(x).-y).^2)/size(x,2)

const AbstractEnsembleSystem = Vector{<:AbstractSystem}
const EnsembleSystem = Vector{System}
const EnsembleDiffSystem = Vector{DiffSystem}
const EnsembleVelSystem = Vector{VelSystem}
# const EnsembleAffineSystem = Vector{AffineSystem}

xy(::Type{<:AbstractSystem}, x, N)     = (x[:,1:N], x[:,2:N+1])
xy(::Type{VelSystem}, x, N)  = (x[:,1:N], x[3:4,2:N+1])


function get_minimum_loss(results, key)
    array = getindex.(get.(results[key]),2)
    return isempty(array) ? Inf : minimum(array)
end

function simulate(ms::AbstractEnsembleSystem,x, testmode=true)
    Flux.testmode!.(ms, testmode)
    xsim = copy(x)
    ns = ms[1].ns
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
        h = ms[1].sys.h
        xsim[1:2,t] = xsim[1:2,t-1] + h*xsim[3:4,t-1] # TODO: * sample time (h)
        xsim[3:4,t] = mean(xsimt).data
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
    h = ms[1].sys.h
    yh = [x[1:2,1].+h*cumsum(yh,2); yh] # TODO: * sample time (h)
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
    h = ms[1].sys.h
    jacs = [[[I h*I h^2/2*eye(2)];Flux.jacobian(m,x)] for m in ms] # The h²/2*I in ∇ᵤ is an approximation since there are (very small) cross terms. TODO: hard coded sample time
    jacmat = cat(3,jacs...)
    squeeze(mean(jacmat, 3), 3), squeeze(std(jacmat, 3), 3)
end

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
    n = ms[1].n
    plot(layout=(2,1), ratio=:equal)
    for evalpoint = 1:10:N
        e = eigvals(jacobian(ms, x[:,evalpoint])[1][1:n,1:n])
        scatter!(real.(e), imag.(e), c=:blue, show=false, subplot=1)
        e = log.(Complex.(e))/sys.h
        scatter!(real.(e), imag.(e), c=:blue, show=false, subplot=2, legend=false)
        plot!(title="Eigenvalue spectrum")
    end
    phi = linspace(0,2pi,300)
    plot!(real.(exp.(phi.*im)), imag.(exp.(phi.*im)), legend=false, c=:black, l=:dash, subplot=1)
end


function eval_pred(results, eval=false)
    @unpack x,u,y,modeltype = results[1]
    ms = models(results)
    ns = ms[1].ns
    yh, bounds = predict(ms, x)
    pred = rms(x[1:ns,2:end].-yh[:,1:end-1]) # TODO: check time alignment
    pred
end

function eval_sim(results, eval=false)
    @unpack x,u,y,modeltype = results[1]
    ms = models(results)
    ns = ms[1].ns
    xh = simulate(ms, x)
    sim = rms(x[1:ns,:].-xh) # TODO: check time alignment
    sim
end

function eval_jac(results, eval=false)
    Jm, Js, Jtrue = all_jacobians(results, eval)
    sqrt(mean(abs2, Jm .- Jtrue))
end

function plotresults(results, eval=false)
    @unpack x,u,y,modeltype = results[1]
    ms = models(results)
    ns = ms[1].ns
    fig = plot(x[1:ns,2:end]', lab="True", layout=ns)
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
foreach(treelike, [SystemD, DiffSystemD, VelSystemD, System, DiffSystem, VelSystem])
end
