if length(workers()) == 1
    addprocs(4)
end
pyplot()
default(grid=false)
# @everywhere using Revise
using ParallelDataTransfer
@everywhere using Parameters, JacProp, OrdinaryDiffEq, LTVModels, LTVModelsBase
@everywhere using Flux: params, jacobian
@everywhere using Flux.Optimise: Param, optimiser, expdecay
@everywhere begin

    @with_kw struct LinearSys <: AbstractSystem
        A
        B
        N = 1000
        nx = size(A,1)
        nu = size(B,2)
        h = 0.02
        σ0 = 0
        sind = 1:nx
        uind = nx+1:(nx+nu)
        s1ind = (nx+nu+1):(nx+nu+nx)
    end

    function LinearSys(seed; nx = 10, nu = nx, h=0.02, kwargs...)
        srand(seed)
        A = randn(nx,nx)
        A = A-A'        # skew-symmetric = pure imaginary eigenvalues
        A = A - h*I     # Make 'slightly' stable
        A = expm(h*A)   # discrete time
        B = h*randn(nx,nu)
        LinearSys(;A=A, B=B, nx=nx, nu=nu, h=h, kwargs...)
    end

    function generate_data(sys::LinearSys, seed, validation=false)
        Parameters.@unpack A,B,N, nx, nu, h, σ0 = sys

        srand(seed)

        u      = filt(ones(5),[5], 10randn(N+2,nu))'
        t      = h:h:N*h+h
        x0     = randn(nx)
        x      = zeros(nx,N+1)
        x[:,1] = x0
        for i = 1:N-1
            x[:,i+1] = A*x[:,i] + B*u[:,i]
        end

        validation || (x .+= σ0 * randn(size(x)))
        u = u[:,1:N]
        @assert all(isfinite, u)
        x,u
    end

    function true_jacobian(sys::LinearSys, x, u)
        [sys.A sys.B]
    end

    function callbacker(epoch, loss,d,trace,model)
        i = length(trace) + epoch - 1
        function ()
            Flux.reset!(model.m)
            l = sum(d->Flux.data(loss(d...)),d)
            increment!(trace,epoch,l)
            i % 500 == 0 && println(@sprintf("Loss: %.4f", l))
            Flux.reset!(model.m)
        end
    end

    num_params = 30
    wdecay     = 0
    stepsize   = 0.02
    const sys  = LinearSys(1, N=200, h=0.02, σ0 = 0.01)
    true_jacobian(x,u) = true_jacobian(sys,x,u)
    nu         = sys.nu
    nx         = sys.nx

end


## Generate validation data
function valdata()
    vx,vu,vy = Vector{Float64}[],Vector{Float64}[],Vector{Float64}[]
    for i = 20:60
        x,u = generate_data(sys,i, true)
        for j in 10:5:(sys.N-1)
            push!(vx, x[:,j])
            push!(vy, x[:,j+1])
            push!(vu, u[:,j])
        end
    end
    hcat(vx...),hcat(vu...),hcat(vy...)
end
vx,vu,vy = valdata()
vt = Trajectory(vx,vu,vy)

trajs = [Trajectory(generate_data(sys, i)...) for i = 1:3]

sendto(collect(2:5), trajs=trajs, vt=vt)



wdecay = 0.0
# f1 = @spawnat 4 begin
srand(1)
models     = [RecurrentSystem(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainers  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
# for i = 1:3
    trainers(trajs[i], epochs=0, jacprop=0, useprior=false)
    # traceplot(trainers)
# end
trainers
# endtrainerswd
trainers(epochs=5000, jacprop=0)
traceplot(trainers, reuse=false)

Flux.reset!.(getfield.(trainers.models, :m))
simplot(trainers.models, trainers.trajs[1], layout=10)
plot!(trainers.trajs[1], ls=:dash, control=false)


# f2 = @spawnat 2 begin
srand(1)
models     = [RecurrentDiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)#; [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainerds  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
for i = 1:3
    trainerds(trajs[i], epochs=0, jacprop=0, useprior=false)
    # traceplot(trainerds)
end
trainerds
# end
trainerds(epochs=5000, jacprop=0)
traceplot(trainerds, reuse=false)

Flux.reset!.(getfield.(trainerds.models, :m))
simplot(trainerds.models, trainerds.trajs[1], layout=10)
plot!(trainerds.trajs[1], ls=:dash, control=false)



wdecay = 0.1
# f3 = @spawnat 3 begin
srand(1)
models     = [RecurrentSystem(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainerswd  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
for i = 1:3
    trainerswd(trajs[i], epochs=0, jacprop=0, useprior=false)
    # traceplot(trainerswd)
end
trainerswd
# end
trainerswd(epochs=5000, jacprop=0)
traceplot(trainerswd, reuse=false)

Flux.reset!.(getfield.(trainerswd.models, :m))
simplot(trainerswd.models, trainerswd.trajs[1], layout=10)
plot!(trainerswd.trajs[1], ls=:dash, control=false)


# f4 = @spawnat 1 begin
srand(1)
models     = [RecurrentDiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainerdswd  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
for i = 1:3
    trainerdswd(trajs[i], epochs=0, jacprop=0, useprior=false)
    # traceplot(trainerdswd)
end
trainerdswd
# end
trainerdswd(epochs=5000, jacprop=0)
traceplot(trainerdswd, reuse=false)

Flux.reset!.(getfield.(trainerdswd.models, :m))
simplot(trainerdswd.models, trainerdswd.trajs[1], layout=10)
plot!(trainerdswd.trajs[1], ls=:dash, control=false)


trainers,trainerds = fetch(f1), fetch(f2)
trainerswd,trainerdswd = fetch(f3), fetch(f4)

traceplot(trainers, layout=4)
traceplot!(trainerds, subplot=2)
traceplot!(trainerswd, subplot=3)
traceplot!(trainerdswd, subplot=4)

eigvalplot(trainers.models, vt, true_jacobian, title="f, wdecay=0", layout=4, subplot=1, markeralpha=0.05, ds=1, markerstrokealpha=0)
eigvalplot!(trainerds.models, vt, true_jacobian,  title="g, wdecay=0", subplot=2, markeralpha=0.05, ds=1, markerstrokealpha=0)
eigvalplot!(trainerswd.models, vt, true_jacobian,  title="f, wdecay=$wdecay", subplot=3, markeralpha=0.05, ds=1, markerstrokealpha=0)
eigvalplot!(trainerdswd.models, vt, true_jacobian,  title="g, wdecay=$wdecay", subplot=4, markeralpha=0.05, ds=1, markerstrokealpha=0)
plot!(link=:both, xlims=(0,1.1), ylims=(-0.5, 0.5))
gui()


ui = display_modeltrainer(trainerdswd, size=(800,600))

for acti in 1:4
    plot(layout=4, link=:both)
    actstr = string(JacProp.default_activations[acti])
    actstr = actstr[1:2] == "NN" ? actstr[7:end] : actstr
    for tri in 1:4
        tr = [trainers, trainerds, trainerswd, trainerdswd][tri]
        eigvalplot!([tr.models[acti]], vt, true_jacobian, title="$actstr, $(tri%2==0 ? "g" : "f"), wdecay=$(tri>2 ? wdecay : 0)", subplot=tri, markeralpha=0.05, ds=3, markerstrokealpha=0)
    end
    gui()
end
