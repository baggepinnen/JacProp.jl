const PARALLEL = false
macro ifparallel(e)
    if PARALLEL
        return e
    end
end
@ifparallel if length(workers()) == 1
    addprocs(4)
end
pyplot()
default(grid=false)
# @everywhere using Revise
@ifparallel using ParallelDataTransfer
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

    lastprint = 0
    function callbacker(epoch, loss,d,trace,model)
        function ()
            global lastprint
            Flux.reset!(model.m)
            l = sum(d->Flux.data(loss(d...)),d)
            Flux.reset!(model.m)
            increment!(trace,epoch,l)
            if epoch % 20 == 0 && lastprint != epoch
                @show epoch
                println(@sprintf("Loss: %.4f", l))
                plot(trace, reuse=true, yscale=:log10)
                gui()
                lastprint = epoch
            end
        end
    end


    num_params = 20
    wdecay     = 0
    stepsize   = 0.005
    const sys  = LinearSys(1, N=200, h=0.02, σ0 = 1)
    true_jacobian(x,u) = true_jacobian(sys,x,u)
    nu         = sys.nu
    nx         = sys.nx

end


numtrajs = 3
vt = Trajectory(generate_data(sys, 100)...) #' Generate validation data
trajs = [Trajectory(generate_data(sys, i)...) for i = 1:numtrajs] #' Generate training data
@ifparallel sendto(collect(2:nprocs()), trajs=trajs, vt=vt) # Send data to workers



# f1 = @spawnat 4 begin
srand(1)
models     = [RecurrentSystem(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)
trainers  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
for i = 1:numtrajs
    trainers(trajs[i], epochs=0, jacprop=0)
end
trainers
# endtrainerswd
trainers(epochs=2000, jacprop=0)
predsimplot(trainers.models, vt, layout=10)
traceplot(trainers, reuse=false)



# f2 = @spawnat 2 begin
srand(1)
models     = [RecurrentDiffSystem(nx,nu,num_params, a) for a in default_activations]
opts       = ADAM.(params.(models), stepsize, decay=0.0005)
trainerds  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
for i = 1:numtrajs
    trainerds(trajs[i], epochs=0, jacprop=0)
end
trainerds
# end
trainerds(epochs=2000, jacprop=0)
predsimplot(trainerds.models, vt, layout=10)
traceplot(trainerds, reuse=false)




wdecay = 0.01
# f3 = @spawnat 3 begin
srand(1)
models     = [RecurrentSystem(nx,nu,num_params, a) for a in default_activations]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainerswd  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
for i = 1:numtrajs
    trainerswd(trajs[i], epochs=0, jacprop=0)
end
trainerswd
# end
trainerswd(epochs=2000, jacprop=0)
predsimplot(trainerswd.models, vt, layout=10)
traceplot(trainerswd, reuse=false)



# f4 = @spawnat 1 begin
srand(1)
models     = [RecurrentDiffSystem(nx,nu,num_params, a) for a in default_activations[3:3]]
opts       = [[ADAM(params(models[i]), stepsize, decay=0.0005); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
trainerdswd  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker)
for i = 1:numtrajs
    trainerdswd(trajs[i], epochs=0, jacprop=0)
    # traceplot(trainerdswd)
end
trainerdswd
# end
trainerdswd(epochs=20000, jacprop=0)
predsimplot(trainerdswd.models, vt, layout=10)
traceplot(trainerdswd, reuse=false)



# trainers,trainerds = fetch(f1), fetch(f2)
# trainerswd,trainerdswd = fetch(f3), fetch(f4)

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

JLD.@save "workspace" trainers trainerds trainerswd trainerdswd trajs
