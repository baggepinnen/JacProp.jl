cd(@__DIR__)
length(workers()) == 1 && @show addprocs(4)

# @everywhere using Revise
using ParallelDataTransfer
isdefined(:simulate_pendcart) || (@everywhere include(joinpath("/local/home/fredrikb/.julia/v0.6/GuidedPolicySearch/src/system_pendcart.jl")))
@everywhere using PendCart
@everywhere using Parameters, JacProp, OrdinaryDiffEq, LTVModels, LTVModelsBase
@everywhere using Flux: params, jacobian
@everywhere using Flux.Optimise: Param, optimiser, expdecay
@everywhere begin
    BLAS.set_num_threads(1)
    @with_kw struct PendcartSys
        N  = 1000
        nu  = 1
        nx = 4
        h = 0.02
        σ0 = 0
        sind = 1:nx
        uind = nx+1:(nx+nu)
        s1ind = (nx+nu+1):(nx+nu+nx)
    end

    let h = 0.01, g = 9.82, l = 0.35, d = 1
        global generate_data, true_jacobian
        function fsys(xd,x,u)
            xd[1] = x[2]
            xd[2] = -g/l * sin(x[1]) + u[]/l * cos(x[1]) - d*x[2]
            xd[3] = x[4]
            xd[4] = u[]
            xd
        end

        function generate_data(sys::PendcartSys, seed, validation=false; ufun=u->filt(ones(200),[200], 20u')')
            @unpack N, nu, h, σ0 = sys
            srand(seed)
            done = false
            local x,u

            fs = logspace(-3,-1.5,50)[randperm(50)[1:5]]
            # u    = ufun(randn(nu,N+2))
            u = sum(f->sin.(2π*f.* (1:N+2) .+ 2π*rand()), fs)'
            t    = 0:h:N*h
            x0   = [0.8π,0,0,0]
            prob = OrdinaryDiffEq.ODEProblem((xd,x,p,t)->fsys(xd,x, u[:,floor(Int,t/h)+1]),x0,(t[[1,end]]...))
            sol  = solve(prob,Tsit5(),reltol=1e-4,abstol=1e-4)
            x    = hcat(sol(t)...)

            u = u[:,1:N]
            validation || (x .+= σ0 * randn(size(x)))
            @assert all(isfinite, x)
            @assert all(isfinite, u)
            x,u
        end

        function true_jacobian(sys::PendcartSys, x::AbstractVector, u::AbstractVector)
            u[isnan.(u)] = 0
            nx = size(x,1)
            nu = size(u,1)
            fx = Array{Float64}(nx,nx)
            fu = Array{Float64}(nx)
            fx[:,:] = [0 1 0 0;
            -g/l*cos(x[1])-u/l*sin(x[1]) -d 0 0;
            0 0 0 1;
            0 0 0 0]
            fu[:,:] = [0, cos(x[1])/l, 0, 1]
            ABd = expm([fx[:,:]*h  fu[:,:]*h; zeros(nu, nx + nu)])[1:4,:]# ZoH sampling
        end
    end

    function callbacker(epoch, loss,d,trace,model,mt)
        i = length(trace) + epoch - 1
        function ()
            l = sum(d->Flux.data(loss(d...)),d)
            increment!(trace,epoch,l)
            i % 500 == 0 && println(@sprintf("Loss: %.4f", l))
        end
    end


    num_params = 40
    wdecay     = 0
    stepsize   = 0.01
    const sys  = PendcartSys(N=200, h=0.02, σ0 = 0.3)
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

sendto(workers(), trajs=trajs, vt=vt)

## Monte-Carlo evaluation
num_montecarlo = 4
it = 1
res = map(1:num_montecarlo) do it
    r2 = @spawn begin
        srand(it)
        model     = JacProp.ADDiffSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
        opt       = LTVModels.ADAMOptimizer(model.w, α = stepsize)
        trainerad = ADModelTrainer(;model=model, opt=opt, λ=0.01, testdata = vt)
        for i = 1:3
            trainerad(trajs[i], epochs=0)
        end
        trainerad(epochs=40)
        trainerad
    end
    r1 = @spawn begin
        srand(it)
        # models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
        models     = [DiffSystem(nx,nu,num_params, tanh)]
        opts       = ADAM.(params.(models), stepsize, decay=0.0005)#;
        trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = 20)
        for i = 1:3
            trainer(trajs[i], epochs=0, jacprop=0, useprior=false)
            # traceplot(trainer)
        end
        trainer(epochs=100)
        trainer
    end

    println("Done with montecarlo run $it")
    r1,r2
end
res = [(fetch.(rs)...) for rs in res]

# serialize("results", res)
# res = deserialize("results")
eigvalplot(res[1][1].models, vt, true_jacobian;layout=2,subplot=1,cont=false,title="Standard", ylims=[-0.1,0.1], xlims=[0.5,1.1])
eigvalplot!(res[1][2].model, vt, true_jacobian;subplot=2,cont=false,title="AD Jacprop", ylims=[-0.1,0.1], xlims=[0.5,1.1]);gui()
plot(res[1][2].trace.iterations,[res[2].trace.values for res in res], c=:blue)
plot!(res[1][2].trace.iterations,[res[2].tracev.values for res in res], c=:orange)
plot!(res[1][1].trace.iterations,[res[1].trace.values for res in res]./4, c=:red, xscale=:log10, yscale=:log10, legend=false)


resdiff = getindex.(res,1)
resad = getindex.(res,2)

nr = length(res[1])÷2
labelvec = ["f" "g"]
infostring = @sprintf("Num hidden: %d, sigma: %2.2f, Montecarlo: %d", num_params, sys.σ0, num_montecarlo)
simvals = [Trajectory(generate_data(sys, i, true)...) for i = 4:6]
pred  = [JacProp.eval_pred.(resdiff, vt) JacProp.eval_pred.(resad, vt)]
sim   = vcat([[JacProp.eval_sim.(resdiff, simval) JacProp.eval_sim.(resad, simval)] for simval in simvals]...)
jac   = [JacProp.eval_jac.(resdiff, vt, true_jacobian,3) JacProp.eval_jac.(resad, vt, true_jacobian,3)]

##
using StatPlots
vio1 = boxplot(pred, lab=["Standard" "Jacobian propagation"], ylabel="Prediction RMS", reuse=false, yscale=:log10)
vio2 = boxplot(sim, lab=["Standard" "Jacobian propagation"], ylabel="Simulation RMS", yscale=:log10)
vio3 = boxplot(jac, lab=["Standard" "Jacobian propagation"], ylabel="Jacobian Error", yscale=:identity)
plot(vio1,vio2,vio3,title=infostring); gui()
##
# savefig2("/local/home/fredrikb/papers/nn_prior/figs/valerr.tex")

plot(trajs[1])
predsimplot(resdiff[1].models, simvals[2]); gui()
predsimplot(resad[1].model, simvals[2], reuse=false); gui()
# simulate(resad[1].model, trajs[1])' |> plot
##
i = 1
j = 3
plot(simvals[i], control=false, layout=4, lab="Traj")
simplot!(resdiff[j].models, simvals[i], lab="Standard")
simplot!(resad[j].model, simvals[i], lab="AD JAcprop")
##


jacplot(resdiff[1].models, trajs[1], true_jacobian, ds=2, reuse=false)
jacplot!(resad[1].model, trajs[1], ds=2); gui()














let h = 0.01, g = 9.82, l = 0.35, d = 2
    function fsys(xd,x,u)
        xd[1] = x[2]
        xd[2] = -g/l * sin(x[1]) + u[]/l * cos(x[1]) - d*x[2]
        xd[3] = x[4]
        xd[4] = u[]
        xd
    end
    x0 = [0.9π,0,0,0]
    @show fsys(similar(x0), x0, 0)

end