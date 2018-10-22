# using ParallelDataTransfer
# using Distributed
# length(workers()) == 1 && @show addprocs(3)
# DID increased sample time

# using Revise
cd(@__DIR__)

@isdefined(simulate_pendcart) || (include(joinpath("/local/home/fredrikb/.julia/v0.6/GuidedPolicySearch/src/system_pendcart.jl")))
using Main.PendCart
using Parameters, JacProp, OrdinaryDiffEq, LTVModels, LTVModelsBase, LinearAlgebra, Statistics, Random, RollingFunctions
using Flux: params, jacobian
using Flux.Optimise: Param, optimiser, expdecay

function simaverage(x, n)
    N,p = size(x)
    res = map(1:p) do i
        mean(reshape(x[:,i], N÷n,:), dims=2)
    end
    hcat(res...)
end

@userplot RibbonPlot

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
    global generate_data, true_jacobian, true_jacobianc
    function fsys(xd,x,u)
        xd[1] = x[2]
        xd[2] = -g/l * sin(x[1]) + u[]/l * cos(x[1]) - d*x[2]
        xd[3] = x[4]
        xd[4] = u[]
        xd
    end

    function generate_data(sys::PendcartSys, seed, validation=false; ufun=u->filt(ones(200),[200], 20u')')
        @unpack N, nu, h, σ0 = sys
        Random.seed!(seed)
        done = false
        local x,u

        fs = exp10.(range(-3, stop=-1.5, length=50))[randperm(50)[1:5]]
        # u    = ufun(randn(nu,N+2))
        u = sum(f->sin.(2π*f.* (1:N+2) .+ 2π*rand()), fs)'
        u .+= 0.1randn(size(u))
        t    = 0:h:N*h
        x0   = [0.8π,0,0,0]
        prob = OrdinaryDiffEq.ODEProblem((xd,x,p,t)->fsys(xd,x, u[:,floor(Int,t/h)+1]),x0,(t[[1,end]]...,))
        sol  = solve(prob,Tsit5(),reltol=1e-4,abstol=1e-4)
        x    = hcat(sol(t)...)

        u = u[:,1:N]
        validation || (x .+= σ0 * randn(size(x)))
        @assert all(isfinite, x)
        @assert all(isfinite, u)
        x,u
    end

    function true_jacobianc(sys::PendcartSys, x::AbstractVector, u::AbstractVector)
        # u[isnan.(u)] = 0
        nx = size(x,1)
        nu = size(u,1)
        fx = Array{Float64}(undef,nx,nx)
        fu = Array{Float64}(undef,nx)
        fx[:,:] .= [0 1 0 0;
        -g/l*cos(x[1])-u[]/l*sin(x[1]) -d 0 0;
        0 0 0 1;
        0 0 0 0]
        fu[:,:] .= [0, cos(x[1])/l, 0, 1]
        [fx  fu]
    end

    function true_jacobian(sys::PendcartSys, x::AbstractVector, u::AbstractVector)
        J = true_jacobianc(sys,x,u)
        ABd = exp([J*sys.h; zeros(nu, nx + nu)])[1:4,:]# ZoH sampling
    end
end

function callbacker(epoch, loss,d,trace,model,mt)
    i = epoch
    # i = length(trace) + epoch - 1
    function ()
        l = sum(d->Flux.data(loss(d...)),d)
        increment!(trace,epoch,l)
        if i % 2 == 0
            # @printf("Loss: %.4f\n", l)
            # jacplot(model, trajs[1], true_jacobian, ds=5,show=true,reuse=true)
        end
    end
end


num_params = 30
wdecay     = 0.1
stepsize   = 0.01#0.01
const sys  = PendcartSys(N=200, h=0.01, σ0 = 0.01) # Slow sampling h = 0.02
true_jacobian(x,u)  = true_jacobian(sys,x,u)
true_jacobianc(x,u) = true_jacobianc(sys,x,u)
nu         = sys.nu
nx         = sys.nx



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


## Monte-Carlo evaluation
num_montecarlo = 40
it = num_montecarlo ÷ 2
res = map(1:num_montecarlo) do it
    trajs = [Trajectory(generate_data(sys, i)...) for i = it .+ (1:2)]
    r1 = begin
        Random.seed!(it)
        wdecay = exp10.(range(-8, stop=0, length=num_montecarlo))[it]
        # wdecay = 1e-5 # Slow, swish
        # wdecay = exp10(-2.5) # Slow, tanh
        # models     = [System(nx,nu,num_params, a) for a in default_activations]
        models     = [System(nx,nu,num_params, tanh)]
        opts       = [[ADAM.(params.(models), stepsize, ϵ=1e-2); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
        # opts       = ADAM.(params.(models), stepsize)
        trainer  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = wdecay)
        for i = 1:2
            trainer(trajs[i], epochs=0, jacprop=0, useprior=false)
        end
        trainer
    end
    r2 = begin
        Random.seed!(it)
        λ = exp10.(range(-4, stop=3, length=num_montecarlo))[it]
        # λ = 1 # Slow, swish
        # λ = 1 # Slow, tanh
        cb(model) = callbacker#(jacplot(model, trajs[1], true_jacobian, ds=5,show=true,reuse=true);gui())
        model     = JacProp.ADSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
        opt       = LTVModels.ADAMOptimizer.(model.w, α = stepsize, expdecay=0.009, ε=1e-2)
        trainerad = ADModelTrainer(;model=model, opt=opt, λ=λ, testdata = vt)
        for i = 1:2
        trainerad(trajs[i], epochs=0)
        end
        trainerad
    end
    r3 = begin
        Random.seed!(it)
        wdecay = exp10.(range(-8, stop=0, length=num_montecarlo))[it]
        # wdecay = 1e-3 # Slow, swish
        # wdecay = 1e-5 # Slow, tanh
        # models     = [DiffSystem(nx,nu,num_params, a) for a in default_activations]
        models     = [DiffSystem(nx,nu,num_params, tanh)]
        opts       = [[ADAM.(params.(models), stepsize, ϵ=1e-2); [expdecay(Param(p), wdecay) for p in params(models[i]) if p isa AbstractMatrix]] for i = 1:length(models)]
        # opts       = ADAM.(params.(models), stepsize)
        trainerd  = ModelTrainer(models = models, opts = opts, losses = JacProp.loss.(models), cb=callbacker, P = 10, R2 = 10000I, σdivider = wdecay)
        for i = 1:2
            trainerd(trajs[i], epochs=0, jacprop=0, useprior=false)
        end
        trainerd
    end
    r4 = begin
        Random.seed!(it)
        λ = exp10.(range(-6, stop=3, length=num_montecarlo))[it]
        # λ = 1e-4 # Slow, swish
        # λ = 1 # Slow, tanh
        cb(model) = callbacker#(jacplot(model, trajs[1], true_jacobian, ds=5,show=true,reuse=true);gui())
        model     = JacProp.ADDiffSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
        opt       = LTVModels.ADAMOptimizer.(model.w, α = stepsize, expdecay=0.001, ε=1e-2)
        traineradd = ADModelTrainer(;model=model, opt=opt, λ=λ, testdata = vt)
        for i = 1:2
        traineradd(trajs[i], epochs=0)
        end
        traineradd
    end

    println("Done with montecarlo run $it")
    r1,r2,r3,r4
end


res = [(fetch.(rs)...,) for rs in res]
ress = getindex.(res,1)
resad = getindex.(res,2)
resdiff = getindex.(res,3)
resaddiff = getindex.(res,4)


Threads.@threads for trainer=ress       train!(trainer, epochs=2500) end
@info("Done ress")
Threads.@threads for trainer=resad      train!(trainer, epochs=2500) end
@info("Done resad")
Threads.@threads for trainer=resdiff    train!(trainer, epochs=2500) end
@info("Done resdiff")
Threads.@threads for trainer=resaddiff  train!(trainer, epochs=2500) end
@info("Done resaddiff")

serialize("results_fast_tanh", (ress,resad,resdiff,resaddiff))
# (ress,resad,resdiff,resaddiff) = deserialize("results")
using Plots
using Plots.PlotMeasures
##
F = font(18, "times")
fontopts = (titlefont= F, tickfont= F, xtickfont= F, ytickfont= F, guidefont= F, xguidefont= F, yguidefont= F)
common = (xlims=(0.4,1.2), ylims=(-0.2,0.2), size=(1000,600), m=(3,),cont=false, grid=false, link = :both, markeralpha=0.15, ds=1, markerstrokealpha=0, xticks=0.6:0.2:1, colorbar=false, fontopts...)
default(size=(1000, 1000))
nice(x) = @sprintf("%.2g", x)
for i = 1#1:num_montecarlo
    rs = ress[i]; rad = resad[i]; rd = resdiff[i]; radd = resaddiff[i]
    eigvalplot(rs.models, vt, true_jacobian;layout=(2,2),subplot=1,title="f Weight decay = $(nice(rs.σdivider))", common...)
    eigvalplot!(rad.model, vt, true_jacobian;subplot=2,title="f Jacprop \\lambda = $(nice(rad.λ))", common...)
    eigvalplot!(rd.models, vt, true_jacobian;subplot=3,title="g Weight decay = $(nice(rd.σdivider))", common...)
    eigvalplot!(radd.model, vt, true_jacobian;subplot=4,title="g Jacprop \\lambda = $(nice(radd.λ))", common...)
    gui()
    # display(current())
end
# savefig("/local/home/fredrikb/phdthesis/blackbox/figs/eigvals_pendcart_slow_tanh.pdf")
##
@recipe function ribbonplot(r::RibbonPlot)
    x,y = r.args[1:2]
    n = length(y)
    ex = map(1:length(x)) do i
        extrema(getindex.(y,i))
    end
    means = map(1:length(x)) do i
        median(getindex.(y,i))
    end
    ma = runmedian(getindex.(ex,2), 400)
    fillalpha --> 0.2
    @series begin
        fillrange := getindex.(ex,1)
        label := ""
        x, means
    end
    @series begin
        fillrange := ma
        x, means
    end
end



ribbonplot(ress[1].trace.iterations,[res.trace.values for res in ress], c=:magenta, lab="f")#, xscale=:log10, yscale=:log10, legend=false)
ribbonplot!(resad[1].trace.iterations,[res.trace.values for res in resad], c=:blue, lab="f JacProp")
# ribbonplot!(resad[1].trace.iterations,[res.tracev.values for res in resad], c=:orange, lab="f JacProp")
ribbonplot!(resdiff[1].trace.iterations,[res.trace.values for res in resdiff], c=:red, lab="g")
ribbonplot!(resaddiff[1].trace.iterations[1:2500],[res.trace.values[1:2500] for res in resaddiff], c=:green, lab="g JacProp")
# ribbonplot!(resaddiff[1].trace.iterations,[res.tracev.values for res in resaddiff], c=:cyan) #|> display
plot!(xscale=:log10,yscale=:log10, xlabel="Epoch", ylabel="Cost")
# savefig3("/local/home/fredrikb/phdthesis/blackbox/figs/trainerr.tex")
##

using JacProp: eval_pred, eval_sim, eval_jac, eval_jac2
nr = length(res[1])÷2
labelvec = ["f" "g"]
infostring = @sprintf("Num hidden: %d, sigma: %2.2f, Montecarlo: %d", num_params, sys.σ0, num_montecarlo)
simvals = [Trajectory(generate_data(sys, i, true)...) for i = 4:6]
pred  = [eval_pred.(ress, (vt,)) eval_pred.(resad, (vt,)) eval_pred.(resdiff, (vt,)) eval_pred.(resaddiff, (vt,))]
sim   = simaverage(vcat([[eval_sim.(ress, (simval,)) eval_sim.(resad, (simval,)) eval_sim.(resdiff, (simval,)) eval_sim.(resaddiff, (simval,))] for simval in simvals]...), 3)
# sim = [x > 10 ? NaN : x for x in sim]
jac   = [eval_jac2.(ress, (vt,), true_jacobian,3) eval_jac2.(resad, (vt,), true_jacobian,3) eval_jac2.(resdiff, (vt,), true_jacobian,3) eval_jac2.(resaddiff, (vt,), true_jacobian,3)]

##
using StatPlots
λs = getfield.(ress, :σdivider)
λad = getfield.(resad, :λ)
λd = getfield.(resdiff, :σdivider)
λadd = getfield.(resaddiff, :λ)
xv = [λs λad λd λadd]
xticks = (1:2, ["\$f\$" "\$g\$"])
xvals = [1 2].*ones(num_montecarlo)
common = (ygrid=true, marker_z=xv, legend=false, xticks=xticks, lab=["Standard" "Jacobian propagation"], c=:red, size=(1000,600))
vio1 = violin(xvals,pred[:,1:2:end]; title="Prediction RMS", side=:left, common...)
violin!(xvals,pred[:,2:2:end], side=:right, c=:blue)
# vio2 = violin(xvals,sim[:,1:2:end]; title="Simulation RMS", side=:left, common...)
# violin!(xvals,sim[:,2:2:end], side=:right, c=:blue)
vio3 = violin(xvals,(jac[:,1:2:end]); title="Jacobian Error", yscale=:log10,side=:left, common...)
violin!(xvals,(jac[:,2:2:end]), yscale=:log10,side=:right, c=:blue)
plot(vio1,vio3,ylabel=infostring, colorbar=false, layout=(1,2))#size=(800,1000)) #|> display
##
# savetikz("/local/home/fredrikb/phdthesis/blackbox/figs/boxplot.tex")
# savefig3("/local/home/fredrikb/phdthesis/blackbox/figs/boxplot.tex")
# savefig3("/local/home/fredrikb/phdthesis/blackbox/figs/boxplot_slow.tex")
##



## Plot error vs λ regularization parameter
plot(xv,pred, layout=(3,1), subplot=1, ylabel="Prediction RMS", lab=["Standard" "Jacobian propagation"], background_color_legend=false, xscale=:log10,yscale=:log10, size=(1200,1100),legend=true)
plot!(xv,min.(sim,4), subplot=2, ylabel="Simulation RMS", lab=["Standard" "Jacobian propagation"], background_color_legend=false, xscale=:log10,yscale=:log10,legend=false)
plot!(xv,jac, subplot=3, ylabel="Jacobian Error", lab=["Standard" "Jacobian propagation"], background_color_legend=false, xscale=:log10,yscale=:log10,legend=false)

##


simvals = [Trajectory(generate_data(sys, i, true)...) for i = 4:6]
##
mno = 1
# jacplot(ress[mno].models, simvals[1], true_jacobian, ds=2, reuse=false, size=(1200,800), c=[:red :cyan], linewidth=[3 1])
# jacplot(resad[mno].model, simvals[1], ds=2, c=c=[:red :blue])
jacplot(resdiff[mno].models, simvals[1], ds=2, c=[:red :green], true_jacobian,reuse=false, size=(1200,800))
jacplot!(resaddiff[mno].model, simvals[1], ds=2, c=:magenta)
gui()



##

plot(ress[1].trajs[1], lab=["\$\\theta\$" "\$\\dot\\theta\$" "\$p\$" "\$v\$" "\$u\$"], grid=false)
# savefig3("/local/home/fredrikb/phdthesis/blackbox/figs/trajectory.tex")
predsimplot(ress[1].models, simvals[2], title="f Standard") #|> display
predsimplot(resad[1].model, simvals[2], reuse=false, title="f Jacprop") #|> display
predsimplot(resdiff[1].models, simvals[2], title="g Standard") #|> display
predsimplot(resaddiff[1].model, simvals[2], reuse=false, title="g Jacprop") #|> display
# simulate(resad[1].model, trajs[1])' |> plot



# jacplot(trainer.models, simvals[1], true_jacobian, ds=2, reuse=false, size=(1200,800))
# jacplot!(trainerad.model, simvals[1], ds=2)
#
# traceplot(trainerad,lab="AD");traceplot!(trainer, yscale=:identity,xscale=:log10, lab="Std"); plot!(trainerad.tracev)




num_montecarlo = 40
it = 1
res = pmap(1:num_montecarlo) do it
    λ = exp10.(range(-2, stop=2, length=100))[rand(1:100)]
    num_params = rand(10:100)
    Random.seed!(it)
    model     = JacProp.ADDiffSystem(nx,nu,num_params,tanh) # TODO: tanh has no effect
    opt       = LTVModels.ADAMOptimizer(model.w, α = stepsize)
    trainerad = ADModelTrainer(;model=model, opt=opt, λ=λ, testdata = vt)
    for i = 1:2
        trainerad(trajs[i], epochs=0)
    end
    trainerad(epochs=2000)
    println(it)
    trainerad
end
# serialize("res", res)

res2 = map(res) do r
    JacProp.eval_jac.(r, vt, true_jacobian,3)
end
im = indmin(res2)
trainerad = res[im]
s = map(res) do r
    r.model.sizes[1][1]
end
λ = map(resad) do r
    r.λ
end

jacplot(trainerad.model, trajs[1], true_jacobian, ds=2, reuse=false)

scatter([s λ], res2, layout=2)
plot!(xscale=:log10, yscale=:log10, subplot=2)




# Generate ss vs tf param plot ==========================================
# using ControlSystems
# t = Trajectory(generate_data(sys, 1)...)
#
# function jacs(t)
#     N = length(t)
#     J = map(1:N) do evalpoint
#         Jd = true_jacobian(t.x[:,evalpoint],t.u[:,evalpoint])
#         Jc = true_jacobianc(t.x[:,evalpoint],t.u[:,evalpoint])
#         Gss = c2d(ss(Jc[1:end,1:nx], Jc[1:end,nx+1:end],[1 0 0 0;0 0 1 0],0), sys.h)[1]
#         Gtf = tf(Gss)
#         # @show numpoly(Gtf)
#         hcat(getfield.(denpoly(Gtf),:a)...)
#         Jd[:], [hcat(getfield.(numpoly(Gtf),:a)...)' hcat(getfield.(denpoly(Gtf),:a)...)']
#     end
#     JacProp.smartcat2(getindex.(J, 1)), JacProp.smartcat3(getindex.(J, 2))
# end
# Gss, Gtf = jacs(t)
# Gtf = reshape(Gtf, 2*9,100)
#
# plot((Gss./maximum(abs.(Gss),2))', layout=2, subplot=1, c=:blue, legend=false)
# plot!((Gtf./maximum(abs.(Gtf),2))', subplot=2, c=:blue, legend=false)
