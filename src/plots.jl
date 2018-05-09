
@recipe function plot_Trajectory(t::Trajectory; filtering=0, control=true)
    layout --> (control ? (2,1) : 1)
    show --> false
    @series begin
        title --> "States"
        xlabel --> "Time"
        control && (subplot --> 1)
        filtering > 0 ? filt(ones(filtering),[filtering], t.x') : t.x'
    end
    control && @series begin
        title --> "Control signal"
        xlabel --> "Time"
        subplot --> 2
        filtering > 0 ? filt(ones(filtering),[filtering], t.u') : t.u'
    end
    delete!(plotattributes, :filtering)
    delete!(plotattributes, :control)
    nothing
end

@userplot EigvalPlot

@recipe function eigvalplot(h::EigvalPlot; ds=10, cont=false, onlyat=0)
    ms = h.args[1]
    t = h.args[2]
    T = length(t)
    title --> "Eigenvalue spectrum"
    show --> false
    seriestype := :scatter
    @unpack xu,nu,nx = t
    lim = 1e-3
    truejac = length(h.args) >= 3
    ħ = 1
    inds = onlyat == 0 ? (1:ds:T) : onlyat
    cmap = colormap("Blues", T)
    cmapt = colormap("Reds", T÷ds+1)
    delete!(plotattributes, :onlyat)
    for evalpoint = inds
        J = jacobian(ms, xu[:,evalpoint])[1]
        e = eigvals(J[1:nx,1:nx])
        if cont
            e = log.(Complex.(e))./ħ
        end
        lim = max(maximum(imag.(e)), lim)
        @series begin
            c --> (onlyat == 0 ? cmap[evalpoint] : (:blue))
            real.(e), imag.(e)
        end
        # scatter!(real.(e), imag.(e), c=:blue, show=false, subplot=2, legend=false)
    end
    if truejac
        truejacfun = h.args[3]
        Jtrue = map(x->truejacfun(x...), (t.x[:,i],t.u[:,i]) for i=inds)
        for (i,J) in enumerate(Jtrue)
            e = eigvals(J[1:nx,1:nx])
            if cont
                e = log.(Complex.(e))./ħ
            end
            @series begin
                c --> (onlyat == 0 ? cmapt[i] : (:red))
                markersize --> 4
                real.(e), imag.(e)
            end
        end

    end
    delete!(plotattributes, :ds)
    phi = linspace(0,2π,300)
    seriestype := :path
    if !cont
        ratio --> :equal
        @series begin
            legend --> false; color --> :black; linestyle --> :dash
            real.(exp.(phi.*im)), imag.(exp.(phi.*im))
        end
    else
        @series (legend := false; linestyle := :dash; color := :black; ([0,0],[-lim,lim]))
    end
    delete!(plotattributes, :cont)
    nothing
end

function display_modeltrainer(mt::ModelTrainer; kwargs...)
    ms = mt.models
    modeldict = OrderedDict("All models" => 0,
                ["Model "*string(i) => i for i = 1:length(mt.models)]...)

    hist = length(mt.modelhistory)
    ui = @manipulate for  t  = togglebuttons(1:length(mt.trajs), selected=1),
                          mn = togglebuttons(modeldict, selected=1),
                          modelversion = slider(1:hist,  value=hist, label="Model state"),
                          f       = slider(1:100, label="Filtering", value=0),
                          ds      = slider(1:100, value=4, label="Downsampling"),
                          eigvals = [true,false],
                          cont    = [true,false],
                          trajplot = togglebuttons(OrderedDict( "Prediction"=>1,
                                                                "Simulation"=>2,
                                                                "LTV pred"=>3,
                                                                "LTV sim"=>4),
                                                                multiselect=true)

        ms        = mt.modelhistory[modelversion]
        modelinds = (mn <= 0 || mn > length(ms)) ? (1:length(ms)) : (mn:mn)
        ms = ms[modelinds]
        if eigvals
            eigvalplot(ms, mt.trajs[t]; ds=ds, cont=cont, kwargs...)
        else
            fig = plot(mt.trajs[t]; filtering=f, lab="True", kwargs...)
            1 ∈ trajplot && predplot!(ms,mt.trajs[t]; filtering=f, l=:dash, subplot=1)
            2 ∈ trajplot && simplot!(ms,mt.trajs[t]; filtering=f, l=:dash, subplot=1)
            3 ∈ trajplot && predplot!(KalmanModel(mt,mt.trajs[t]),mt.trajs[t]; filtering=f, l=:dash, subplot=1)
            4 ∈ trajplot && simplot!(KalmanModel(mt,mt.trajs[t]),mt.trajs[t]; filtering=f, l=:dash, subplot=1)
            fig
        end
    end
end


@userplot PredPlot
@recipe function predplot(h::PredPlot; filtering=1)
    ms = h.args[1]
    t = h.args[2]
    lab --> "Prediction"
    if ms isa Vector
        filt(ones(filtering),[filtering], predict(ms,t)[1]')
    else
        filt(ones(filtering),[filtering], LTVModelsBase.predict(ms,t)')
    end
end

@userplot SimPlot
@recipe function simplot(h::SimPlot; filtering=1)
    ms = h.args[1]
    t = h.args[2]
    lab --> "Simulation"
    simfun = ms isa LTVModelsBase.AbstractModel ? LTVModelsBase.simulate : simulate
    filt(ones(filtering),[filtering], simfun(ms,t)')
end

@userplot PredSimPlot
@recipe function predsimplot(h::PredSimPlot; filtering=1)
    ms = h.args[1]
    t = h.args[2]
    layout --> size(t.x,1)
    @series begin
        label --> "True"
        filt(ones(filtering),[filtering], t.x')
    end
    @series begin
        lab --> "Prediction"
        filt(ones(filtering),[filtering], predict(ms,t)[1]')
    end

    @series begin
        lab --> "Simulation"
        filt(ones(filtering),[filtering], simulate(ms,t)')
    end
end

"""
    jacplot(model(s), t [,truejac::(x,u)->J])
"""
jacplot

@userplot JacPlot
@recipe function jacplot(h::JacPlot; ds=10, cont=false)
    ms = h.args[1]
    t = h.args[2]
    truejac = length(h.args) >= 3
    if truejac
        truejacfun = h.args[3]
        Jtrue = map(x->truejacfun(x...), [(t.x[:,i],t.u[:,i]) for i=1:ds:size(t.x,2)])
        @show Jtrue[1] - Jtrue[end]
    end
    show --> false
    conf = !(ms isa LTVModelsBase.AbstractModel)
    if !conf
        Jm = reshape(cat(2,ms.At, ms.Bt), :, length(ms))[:,1:ds:end]
    else
        Jm,Js = jacobians(ms, t, ds)
    end
    N = size(Jm,1)
    layout --> N
    legend := false
    for i = 1:N
        seriestype --> :line
        subplot := i
        truejac && @series begin
            linestyle --> :dash
            getindex.(Jtrue,i)
        end
        @series begin
            conf && (ribbon --> 2Js[i,:])
            label --> "Estimated"
            fillalpha --> 0.4
            Jm[i,:]
        end
        @series [0 0]
    end
    delete!(plotattributes, :ds)
    delete!(plotattributes, :cont)
    nothing
end


@userplot MutRegPlot
@recipe function mutregplot(h::MutRegPlot; useprior = true, showltv = true)
    @assert length(h.args) >= 2 "Call with (mt::ModelTrainer, t::Trajectory, true_jacobian::(x,u)->J)\n or (mt::ModelTrainer, true_jacobian::(x,u)->J"
    mt         = h.args[1]
    manytrajs  = length(h.args) == 2
    if manytrajs
        t          = mt.trajs
        truejacfun = h.args[2]
    else
        t          = [h.args[2]]
        truejacfun = h.args[3]
    end

    errorhistory = map(mt.modelhistory) do ms
        map(t) do t
            showltv && (ltvmodel = KalmanModel(mt, t, ms, useprior=useprior))
            error_nn = 0.
            error_ltv = 0.
            for (i,xu) in enumerate(t)
                xi,ui = xu
                Jtrue = truejacfun(xi,ui)
                Jm, Js = jacobian(ms, [xi;ui])
                error_nn += eval_jac(Jm, Jtrue)
                showltv && (error_ltv += eval_jac([ltvmodel.At[:,:,i] ltvmodel.Bt[:,:,i]], Jtrue))
            end
            error_nn/length(t), error_ltv/length(t)
        end
    end
    title --> "Jacobian error"
    xlabel --> "Number of training sessions"
    ylims --> (0, Inf)
    N = length(errorhistory)
    errorhistory = [errorhistory[mi][ti] for mi in 1:N, ti in eachindex(errorhistory[1])]
    seriestype --> (showltv ? (:scatter) : (:bar))
    c1,c2 = :red, :blue
    @series (color --> c1;label := "NN"; ((1:N) .- 0.1, getindex.(errorhistory,1)))
    showltv && @series (color --> c2;label := "LTV"; ((1:N) .+ 0.1, getindex.(errorhistory,2)))
end





@userplot TracePlot
@recipe function traceplot(h::TracePlot)
    trace         = h.args[1]
    if !(trace isa ValueHistory)
        trace = trace.trace
    end
    yscale --> :log10
    trace
end
