# pyplot()
using Plots
default(grid=false) #src
plot(randn(10))
# closeall();gui()
using Parameters, LTVModelsBase, ValueHistories, DSP, ForwardDiff, Nabla#, JLD, ReverseDiff#, JacProp
const Diff = ForwardDiff
# const RDiff = ReverseDiff
@with_kw struct LinearSys
    A
    B
    N     = 1000
    nx    = size(A,1)
    nu    = size(B,2)
    h     = 0.02
    σ0    = 0
    sind  = 1:nx
    uind  = nx+1:(nx+nu)
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
    u      = DSP.filt(ones(5),[5], 10randn(N+2,nu))'
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


num_params = 30
wdecay     = 0
stepsize   = 0.02
const sys  = LinearSys(1, nx=5, N=100, h=0.02, σ0 = 0.01)
true_jacobian(x,u) = true_jacobian(sys,x,u)
nu         = sys.nu
nx         = sys.nx



# Generate validation data
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
const vt = Trajectory(vx,vu,vy)

# Generate training trajectories
const trajs = [Trajectory(generate_data(sys, i)...) for i = 1:3]
# TODO: take care of separate trajs
function todata(trajs::Vector{Trajectory})
    xu = hcat(getfield.(trajs,:xu)...)
    y = hcat(getfield.(trajs,:y)...)
    xu, y
end


function pred(w,x)
    state = copy(x)
    for i=1:2:length(w)-2
        state = tanh.(w[i]*state .+ w[i+1])
    end
    return w[end-1]*state .+ w[end] .+ x[1:nx,:]
end


cost(w,x,y)  = sum(abs2.( pred(w,x) .- y))/size(y,2)
# cost(w,data) = sum(cost(w, d...) for d in data)/length(data)
cost(w,data) = cost(w, data...)

function loss(w,x,y)
    chunk = Diff.Chunk(x[:,1])
    function lf(w)
        # println("Entering loss function, typeof(w):", typeof(w))
        model(x)    = pred(w,x)
        jcfg        = Diff.JacobianConfig(model, x[:,1], chunk)
        jacobian(x) = Diff.jacobian(model, x, jcfg)
        @show l = cost(w,x,y)
        @show J1 = jacobian(x[:,1])
        for t = 2:size(x,2)
            J2 = jacobian(x[:,t])
            l += 2sum(abs2.(J1.-J2))
            J1 = J2
        end
        l
    end
end

##

function runtrain(w, loss; epochs = 500)
    dtrn        = todata(trajs)
    dtst        = todata([vt])
    x,y         = dtrn
    lossfun     = loss(w,x,y)
    # gradient(w) = Diff.gradient(lossfun, w, gcfg)
    trace       = History(Float64)
    tracev      = History(Float64)
    push!(trace, 0, lossfun(w))
    push!(tracev, 0, cost(w,dtst))
    g = similar.(w)
    m = zeros.(size.(g))
    plot(reuse=false)
    for epoch=1:epochs
        lossfun     = loss(w,x,y)
        γ = 0.85
        g = ∇(lossfun)(w.-γ.*m)
        for i in eachindex(g)
            @. m[i] = 0.002g[i] + γ*m[i]
            @. w[i] -= m[i]
        end
        if epoch % 10 == 0
            push!(trace, epoch, lossfun(w))
            push!(tracev, epoch, cost(w,dtst))
            plot(trace, reuse=true)
            plot!(tracev)
            gui()
            println(last(trace), last(tracev))
        end
    end
    trace,tracev
end


Base.promote_rule(::Type{Float64}, ::Type{Yota.TReal}) = Yota.TReal
Base.convert(::Type{Yota.TReal}, f::Float64) = Yota.TReal(f)
const sizes = ((num_params,nx+nu), (num_params,1), (nx,num_params), (nx,1))
srand(1)
w1 = [ 0.1randn(Float64,sizes[1]), zeros(Float64,sizes[2]),0.1randn(Float64,sizes[3]),  zeros(Float64,sizes[4]) ]
w1 = tovec(w1)
w2 = deepcopy(w1)

# res = runtrain(w1, (w,x,y)-> (w-> cost(w,x,y)), epochs=300)
resj = runtrain(w2, loss, epochs=1000)
# @save "res1.jld" res w1
# @save "resj.jld" resj w2
##

# Eval jac
model(x)    = pred(w2,x,sizes)
jcfg        = Diff.JacobianConfig(model, vt.xu[:,1])
jacobian(x) = Diff.jacobian(model, x, jcfg)
jacs = vcat([eigvals(jacobian(vt.xu[:,i])[1:nx,1:nx]) for i = 1:length(vt)]...)
scatter(real.(jacs), imag.(jacs))
scatter!(real.(eigvals(sys.A)), imag.(eigvals(sys.A)))
phi = linspace(-π/5,π/5,200)
plot!(real.(exp.(phi.*im)), imag.(exp.(phi.*im)), l=(:dash, :black), yaxis=[-0.5,0.5], xaxis=[0,1.2])


@load "resj.jld"
model(x)    = pred(w2,x,sizes)
jfg         = ForwardDiff.JacobianConfig(model, zeros(nx+nu))
jacobian(x) = ForwardDiff.jacobian(model, x)
m = JacProp.ADDiffSystem(w2,sizes,nx,nu, 1, jacobian)
JacProp.eigvalplot(m,vt, true_jacobian)

# Simple test
using BenchmarkTools
using ForwardDiff
const Diff = ForwardDiff
w = randn(20,20)
# x = [randn(20) for i = 1:100]
x = randn(20,100)
f(w::AbstractMatrix,x) = w*x
f(w::AbstractVector,x) = f(reshape(w, 20,20),x)
# const jcfg  = Diff.JacobianConfig(model, x)
function get_loss(w,x)
    modelw(w)    = [f(w,x) for x = x]
    function loss(w)
        modelx(x::AbstractVector)    = f(w,x)
        jacobian(x) = Diff.jacobian(modelx, x)
        # @assert jacobian(x) == w
        # sum(sum(modelw(w))) + sum(sum(jacobian(xi) for xi in x))
        sum(sum(modelw(w))) + sum(sum(jacobian(x[:,i]) for i in 1:size(x,2)))
        # sum(modelw(w)) + sum(jacobian(x))
    end
    loss
end
loss = get_loss(w,x)
grad(w) = Diff.gradient(loss,vec(w))
grad(w)

@btime grad($w)


# f(w,x) = NeuralNetwork(w,x)
# function loss(w,x)
#     ∇xf(x) = jacobian(x->f(w,x), x)
#     w -> norm(∇xf(x))
# end
#
# l = loss(w,x)
# gradient(loss,w) # Good luch with this




function i2m(w,i)
    s = [1; cumsum([prod.(sizes)...]) .+ 1]
    reshape(w[s[i]:(s[i+1]-1)], sizes[i])
end

function pred(w,x)
    state = copy(x)
    for i=1:2:length(sizes)-2
        state = tanh.(i2m(w,i)*state .+ i2m(w,i+1))
    end
    return i2m(w,length(sizes)-1)*state .+ i2m(w,length(sizes))
end


cost(w,x,y)  = sum(abs2.( pred(w,x) .- y))/size(y,2)




function loss(w,x,y)
    chunk = Diff.Chunk(x[:,1])
    function lf(w)
        jacobian(x) = Diff.jacobian(x->pred(w,x), x)
        @show l = cost(w,x,y)
        @show J1 = jacobian(x[:,1])
        for t = 2:size(x,2)
            J2 = jacobian(x[:,t])
            l += 2sum(abs2.(J1.-J2))
            J1 = J2
        end
        l
    end
end
