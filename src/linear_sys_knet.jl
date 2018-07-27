default(grid=false) #src
using Parameters, Knet, LTVModelsBase, ValueHistories
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


num_params = 30
wdecay     = 0
stepsize   = 0.02
const sys  = LinearSys(1, N=200, h=0.02, σ0 = 0.01)
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


# # Without jacprop
srand(1)

function pred(w,x)
    x = mat(x)
    for i=1:2:length(w)-2
        x = tanh.(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

function jacobian(f)
    (J,x) -> begin
    for i=1:nx
        J[:,i] .= grad(x->f(x)[i])(x)
    end
    J
    end
end


cost(jacfun) = cost
cost(w,x,y)  = sum(abs2, pred(w,x) .- y)/size(y,2)
function loss(jacfun)
    function lf(w,x,y)
        l = cost(w,x,y)
        J2 = zeros(nx+nu, nx)
        J1 = similar(J2)
        jacfun(J1, x[:,1])
        for t = 2:size(x,2)
            jacfun(J2, x[:,t])
            l += 0.01sum(abs2.(J1.-J2))
            @show typeof(l)
            copy!(J1, J2)
        end
        l
    end
    lf(w,data) = sum(lf(w, d...) for d in data)/length(data)
    lf
end


cost(w,data) = sum(cost(w, d...) for d in data)/length(data)

function train(model, data, optim, lossgradient)
    for (x,y) in data
        grads = lossgradient(model,x,y)
        update!(model, grads, optim)
    end
end
# Here the optim argument specifies the optimization algorithm and state for each model parameter (see Optimization methods for available algorithms). update! uses optim to update each model parameter and optimization state. optim has the same size and shape as model, i.e. we have a separate optimizer for each model parameter. For simplicity we will use the optimizers function to create an Adam optimizer for each parameter:

function todata(trajs::Vector{Trajectory})
    xu = hcat(getfield.(trajs,:xu)...)
    y = hcat(getfield.(trajs,:y)...)
    minibatch(xu, y, length(trajs[1]))
end

function runtrain(w, lossfunner; epochs = 500)
    jacfun       = jacobian(x->pred(w,x))
    lossfun      = lossfunner(jacfun)
    lossgradient = grad(lossfun)
    o            = optimizers(w, Adam)
    dtrn         = todata(trajs)
    dtst         = todata([vt])
    trace        = History(Float64)
    tracev       = History(Float64)
    push!(trace, 0, lossfun(w,dtrn))
    push!(tracev, 0, cost(w,dtst))
    for epoch=1:epochs
        train(w, dtrn, o, lossgradient)
        if epoch % 5 == 0
            push!(trace, epoch, lossfun(w,dtrn))
            push!(tracev, epoch, cost(w,dtst))
            plot(trace, reuse=true)
            plot!(tracev)
            println(last(trace), last(tracev))
        end
    end
    trace,tracev
end

srand(1)
w1 = Any[ 0.1f0*randn(Float64,num_params,nx+nu), zeros(Float64,num_params,1),
0.1f0*randn(Float64,nx,num_params),  zeros(Float64,nx,1) ]
srand(1)
w2 = Any[ 0.1f0*randn(Float64,num_params,nx+nu), zeros(Float64,num_params,1),
0.1f0*randn(Float64,nx,num_params),  zeros(Float64,nx,1) ]

res = runtrain(w1, cost, epochs=10)
resj = runtrain(w2, loss, epochs=10)
