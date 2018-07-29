default(grid=false) #src
using Parameters, ForwardDiff, LTVModelsBase, ValueHistories
const Diff = ForwardDiff
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


num_params = 20
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

function i2m(w,i,sizes)
    s = [1; cumsum([prod.(sizes)...]) .+ 1]
    reshape(view(w,s[i]:(s[i+1]-1)), sizes[i])
end

function pred(w,x,sizes)
    state = copy(x)
    for i=1:2:length(sizes)-2
        state = tanh.(i2m(w,i,sizes)*state .+ i2m(w,i+1,sizes))
    end
    return i2m(w,length(sizes)-1,sizes)*state .+ i2m(w,length(sizes),sizes)
end



cost(jacfun) = cost
cost(w,x,y)  = sum(abs2, pred(w,x,sizes) .- y)/size(y,2)
# cost(w,data) = sum(cost(w, d...) for d in data)/length(data)
cost(w,data) = cost(w, data...)

function loss(w,x,y)
    model(x)    = pred(w,x,sizes)
    jcfg        = Diff.JacobianConfig(model, x[:,1])
    jacobian(x) = Diff.jacobian(model, x, jcfg)
    function lf(w)
        l = cost(w,x,y)
        J2 = zeros(nx+nu, nx)
        J1 = jacobian(x[:,1])
        for t = 2:size(x,2)
            J2 = jacobian(x[:,t])
            l += sum(abs2.(J1.-J2))
            J1 = J2
        end
        l
    end
end

# TODO: take care of separate trajs
function todata(trajs::Vector{Trajectory})
    xu = hcat(getfield.(trajs,:xu)...)
    y = hcat(getfield.(trajs,:y)...)
    xu, y
end

function runtrain(w, loss; epochs = 500)
    dtrn        = todata(trajs)
    dtst        = todata([vt])
    x,y         = dtrn
    lossfun     = loss(w,x,y)
    gcfg        = Diff.GradientConfig(lossfun, w)
    gradient(w) = Diff.gradient(lossfun, w, gcfg)
    trace       = History(Float64)
    tracev      = History(Float64)
    push!(trace, 0, lossfun(w))
    push!(tracev, 0, cost(w,dtst))
    for epoch=1:epochs
        g = gradient(w)
        # @show size.((g,w))
        @. w -= 0.01g
        if epoch % 2 == 0
            push!(trace, epoch, lossfun(w))
            push!(tracev, epoch, cost(w,dtst))
            plot(trace, reuse=true)
            plot!(tracev)
            println(last(trace), last(tracev))
        end
    end
    trace,tracev
end

sizes = ((num_params,nx+nu), (num_params,1), (nx,num_params), (nx,1))
tovec(w) = vcat([vec(w) for w in w]...)

srand(1)
w1 = [ 0.1f0*randn(Float64,sizes[1]), zeros(Float64,sizes[2]),0.1f0*randn(Float64,sizes[3]),  zeros(Float64,sizes[4]) ]
w1 = tovec(w1)
w2 = deepcopy(w1)


# res = runtrain(w1, cost, epochs=10)
resj = runtrain(w2, loss, epochs=10)



w = copy(w1)
model(x)    = pred(w,x,sizes)
const dtrn        = todata(trajs)
const dtst        = todata([vt])
const x,y         = dtrn
const jcfg        = Diff.JacobianConfig(model, x[:,1])
jacobian(x) = Diff.jacobian(model, x, jcfg)

jacobian(x[:,1])

function test(w)
    l = cost(w,x,y)
    J2 = zeros(nx+nu, nx)
    J1 = jacobian(x[:,1])
    for t = 2:size(x,2)
        J2 = jacobian(x[:,t])
        l += sum(abs2.(J1.-J2))
        J1 = J2
    end
    l
end


test(w1)
gcfg = Diff.GradientConfig(test, w1)
testgradient(w1) = Diff.gradient(test, w1, gcfg)
testgradient(w1)
