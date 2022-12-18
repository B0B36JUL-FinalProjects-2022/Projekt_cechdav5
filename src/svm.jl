using LinearAlgebra
using COSMO

abstract type KernelSpecification end

struct LinearKernel <: KernelSpecification end

struct PolynomialKernel <: KernelSpecification 
    d::Integer
end

struct RBFKernel <: KernelSpecification 
    std::Real
end

computeKernel(X, ::LinearKernel) = (X*X')

computeKernel(X, pk::PolynomialKernel) = ((X*X') + ones((size(X, 1), size(X,1)))) .^ pk.d

function computeKernel(X, rk::RBFKernel)
    n = size(X,1)
    d = size(X,2)
    sq_dist = reshape(sum((reshape(X[1,:], (1,d)) .- X).^2, dims=2), (1,n))

    for i in 2:n
        curr_sq_dist = reshape(sum((reshape(X[i,:], (1,d)) .- X).^2, dims=2), (1,n))
        sq_dist = vcat(sq_dist, curr_sq_dist)
    end

    return exp.(-sq_dist ./ (2*rk.std^2))
end

#computeQ(X, y; kernel = LinearKernel()) = y'.*computeKernel(X, kernel).*y
 
function computeW(X, y, z)
    w = sum((y.*z.*X), dims=1)
    reshape(w, (length(w),))
end
 
function solve_SVM_dual(K, y, C)
    my_dim = size(K, 1)

    P = y'.*K.*y
    q = -ones((my_dim,))

    diagm(ones(3))
    #zi <= C
    c1 = COSMO.Constraint(-diagm(ones(my_dim)), C*ones((my_dim, 1)), COSMO.Nonnegatives, my_dim, 1:my_dim)
    #0 <= zi
    c2 = COSMO.Constraint(diagm(ones(my_dim)), zeros((my_dim, 1)), COSMO.Nonnegatives, my_dim, 1:my_dim)
    #y' * z = 0
    c3 = COSMO.Constraint(y', 0, COSMO.ZeroSet)

    model = COSMO.Model();
    assemble!(model, P, q, [c1; c2; c3])
    result = COSMO.optimize!(model);

    return result.x
end
 
function solve_SVM(X, y, C; kernel=LinearKernel(), kwargs...)
    K = computeKernel(X, LinearKernel())

    show(K)
    z = solve_SVM_dual(K, y, C; kwargs...)

    b = compute_bias(K, y, z, C)

    return z
end
 
function compute_bias(K, y, z, C)
    eps = 1e-10 #accounting for numerical error while computing dual solution
    zy = y.*z

    on_margin_mask = (z .> eps) .&& (z .< C) #support vectors exactly on margin
    if any(on_margin_mask)
        ids = collect(1:length(y))[on_margin_mask] #margin support vectors indices
    
        ct = count(on_margin_mask)
    
        
        # b = mean(ys - z*xs) where xs, ys are all support vectors
        bias = sum(y[ids] .- reshape(sum(zy .* K[:, ids]; dims=1), (ct,))) / length(ids)
    else #all support vectors violate the margin
        sv_mask = z .> eps
        not_sv_mask = (sv_mask) .== false

        e_i = y .- reshape(sum(zy .* K; dims=1), (length(y,)))

        LB_mask = (not_sv_mask .&& (y.==1)) .|| (sv_mask .&& (y.==-1))
        LB_val, _ = findmax(e_i[LB_mask])

        UB_mask = (sv_mask .&& (y.==1)) .|| (not_sv_mask .&& (y.==-1))
        UB_val, _ = findmin(e_i[UB_mask])

        bias = (LB_val + UB_val) / 2
    end

    return bias
end