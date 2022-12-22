using LinearAlgebra
using COSMO
using Random

abstract type KernelSpecification end

struct LinearKernel <: KernelSpecification end

struct PolynomialKernel <: KernelSpecification 
    d::Integer
end

struct RBFKernel <: KernelSpecification 
    std::Real
end

compute_kernel(Xi, Xj, ::LinearKernel) = (Xi*Xj')

compute_kernel(Xi, Xj, pk::PolynomialKernel) = ((Xi*Xj') + ones((size(Xi, 1), size(Xj,1)))) .^ pk.d

function compute_kernel(Xi, Xj, rk::RBFKernel)
    n = size(Xi,1)
    m = size(Xj,1)
    d = size(Xi,2)
    sq_dist = reshape(sum((reshape(Xi[1,:], (1,d)) .- Xj).^2, dims=2), (1,m))

    for i in 2:n
        curr_sq_dist = reshape(sum((reshape(Xi[i,:], (1,d)) .- Xj).^2, dims=2), (1,m))
        sq_dist = vcat(sq_dist, curr_sq_dist)
    end

    return exp.(-sq_dist ./ (2*rk.std^2))
end
 
function solve_SVM_dual(K::Matrix{<: Real}, y::Vector{<: Real}, C::Real; eps::Real=1e-6)
    my_dim = size(K, 1)

    P = y'.*K.*y
    q = -ones((my_dim,))

    my_settings = COSMO.Settings(eps_abs = 1e-2*eps, eps_rel = 1e-2*eps)

    diagm(ones(3))
    #zi <= C
    c1 = COSMO.Constraint(-diagm(ones(my_dim)), C*ones((my_dim, 1)), COSMO.Nonnegatives, my_dim, 1:my_dim)
    #0 <= zi
    c2 = COSMO.Constraint(diagm(ones(my_dim)), zeros((my_dim, 1)), COSMO.Nonnegatives, my_dim, 1:my_dim)
    #y' * z = 0
    c3 = COSMO.Constraint(y', 0, COSMO.ZeroSet)

    model = COSMO.Model();
    assemble!(model, P, q, [c1; c2; c3], settings=my_settings)
    result = COSMO.optimize!(model);

    return result.x
end
 
function solve_SVM(X, y::Vector{<: Integer}, C::Real; kernel::KernelSpecification=LinearKernel(), eps::Real=1e-6)
    K = compute_kernel(X, X, kernel)

    z = solve_SVM_dual(K, y, C; eps)

    b = compute_bias(K, y, z, C; eps)

    return Dict("z" => z[z .> eps], "bias" => b, "y"=>y[z .> eps], "sv" => X[z .> eps, :], "kernel" => kernel)
end
 
function compute_bias(K::Matrix{<: Real}, y::Vector{<: Integer}, z::Vector{<: Real}, C::Real; eps::Real=1e-6)
    zy = y.*z

    on_margin_mask = (z .> eps) .&& (z .< C-eps) #support vectors exactly on margin
    if any(on_margin_mask)
        ids = collect(1:length(y))[on_margin_mask] #margin support vectors indices
    
        ct = count(on_margin_mask)
    
        # b = mean(ys - z*xs) where xs, ys are all support vectors
        bias = sum(y[ids] .- reshape(sum(zy .* K[ids, :]'; dims=1), (ct,))) / length(ids)
    else #all support vectors violate the margin
        sv_mask = z .> eps
        not_sv_mask = (sv_mask) .== false

        e_i = y .- reshape(sum(zy .* K; dims=1), (length(y,)))

        LB_mask = (not_sv_mask .&& (y.==1)) .|| (sv_mask .&& (y.==-1))
        UB_mask = (sv_mask .&& (y.==1)) .|| (not_sv_mask .&& (y.==-1))

        LB_val = reduce((x,y) -> max.(x,y), e_i[LB_mask])        
        UB_val = reduce((x,y) -> min.(x,y), e_i[UB_mask])

        bias = (LB_val + UB_val) / 2
    end

    return bias
end

function classify_SVM(X, model::Dict)
    K = compute_kernel(X, model["sv"], model["kernel"])

    #w' * X + bias
    res = (K .* reshape(model["y"], (1, length(model["y"])))) * model["z"] .+ model["bias"]
    
    classif = ones(Int, size(res))
    classif[res .< 0] .= -1

    return classif
end

function hyperparam_cross_validation(X, y::Vector{<: Integer}; train_ratio::Float64=0.8, num_iter::Integer = 10, Cs=nothing, kernels=nothing)
    Cs = Cs === nothing ? [0.001, 0.1, 1, 10, 1000] : Cs
    kernels = kernels === nothing ? [LinearKernel(), PolynomialKernel(1), PolynomialKernel(3), 
    PolynomialKernel(5), RBFKernel(0.1), RBFKernel(1), RBFKernel(10), RBFKernel(20),
    RBFKernel(100), RBFKernel(1000)] : kernels

    best_err = Inf
    best_hyperparams = nothing

    for C in Cs
        for kern in kernels
            avg_error = 0
            for i in 1:num_iter
                trn_x, trn_y, tst_x, tst_y = random_data_split(X, y; train_ratio)
                model = solve_SVM(trn_x, trn_y, C; kernel = kern)
                labels = classify_SVM(tst_x, model)
                tmp = ones(length(tst_y))
                avg_error += sum(tmp[labels .!= tst_y])/length(tst_y)
            end
            avg_error /= num_iter

            if (avg_error < best_err)
                best_hyperparams = Dict("C" => C, "Kernel" => kern)
                best_err = avg_error
            end
        end
    end

    return best_err, best_hyperparams
end

function random_data_split(X, y::Vector{<: Integer}; train_ratio::Float64)
    n = length(y)
    cv_train_cnt = floor(Int, n*train_ratio)
    
    idx_perm = randperm(n)
    
    cv_train_x = X[idx_perm[1:cv_train_cnt], :]
    cv_train_y = y[idx_perm[1:cv_train_cnt]]
    cv_test_x = X[idx_perm[cv_train_cnt+1:n], :]
    cv_test_y = y[idx_perm[cv_train_cnt+1:n]]
        
    return cv_train_x, cv_train_y, cv_test_x, cv_test_y
end

function prepare_data_for_SVM(X, y::Vector{<: Integer})
    X = standardize_data(X)       # mean = 0, std = 1
    X = hcat(X, ones(size(X, 1))) # add bias

    y[y .== 0] .= -1; # transform 0 labels to -1

    return X, y 
end