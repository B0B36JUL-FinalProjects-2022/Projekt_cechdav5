using LinearAlgebra
using COSMO

computeQ(X, y; kernel=0) = y'.*(X*X').*y
 
function computeW(X, y, z)
    w = sum((y.*z.*X), dims=1)
    reshape(w, (length(w),))
end
 
function solve_SVM_dual(Q, y, C)
    my_dim = size(Q, 1)

    P = Q
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
 
function solve_SVM(X, y, C; kwargs...)
    Q = computeQ(X, y)
    z = solve_SVM_dual(Q, y, C; kwargs...)
    print(z)
    return computeW(X, y, z)
end
 
f(Q, z) = -0.5 * z' * Q * z + sum(z)
