function marginal_likelihood(model::BayesianLinearModel)
    (; s, X, y, hyperparams) = model
    (; μ, v, λ, ϕ) = hyperparams
    n = size(X, 2)
    μ_vector = generate_μ(μ, s, X, y)
    V = generate_V(ϕ, s, X, y)
    IXVXt = I + X * V * transpose(X)
    yXμ = y - X * μ_vector
    return det(IXVXt)^(-1 // 2) *
           (λ * v +
            transpose(yXμ) *
            inv(IXVXt) *
            (yXμ))^((v + n) / 2)
end 
