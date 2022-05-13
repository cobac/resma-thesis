# β ~ Normal(μ, σ²V)
# (vλ)/(σ²) ~ χ²ᵥ
# By default using hyperparameters from Raftery & Madigan (1997) doi:10.1080/01621459.1997.10473615

Base.@kwdef struct BayesianLinearModelHyperparams{F<:AbstractFloat}
    μ::F = 0.0 # Mean of the prior of the slopes
    v::F = 2.58
    λ::F = 0.28
    ϕ::F = 2.85
end

function generate_μ(μ, s, X, y)
    if (hasintercept(s))
        return [ols_intercept(X, y); fill(μ, size(X)[2] - 1)]
    else
        return fill(μ, size(X)[2])
    end
end

ols_intercept(X, y) = (y\X)[1]

function generate_V(ϕ, s, X, y)
    v = map(eachcol(X)) do x
        ϕ^2 * inv(var(x, corrected = false))
    end
    if (hasintercept(s))
        v[1] = var(y)
    end
    return Diagonal(v)
end

# TODO: Piracy!
function hasintercept(s)
    first_term = s.rhs.terms[1]
    return typeof(first_term) <: StatsModels.InterceptTerm
end 
