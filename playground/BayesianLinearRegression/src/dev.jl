using BayesianLinearRegression, StatsModels, Distributions, DataFrames, Random

p = 10 # no. params
n = 5000 # no. obs

Random.seed!(1)
x = rand(Float64, (n, p));
X = [ones(n) x];
β = rand(Normal(0, 10), p + 1);

function generate_data(no_vars, X, β)
    is = sample(1:size(X)[2], no_vars, replace = false)
    y = X[:, is] * β[is] .+ rand(Normal(), n)
    return y, sort(is .- 1)
end

y, true_is = generate_data(11, X, β)

df = DataFrame(x, [Symbol("x" * string(i)) for i in 1:p]);
df[!, :y] = y;

f = @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9)
f_saturated = term(:y) ~ term(1) + foldl(+, [term(Symbol("x" * string(i))) for i in 1:p])

params = BayesianLinearModelHyperparams()

model = blm(f, df, params)
saturated_model = blm(f_saturated, df, params)

ml = BayesianLinearRegression.marginal_likelihood(model)
BayesianLinearRegression.marginal_likelihood(model)

saturated_ml = BayesianLinearRegression.marginal_likelihood(saturated_model)

(; s, X, y, hyperparams) = model
(; μ, v, λ, ϕ) = hyperparams
n = size(X, 2)
μ_vector = BayesianLinearRegression.generate_μ(μ, s, X, y)
V = BayesianLinearRegression.generate_V(ϕ, s, X, y)
IXVXt = I + X * V * transpose(X)
yXμ = y - X * μ_vector
det(IXVXt)^(1 // 2) *
(λ * v +
 transpose(yXμ) *
 inv(IXVXt) *
 yXμ)^((v + n) / 2)
