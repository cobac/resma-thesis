using OccamsWindow
using BayesianLinearRegression
using DataFrames, GLM, Distributions, BenchmarkTools, Random
ENV["JULIA_DEBUG"] = OccamsWindow
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

y, true_is = generate_data(5, X, β)

df = DataFrame(x, [Symbol("var_" * string(i)) for i in 1:p]);
df[!, :y] = y;

saturated_lm = lm(
    term(:y) ~ term(1) + foldl(+, [term(Symbol("var_" * string(i))) for i in 1:p]),
    df);


#saturated_blm = blm(term(:y) ~ term(1) + foldl(+, [term(Symbol("var_" * string(i))) for i in 1:p]), df)

@time solution_lm = model_search(saturated_lm, BIC())
# @time solution_blm, accepted_blm = model_search(saturated_blm, Analytical())

