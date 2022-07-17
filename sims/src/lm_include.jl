import Pkg
Pkg.activate(".")

using OccamsWindow
#ENV["JULIA_DEBUG"] = OccamsWindow
using RCall, DataFrames, GLM, Distributions, Random, JLD2, ProgressMeter
import InteractiveUtils.subtypes

abstract type AbstractSimType end

abstract type SimType <: AbstractSimType end

struct BAS <: SimType end
struct BMA <: SimType end
struct OccamsLeaps <: SimType end
struct OccamsSat <: SimType end
struct OccamsSatNoWindow <: SimType end
struct OccamsLeapsNoWindow <: SimType end

function generate_y(no_vars, n, x, β)
    is = sample(1:size(x, 2), no_vars, replace = false)
    y = β[1] .+ x[:, is] * β[is.+1] .+ rand(Normal(), n)
    return y, sort(is)
end

function generate_data(n, p, no_vars)
    x = rand(Normal(), (n, p))
    β = rand(Normal(0, 10), p + 1)
    y, true_is = generate_y(no_vars, n, x, β)
    return x, y, true_is
end

function fit_occams(x, y, startup, Oₗ = log(20.0))
    df = DataFrame(x, :auto)
    df[!, :y] = y

    saturated_lm = lm(
        term(:y) ~ term(1) +
                   foldl(+, [term(Symbol("x" * string(i))) for i in 1:(size(x, 2))]),
        df)

    solution_lm = model_search(saturated_lm, BIC(),
        hyperparams = OccamsWindowParams(startup = startup, Oₗ = Oₗ), max_time = 10)

    models = reduce(hcat, solution_lm.modelset.bits)' |> collect |> BitMatrix
    #solution_lm.timeout && @info "Model search timed out"
    return models, solution_lm.modelset.weights, solution_lm.coef_weights, solution_lm.timeout
end

fit_model(x, y, model_type::OccamsLeaps) = fit_occams(x, y, :leaps)
fit_model(x, y, model_type::OccamsSat) = fit_occams(x, y, :saturated)
fit_model(x, y, model_type::OccamsLeapsNoWindow) = fit_occams(x, y, :leaps, 0.0)
fit_model(x, y, model_type::OccamsSatNoWindow) = fit_occams(x, y, :saturated, 0.0)

function fit_model(x, y, model_type::BAS)
    df = DataFrame(x, :auto)
    df[!, :y] = y
    @rput df

    R"library(BAS)"
    R"bas_solution <- bas.lm(y ~ 1 + ., data = df)"
    R"p_models <- bas_solution$postprobs"
    R"p_param <- bas_solution$probne0[-1]" # drop intercept
    R"model_is <- bas_solution$which"
    @rget model_is
    @rget p_models
    @rget p_param

    models = falses(length(p_models), length(p_param))
    for model in eachindex(model_is)
        model_is[model] == 0 && continue
        for i in model_is[model]
            i == 0 && continue
            models[model, i] = true
        end
    end

    return models, p_models, p_param, false
end

function fit_model(x, y, model_type::BMA)
    @rput x
    @rput y

    R"library(BMA)"
    R"bma_solution <- bicreg(x, y)"
    R"models <- bma_solution$which"
    R"p_models <- bma_solution$postprob"
    R"p_param <- bma_solution$probne0"

    @rget models
    @rget p_models
    @rget p_param

    return models, p_models, p_param ./ 100, false
end

function one_sim(sim_id, n, p, no_vars, model_type)
    # Distributed safe seeding
    Random.seed!(parse(Int, prod(lpad.([sim_id, n, p, no_vars], 4, "0"))))
    x, y, true_is = generate_data(n, p, no_vars)
    return true_is, fit_model(x, y, model_type)...
end

function save_sim!(sim_id, p, obs, prop_p, model_type, true_is, models, models_p, param_p, timeout)
    model_name = string(typeof(model_type))
    jldsave("output/$(model_name)_$(p)_$(obs)_$(prop_p)_$(sim_id).jld2";
        true_is, models, models_p, param_p, timeout)
    return nothing
end

macro parse_type(str)
    :($(Symbol(eval(str)))())
end
