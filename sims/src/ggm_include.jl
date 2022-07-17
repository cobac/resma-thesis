import Pkg
Pkg.activate(".")

using OccamsWindow
#ENV["JULIA_DEBUG"] = OccamsWindow
using RCall, DataFrames, SimpleGGM, Distributions, Random, JLD2, ProgressMeter, LinearAlgebra
import InteractiveUtils.subtypes

abstract type AbstractSimType end

abstract type GGMSimType <: AbstractSimType end

struct Occams <: GGMSimType end
struct BDgraph <: GGMSimType end
struct BGGM <: GGMSimType end

function generate_data(p, sparse, n)
    no_param = binomial(p, 2)
    no_ones = floor(no_param * (1 - sparse)) |> Int
    one_is = sample(1:no_param, no_ones, replace = false)
    kappa = zeros(p, p)
    for one_i in one_is
        index = OccamsWindow.get_indices(one_i, p)
        set_ones!(kappa, index)
    end

    for edge in eachindex(kappa)
        if isone(kappa[edge])
            kappa[edge] = rand(Uniform(0.5, 1)) * ((-1)^(sample(1:2)))
        end
    end

    kappa[diagind(kappa)] .=
        sum((x -> abs.(x)).((collect(eachrow(kappa))))) .* 1.5
    kappa_edges = copy(kappa)
    kappa[diagind(kappa)] = replace!(diag(kappa), 0 => 1)
    kappa = hcat((eachrow(kappa) ./ diag(kappa))...)
    for i in 1:p, j in 1:p
        if i > j
            kappa[i, j] = kappa[j, i] = (kappa[i, j] + kappa[j, i]) / 2
        end
    end

    if !ishermitian(kappa)
        # Propagate the error to the main process
        error("kappa not hermitian: ", kappa_edges)
    end

    dist = MvNormalCanon(kappa)
    x = rand(dist, n)' |> collect

    return sort(one_is), x
end

function set_ones!(target, indices)
    target[indices...] = 1
    target[reverse(indices)...] = 1
end

function fit_model(x, model_type::Occams)
    df = DataFrame(x, :auto)
    model = ggm(df)
    hyper = OccamsWindowParams(startup = :saturated)
    # marginal_approximation = OccamsWindow.FakeMarginal()
    marginal_approximation = BIC()
    solution = model_search(model, marginal_approximation, hyperparams = hyper, max_time = 3600)
    models = reduce(hcat, solution.modelset.bits)' |> collect |> BitMatrix
    return models, solution.modelset.weights, solution.coef_weights, solution.timeout
end

function fit_model(x, model_type::BDgraph)
    R"library(BDgraph)"
    @rput x
    R"bdgraph_search <- bdgraph(x, cores = 1)"
    R"mat_p <- bdgraph_search$p_links"
    @rget mat_p

    param_p = Float64[]
    # For the life of me I don't get how to to this with CartesianIndices
    # without Tuple.(vec(collect(CartesianIndices(
    for i in 1:size(mat_p, 1), j in 1:size(mat_p, 2)
        if i < j
            push!(param_p, mat_p[i, j])
        end
    end
    return nothing, nothing, param_p, false
end

function fit_model(x, model_type::BGGM)
    R"library(BGGM)"
    @rput x
    R"bggm_search <- explore(x)"
    R"bf_mat <- select(bggm_search)$BF_10"
    @rget bf_mat

    param_bf = Float64[]
    # For the life of me I don't get how to to this with CartesianIndices
    # without Tuple.(vec(collect(CartesianIndices(
    for i in 1:size(bf_mat, 1), j in 1:size(bf_mat, 2)
        if i < j
            push!(param_bf, bf_mat[i, j])
        end
    end
    return nothing, nothing, param_bf, false
end

function one_sim(sim_id, n, p, sparse, model_type)
    # Distributed safe seeding
    sparse < 1 || error("sparse must be less than 1")
    Random.seed!(parse(Int, prod(lpad.([sim_id, n, p], 4, "0"))))
    true_is, x = generate_data(p, sparse, n)
    return true_is, fit_model(x, model_type)...
end

function save_sim!(model_type, sim_id, p, n, sparse, true_is, _, _, param_p, timeout)
    model_name = string(typeof(model_type))
    sparse = split(string(float(sparse)), ".")[2] # 1 => 0, 0.x => x
    jldsave("output/$(model_name)_$(p)_$(n)_$(sparse)_$(sim_id).jld2";
        true_is, param_p)
    return nothing
end

function save_sim!(model_type::Occams, sim_id, p, n, sparse, true_is, models, models_p, param_p, timeout)
    model_name = string(typeof(model_type))
    sparse = split(string(float(sparse)), ".")[2] # 1 => 0, 0.x => x
    jldsave("output/$(model_name)_$(p)_$(n)_$(sparse)_$(sim_id).jld2";
        true_is, models, models_p, param_p, timeout)
    return nothing
end

macro parse_type(str)
    :($(Symbol(eval(str)))())
end
