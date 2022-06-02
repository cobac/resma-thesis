module OccamsWindow

using DataFrames, GLM, Distributions, StatsAPI, StatsModels, StatsBase, BayesianLinearRegression, RCall

import StatsAPI.bic

export model_search, OccamsWindowParams, AbstractMarginalApproximation, BIC, Analytical

include("submodels.jl")
include("marginal_approximations.jl")
include("supported_models.jl")
include("cache.jl")
include("modelsets.jl")
include("hyperparams.jl")
include("solution.jl")
include("modelsearch.jl")
end # module
