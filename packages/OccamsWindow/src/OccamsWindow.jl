module OccamsWindow

using DataFrames, GLM, Distributions, StatsAPI, StatsModels, StatsBase, BayesianLinearRegression

import StatsAPI.bic

export model_search, OccamsWindowParams, AbstractMarginalApproximation, BIC, Analytical

include("marginal_approximations.jl")
include("submodels.jl")
include("modelsets.jl")
include("supported_models.jl")
include("hyperparams.jl")
include("solution.jl")
include("modelsearch.jl")
end # module
