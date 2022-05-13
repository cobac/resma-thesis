module BayesianLinearRegression
using LinearAlgebra, StatsAPI, StatsBase, StatsModels, DataFrames
export BayesianLinearModel, blm, BayesianLinearModelHyperparams

include("hyperparams.jl")
include("BayesianLinearModel.jl")
include("blm.jl")
include("marginal_likelihood.jl")

end # module
