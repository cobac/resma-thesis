
# Supported model types need to implement a ModelSpecs struct generated from a model_specs() function,
# which contains all necessary information to fit the models and calculate their marginal likelihood

abstract type AbstractModelSpecs end

function model_specs(model::StatisticalModel)
    throw(ArgumentError("Unsupported model type: $(typeof(model))"))
end

# They also need to implement the following functions

"""
    marginal_likelihood(specs::AbstractModelSpecs, bits::BitVector, marginal_approximation::AbstractMarginalApproximation)

Compute the marginal likelihood of the model codified by `bits`.
"""
function marginal_likelihood(specs::AbstractModelSpecs,
    bits::BitVector,
    marginal_approximation::AbstractMarginalApproximation)
    throw(ArgumentError("Unsupported combination of model specs ($(typeof(specs))) and marginal approximation ($(typeof(marginal_approximation)))."))
end

marginal_likelihood(specs::AbstractModelSpecs, bits::BitVector, approximation::FakeMarginal) =
    rand(append!(fill(0.2, 10), 0.8))

"""
    param_names(specs::AbstractModelSpecs)

Returns a vector of strings with the parameter names.
"""
function param_names(specs::AbstractModelSpecs)
    throw(ArgumentError("param_names() function not implemented for specs of type: $(typeof(specs))"))
end

supportsleaps(specs::AbstractModelSpecs) = false

"""
    leapsdata

Returns a tuple with the predictors matrix and response vector.
"""
function leaps_data end # if supportsleaps() is true

include("supported_models/blm.jl")
include("supported_models/lm.jl")
include("supported_models/ggm.jl")
