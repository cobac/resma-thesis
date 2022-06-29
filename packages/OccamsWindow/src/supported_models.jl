
# Supported model types need to implement a ModelSpecs struct generated from a model_specs() function,
# which contains all necessary information to fit the models and calculate their marginal likelihood

abstract type AbstractModelSpecs end

# They also need to implement the following functions if the defaults don't work

function model_specs(model::StatisticalModel)
    throw(ArgumentError(string("Unsupporded model type: ", typeof(model))))
end

function fit(specs::AbstractModelSpecs, bits::BitVector)
    throw(ArgumentError(string("fit() function not implemented for specs of type: ", typeof(specs))))
end

# And optionally a marginal_likelihood(model, marginal_approximation) method for each combination.

function param_names(specs::AbstractModelSpecs)
    throw(ArgumentError(string("param_names() function not implemented for specs of type: ", typeof(specs))))
end

supportsleaps(specs::AbstractModelSpecs) = false
function leaps_data end # if supportsleaps() is true

include("supported_models/blm.jl")
include("supported_models/lm.jl")
include("supported_models/ggm.jl")
