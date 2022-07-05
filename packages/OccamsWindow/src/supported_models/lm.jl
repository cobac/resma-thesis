struct lmModelSpecs{F<:AbstractFloat,S<:AbstractString} <: AbstractModelSpecs
    X::Matrix{F}
    y::Vector{F}
    names::Vector{S}
end

supportsleaps(specs::Union{lmModelSpecs,blmModelSpecs}) = true
leaps_data(specs::Union{lmModelSpecs,blmModelSpecs}) = specs.X[:, 2:end], specs.y

#response(specs::lmModelSpecs) = specs.y

# We need to use the wrapped typed from StatsModels.jl because the model types from GLM.jl don't store the name of the variables
#get_model_type(model::StatsModels.TableRegressionModel) = typeof(model.model)

function get_tablemodel_type(m::StatsModels.TableRegressionModel{M,T}) where {M,T}
    return M
end

function model_specs(model::StatsModels.TableRegressionModel)
    model_type = get_tablemodel_type(model)
    if (model_type <: GLM.LinearModel)
        X = modelmatrix(model)
        y = StatsModels.response(model)
        # drop intercept name
        names = StatsAPI.coefnames(model)[2:end]
        return lmModelSpecs(X, y, names)
    else
        throw(ArgumentError("Model type not supported:$(typeof_model)"))
    end
end

param_names(specs::lmModelSpecs) = specs.names

marginal_likelihood(specs::lmModelSpecs, bits::BitVector, approximation::BIC) = -bic(fit(specs, bits)) / 2

function fit(specs::lmModelSpecs, bits::BitVector)
    vars = findall(bits) .+ 1
    # Always fit intercept
    pushfirst!(vars, 1)
    return GLM.fit(GLM.LinearModel, specs.X[:, vars], specs.y)
end
