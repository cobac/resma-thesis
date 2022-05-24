struct lmModelSpecs{F<:AbstractFloat} <: AbstractModelSpecs
    X::Matrix{F}
    y::Vector{F}
end

function predictors(specs::lmModelSpecs)
    x = specs.X
    if (all(x[:, 1] .== 1.0))
        has_intercept = true
        x = x[:, 2:end]
    end
    return x, has_intercept
end

response(specs::lmModelSpecs) = specs.y

# We need to use the wrapped typed from StatsModels.jl because the model types from GLM.jl don't store the name of the variables
get_model_type(model::StatsModels.TableRegressionModel) = typeof(model.model)

function get_tablemodel_type(m::StatsModels.TableRegressionModel{M,T}) where {M,T}
    return M
end

function get_model_specs(model::StatsModels.TableRegressionModel)
    model_type = get_tablemodel_type(model)
    if (model_type <: GLM.LinearModel)
        X = modelmatrix(model)
        y = response(model)
        return lmModelSpecs(X, y)
    else
        error("Model type not supported: ", typeof(model))
    end
end

function fit(specs::lmModelSpecs, bits)
    vars = findall(bits)
    return GLM.fit(GLM.LinearModel, specs.X[:, vars], specs.y)
end
