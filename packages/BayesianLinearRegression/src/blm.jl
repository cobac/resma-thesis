# TODO: Generalize to Tables.jl interface like StatsModels does
function blm(f::FormulaTerm, data::AbstractDataFrame,
    hyperparams::BayesianLinearModelHyperparams = BayesianLinearModelHyperparams())
    s = apply_schema(f, schema(f, data), BayesianLinearModel)
    resp, preds = modelcols(s, data)
    return fit(BayesianLinearModel, s, preds, resp, hyperparams)
end

function blm(s::FormulaTerm, preds, resp, hyperparams::BayesianLinearModelHyperparams = BayesianLinearModelHyperparams())
    length(s.rhs.terms) == size(preds)[2] || error("Mismatched dimensions between schema and predictors matrix.")
    return fit(BayesianLinearModel, s, preds, resp, hyperparams)
end 

# TODO: Ideally the model object would be more output-like with information about parameters
function StatsAPI.fit(type::Type{BayesianLinearModel}, s::FormulaTerm, X, y, hyperparams::BayesianLinearModelHyperparams)
    return BayesianLinearModel(s, X, y, hyperparams)
end

StatsModels.modelmatrix(model::BayesianLinearModel) = model.X
StatsModels.response(model::BayesianLinearModel) = model.y
