struct blmModelSpecs{F<:AbstractFloat} <: AbstractModelSpecs
    s::FormulaTerm
    X::Matrix{F}
    y::Vector{F}
    hyper::BayesianLinearRegression.BayesianLinearModelHyperparams
end


# supportsleaps() and leaps_data() defined in file lm.jl

function model_specs(model::BayesianLinearRegression.BayesianLinearModel)
    return blmModelSpecs(model.s, model.X, model.y, model.hyperparams)
end

function fit(specs::blmModelSpecs, bits::BitVector)
    vars = findall(bits)
    new_s = FormulaTerm(specs.s.lhs, StatsModels.MatrixTerm(specs.s.rhs.terms[vars]))
    return BayesianLinearRegression.blm(new_s, specs.X[:, vars], specs.y, specs.hyper)
end

marginal_likelihood(model::BayesianLinearRegression.BayesianLinearModel, approximation::Analytical) =
    log(BayesianLinearRegression.marginal_likelihood(model))


# function get_formula(bits, saturated_model::BayesianLinearRegression.BayesianLinearModel)
#     vars = findall(bits)
#     new_s = FormulaTerm(saturated_model.s.lhs, StatsModels.MatrixTerm(saturated_model.s.rhs.terms[vars]))
#     return string(new_s)
# end
