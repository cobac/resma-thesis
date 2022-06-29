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


param_names(specs::blmModelSpecs) = coefnames(specs.s)[2][2:end]

# TODO: Don't fit for every model
marginal_likelihood(specs::blmModelSpecs, bits::BitVector, approximation::Analytical) =
    log(BayesianLinearRegression.marginal_likelihood(fit(specs, bits)))

function fit(specs::blmModelSpecs, bits::BitVector)
    vars = findall(bits) .+ 1
    # Always fit intercept
    pushfirst!(vars, 1)
    new_s = FormulaTerm(specs.s.lhs, StatsModels.MatrixTerm(specs.s.rhs.terms[vars]))
    return BayesianLinearRegression.blm(new_s, specs.X[:, vars], specs.y, specs.hyper)
end
