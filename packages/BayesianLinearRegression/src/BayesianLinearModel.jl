struct BayesianLinearModel{F,M<:AbstractMatrix{F},V<:AbstractVector{F}} <: RegressionModel
    s::FormulaTerm
    X::M
    y::V
    hyperparams::BayesianLinearModelHyperparams{F}
end

function Base.show(io::IO, model::BayesianLinearModel)
    println(io, "Bayesian simple linear regression model with formula: ")
    show(io, model.s)
end


StatsBase.coefnames(model::BayesianLinearModel) = coefnames(model.s.rhs.terms)
