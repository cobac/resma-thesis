abstract type AbstractMarginalApproximation end

function marginal_likelihood end

struct BIC <: AbstractMarginalApproximation end
marginal_likelihood(model::StatisticalModel, approximation::BIC) = -bic(model) / 2

abstract type AbstractLaplace <: AbstractMarginalApproximation end
struct TestLaplace <: AbstractLaplace end

marginal_likelihood(model::StatisticalModel, approximation::AbstractLaplace) =
    error("Laplace approximation not yet implemented.")

struct Analytical <: AbstractMarginalApproximation end

marginal_likelihood(model::StatisticalModel, approximation::Analytical) =
    error("Analytical marginal likelihood not implemented for models with type: ", typeof(model))

