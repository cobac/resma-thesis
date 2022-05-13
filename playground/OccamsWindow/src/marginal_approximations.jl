abstract type AbstractMarginalApproximation end

function marginal_likelihood end

struct BIC <: AbstractMarginalApproximation end
marginal_likelihood(model::StatisticalModel, approximation::BIC) = -bic(model) / 2

abstract type AbstractLaplace <: AbstractMarginalApproximation end
struct TestLaplace <: AbstractLaplace end

marginal_likelihood(model::StatisticalModel, approximation::A) where {A<:AbstractLaplace} = error("Laplace approximation not yet implemented.")
# marginal_likelihood(model, Laplace{TestLaplace}())

struct Analytical <: AbstractMarginalApproximation end

marginal_likelihood(model::StatisticalModel, approximation::Analytical) =
    error("Analytical marginal likelihood not implemented for models with type: ", typeof(model))

struct ModelAndMarginal{M<:StatisticalModel,A<:AbstractMarginalApproximation}
    model::M
    approximation::A
    value
    # TODO ModelAndMarginal positive marginals check
    # function ModelAndMarginal{M, A}(model, approximation, value) where
    # {M <: StatisticalModel, A <: AbstractMarginalApproximation}
    # value < 0 && error("The value of the marginal likelihood has to be positive.")
    # return new(model, approximation, value)
    # end 
end

ModelAndMarginal(model::M, approximation::A) where {M<:StatisticalModel,A<:AbstractMarginalApproximation} =
    ModelAndMarginal(model, approximation, marginal_likelihood(model, approximation))

function Base.show(io::IO, model_and_marginal::ModelAndMarginal)
    println(io, "Model with marginal approximation: ",
        round(model_and_marginal.value, digits = 2), " by ", model_and_marginal.approximation)
end
