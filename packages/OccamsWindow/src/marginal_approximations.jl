abstract type AbstractMarginalApproximation end

"""
    FakeMarginal
Random values as marginal approximation. Useful for testing.
"""
struct FakeMarginal <: AbstractMarginalApproximation end

struct BIC <: AbstractMarginalApproximation end

abstract type AbstractLaplace <: AbstractMarginalApproximation end
struct TestLaplace <: AbstractLaplace end

marginal_likelihood(specs, bits::BitVector, approximation::AbstractLaplace) =
    throw(ArgumentError("Laplace approximation not yet implemented."))

struct Analytical <: AbstractMarginalApproximation end

marginal_likelihood(specs, bits::BitVector, approximation::Analytical) =
    throw(ArgumentError("Analytical marginal likelihood not implemented for models with type: $(typeof(specs))"))
