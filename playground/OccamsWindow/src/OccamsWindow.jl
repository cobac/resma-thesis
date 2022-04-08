module OccamsWindow

using DataFrames, GLM, Distributions, StatsAPI

import StatsAPI.bic

p = 10 # no. params
n = 500 # no. obs

x = rand(Float64, (n, p))
X = [ones(n) x]
β = rand(Normal(0, 10), p + 1)
y = X * β .+ rand(Normal(), n)

df = DataFrame(x, :auto)
df[!, :y] = y

saturated_model = lm(term(:y) ~ term(1) + foldl(+, [term(Symbol("x" * string(i)))
                                                    for i in 1:p]),
    df)

model = lm(@formula(y ~ x1 + x2), df)

abstract type AbstractMarginalApproximation end

function marginal_likelihood end

struct BIC <: AbstractMarginalApproximation end
marginal_likelihood(model, approximation::BIC) = bic(model)

abstract type AbstractLaplace end
struct Laplace{T} <: AbstractMarginalApproximation where {T<:AbstractLaplace} end
struct TestLaplace <: AbstractLaplace end

marginal_likelihood(model, approximation::Laplace{TestLaplace}) = @info "hey"
# marginal_likelihood(model, Laplace{TestLaplace}())

struct ModelAndMarginal{M<:StatisticalModel,A<:AbstractMarginalApproximation}
    model::M
    approximation::A
    value
    # TODO
    # function ModelAndMarginal{M, A}(model, approximation, value) where
    # {M <: StatisticalModel, A <: AbstractMarginalApproximation}
    # value < 0 && error("The value of the marginal likelihood has to be positive.")
    # return new(model, approximation, value)
    # end 
end

ModelAndMarginal(model::StatisticalModel, approximation::AbstractMarginalApproximation) =
    ModelAndMarginal(model, approximation, marginal_likelihood(model, approximation))

function Base.show(io::IO, model_and_marginal::ModelAndMarginal)
    print(io, formula(model_and_marginal.model), "\nMarginal approximation: ",
        round(model_and_marginal.value, digits = 2), " by ", model_and_marginal.approximation)
end

struct WeightedModelSet
    models::Vector{StatisticalModel}
    weights::Vector{AbstractFloat}
    function WeightedModelSet(models, weights)
        length(models) != length(weights) &&
            error("The length of the set of models and the set of weights has to match.")
        !(sum(weights) ≈ 1) && error("The sum of all weights has to be 1.")
        return new(models, weights)
    end
end

function WeightedModelSet(models::Vector{ModelAndMarginal{M,A}}) where
{M<:StatisticalModel,A<:AbstractMarginalApproximation}
    # Assumes uniform model priors for now
    marginal_total = mapreduce(x -> x.value, +, models)
    return WeightedModelSet([m.model for m in models],
        [m.value / marginal_total for m in models])
end

function Base.show(io::IO, weighted_models::WeightedModelSet)
    (; models, weights) = weighted_models
    for i in eachindex(models)
        print(io, "\n", round(weights[i], digits = 2), ": ",
            formula(models[i])
        )
    end
end

StatsAPI.predict(weighted_models::WeightedModelSet, x) =
    weighted_models.weights .* [predict(m, x) for m in weighted_models.models]

struct NamedBits{N}
    bits::BitArray{N}
    names::Array{String,N}
    # TODO
    # function NamedBits(bits, names)
    #     size(bits) != size(names) && error("Mismatch dimensions between bits and names.")
    #     return new{typeof(bits)}(bits, names)
    # end
end

function get_coef_bits(model::StatisticalModel)
    names = StatsAPI.coefnames(model)
    return NamedBits(BitVector(ones(Int, length(names))), names)
end

saturated_bits = get_coef_bits(saturated_model)

function get_coef_bits(saturated_bits::NamedBits, model::StatisticalModel)
    all_names = saturated_bits.names
    model_names = StatsAPI.coefnames(model)
    return all_names .∈ (model_names,)
end


# TODO: Deal with is_submodel(x, x)
function is_submodel(sub_bits::BitArray{N}, super_bits::BitArray{N}) where {N}
    sub_params = foldl(+, sub_bits)
    sum_and = mapreduce(x -> x[1] && x[2], +, zip(sub_bits, super_bits))
    return sub_params == sum_and
end

sub_bits = get_coef_bits(saturated_bits, model)
super_bits = get_coef_bits(saturated_bits, saturated_model)

@assert is_submodel(sub_bits, super_bits) == true

bits_1 = BitVector([0, 1, 0, 0, 0, 1, 0, 0, 0, 1,])
@assert is_submodel(bits_1, super_bits) == true
@assert is_submodel(bits_1, sub_bits) == false

bits_2 = BitVector([1, 1, 0, 0, 0, 0, 0, 1, 0, 0,])
@assert is_submodel(bits_2, super_bits) == true
@assert is_submodel(bits_2, sub_bits) == false

bits_3 = BitVector([0, 0, 0, 0, 0, 0, 0, 1, 0, 0,])
@assert is_submodel(bits_3, super_bits) == true
@assert is_submodel(bits_3, sub_bits) == false
@assert is_submodel(bits_3, bits_2) == true

end # module
