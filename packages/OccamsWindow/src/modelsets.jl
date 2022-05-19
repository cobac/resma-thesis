struct ModelSet{MA<:ModelAndMarginal}
    dict::Dict{BitVector,MA}
end

emptyModelSet(MA) = ModelSet(Dict{BitVector,MA}())
ModelSet(xs::Vector) = ModelSet(Dict(xs))

function Base.show(io::IO, ms::ModelSet)
    for k in keys(ms.dict)
        println(io, k)
    end
end

Base.length(ms::ModelSet) = length(ms.dict)
Base.rand(ms::ModelSet) = rand(ms.dict)

function pop_rand!(ms::ModelSet)
    m = rand(ms)
    delete!(ms, first(m))
    return m
end

Base.isempty(ms::ModelSet) = isempty(ms.dict)
Base.keys(ms::ModelSet) = keys(ms.dict)
Base.haskey(ms::ModelSet, k) = haskey(ms.dict, k)
Base.delete!(ms::ModelSet, m) = delete!(ms.dict, m)
Base.getindex(ms::ModelSet, i) = getindex(ms.dict, i)
Base.setindex!(ms::ModelSet, k, val) = setindex!(ms.dict, k, val)

Base.iterate(ms::ModelSet) = iterate(ms.dict)
Base.iterate(ms::ModelSet, xs...) = iterate(ms.dict, xs...)

Base.copy(ms::ModelSet) = ModelSet(copy(ms.dict))

struct WeightedModelSet{M<:StatisticalModel,F<:AbstractFloat}
    bits::Vector{BitVector}
    models::Vector{M}
    weights::Vector{F}
    # TODO: WeightedModelSet length weights and sum of weights check
    # function WeightedModelSet{M}(models::Vector{M}, weights::Vector{F}) where {M <: StatisticalModel, F <: AbstractFloat}
    #     length(models) != length(weights) &&
    #         error("The length of the set of models and the set of weights has to match.")
    #     !(sum(weights) ≈ 1) && error("The sum of all weights has to be 1.")
    #     return new(models, weights)
    # end
end

# WeightedModelSet(models::Vector{T}, weights::Vector{F}) where {T <: StatisticalModel, F <: AbstractFloat} = WeightedModelSet(models, weights)

function WeightedModelSet(bits::Vector{BitVector}, models::Vector{ModelAndMarginal{M,A}}) where
{M<:StatisticalModel,A<:AbstractMarginalApproximation}
    sort!(models, by = x -> -x.value)
    # Assumes uniform model priors for now
    # TODO: Allow for custom prios
    marginal_total = mapreduce(x -> exp(big(x.value)), +, models)
    return WeightedModelSet(bits,
        [m.model for m in models],
        [Float64(exp(big(m.value)) / marginal_total) for m in models])
end

Base.length(wms::WeightedModelSet) = length(wms.weights)

StatsAPI.predict(weighted_models::WeightedModelSet, x) =
    weighted_models.weights .* [predict(m, x) for m in weighted_models.models]
