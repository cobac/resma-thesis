
const ModelSet = Set{BitVector}

function pop_rand!(ms::ModelSet)
    m = rand(ms)
    delete!(ms, m)
    return m
end


struct WeightedModelSet{F<:AbstractFloat}
    bits::Vector{BitVector}
    weights::Vector{F}

end

function WeightedModelSet(bits::Vector{BitVector}, cache::Cache)
    sort!(bits, by = x -> -cache[x])
    # TODO: Assumes uniform model priors for now
    marginal_total = mapreduce(x -> exp(big(cache[x])), +, bits)
    return WeightedModelSet(bits,
        [Float64(exp(big(cache[m])) / marginal_total) for m in bits])
end

Base.length(wms::WeightedModelSet) = length(wms.weights)
 

function Base.show(io::IO, weighted_models::WeightedModelSet)
    (; bits, weights) = weighted_models
    no_models = length(bits)
    for i in eachindex(bits)
        println(io, round(weights[i], digits = 4), ": ", bits[i])
        if i > 12
            println("... and $(no_models - i) more selected models.")
            break
        end
    end
end
