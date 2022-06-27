
const Cache = Dict{BitVector,Float64}

function get_ml!(bits::BitVector,
    specs::AbstractModelSpecs,
    marginal_approximation::AbstractMarginalApproximation,
    cache::Cache)
    haskey(cache, bits) && return cache[bits]
    ml = marginal_likelihood(fit(specs, bits), marginal_approximation)
    cache[bits] = ml
    return ml
end
