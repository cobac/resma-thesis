struct OccamsWindowSolution{F<:AbstractFloat,M<:StatisticalModel,MS<:StatisticalModel,A<:AbstractMarginalApproximation}
    modelset::WeightedModelSet{M}
    saturated_model::MS
    approximation::A
    down_iters::Int
    up_iters::Int
    coef_weights::Vector{F}
end

function Base.show(io::IO, solution::OccamsWindowSolution)
    (; modelset, saturated_model, approximation, down_iters, up_iters, coef_weights) = solution
    println(io, "Occam's Window executed for $down_iters + $up_iters = $(down_iters+up_iters) iterations, using the $approximation approximation to the marginal likelihood.")
    println(io, "")
    println(io, "Weight: model formula")
    println(io, "---------------------")
    show(io, modelset, saturated_model)
    println(io, "---------------------")
    println(io, "Parameter: posterior weight")
    println(io, "---------------------")
    coef_names = get_coef_bits(saturated_model).names
    show(io, Pair.(coef_names, coef_weights))
end

function Base.show(io::IO, weighted_models::WeightedModelSet, saturated_model::StatisticalModel)
    (; bits, weights) = weighted_models
    for i in eachindex(bits)
        println(io, round(weights[i], digits = 4), ": ", get_formula(bits[i], saturated_model))
    end
end

function get_formula(bits, saturated_model)
    f = formula(saturated_model)
    return f.lhs ~ foldl(+, f.rhs.terms[bits])
end
