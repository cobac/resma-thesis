struct OccamsWindowSolution{F<:AbstractFloat,N,M<:StatisticalModel,MS<:StatisticalModel,A<:AbstractMarginalApproximation}
    modelset::WeightedModelSet{N,M}
    saturated_model::MS
    approximation::A
    down_iters::Int
    up_iters::Int
    coef_weights::Vector{F}
end

function OccamsWindowSolution(modelset::WeightedModelSet,
                              saturated_models::StatisticalModel,
                              approximation::AbstractMarginalApproximation,
                              down_iters::Int,
                              up_iters::Int)
    N = length(modelset.bits[1])
    coef_weights = zeros(N)
    for bit in 1:N
        for model in eachindex(modelset.models)
            if modelset.bits[model][bit]
                coef_weights[bit] += modelset.weights[model]
            end
        end 
    end
    return OccamsWindowSolution(modelset,
                                saturated_models,
                                approximation,
                                down_iters,
                                up_iters,
                                coef_weights)
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
