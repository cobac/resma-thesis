struct OccamsWindowSolution{F<:AbstractFloat,S<:AbstractModelSpecs,A<:AbstractMarginalApproximation}
    modelset::WeightedModelSet{F}
    specs::S
    approximation::A
    hyperparams::OccamsWindowParams{F}
    iters::Tuple{Int,Int,Int}
    coef_weights::Vector{F}
    timeout::Bool
end

function make_solution(accepted_models, cache, specs, hyperparams, down_iter, up_iter, marginal_approximation, timeout)
    length(accepted_models) == 0 &&
        error("The model search did not accept any model.")

    out_bits = collect(accepted_models)
    out_modelset = WeightedModelSet(out_bits, cache)

    no_params = length(param_names(specs))
    coef_weights = zeros(no_params)
    for bit in UnitRange(1, no_params)
        for model in eachindex(out_bits)
            if out_bits[model][bit]
                coef_weights[bit] += out_modelset.weights[model]
            end
        end
    end

    return OccamsWindowSolution(out_modelset,
        specs,
        marginal_approximation,
        hyperparams,
        (down_iter, up_iter, length(cache)),
        coef_weights,
        timeout)
end

function Base.show(io::IO, solution::OccamsWindowSolution)
    (; modelset, specs, approximation, iters, coef_weights) = solution
    println(io, "Occam's Window explored $(iters[3]) models in total for $(iters[1]) + $(iters[2]) = $(iters[1]+iters[2]) iterations, using the $approximation approximation to the marginal likelihood.")
    println(io, "")
    println(io, "Weight: model formula")
    println(io, "---------------------")
    show(io, modelset)
    println(io, "")
    println(io, "Parameter => posterior weight")
    println(io, "---------------------")
    coef_names = param_names(specs)
    show(io, Pair.(coef_names, coef_weights))
end
