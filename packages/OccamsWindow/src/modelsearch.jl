
function model_search(saturated_model::StatisticalModel,
    marginal_approximation::AbstractMarginalApproximation;
    hyperparams::OccamsWindowParams = OccamsWindowParams())

    specs = model_specs(saturated_model)
    no_params = length(param_names(specs))
    (; Oᵣ, Oₗ, startup) = hyperparams

    # All explored models
    cache = Cache()

    # 𝓐  from Madigan & Raftery (1994)
    accepted_models = ModelSet()
    # 𝒞 from Madigan & Raftery (1994)
    candidate_models = ModelSet(starting_models(startup, specs))
    # Down pass
    down_iter = 0
    while !isempty(candidate_models)
        down_iter += 1
        @debug "\rDown pass iter: $down_iter"
        m = pop_rand!(candidate_models)
        ml = get_ml!(m, specs, marginal_approximation, cache)
        push!(accepted_models, m)
        sum(m) == 1 && continue
        m₀s = sample_submodels(m)
        for m₀ in m₀s
            B = get_ml!(m₀, specs, marginal_approximation, cache) - ml
            if B > Oᵣ
                delete!(accepted_models, m)
                if !(m₀ ∈ candidate_models)
                    push!(candidate_models, m₀)
                end
            elseif Oₗ <= B <= Oᵣ && !(m₀ ∈ candidate_models)
                push!(candidate_models, m₀)
            end
        end
    end # Down pass

    # Up pass
    up_iter = 0
    candidate_models = accepted_models
    accepted_models = ModelSet()
    while !isempty(candidate_models)
        up_iter += 1
        @debug "\rUp pass iter: $up_iter"
        m = pop_rand!(candidate_models)
        push!(accepted_models, m)
        ml = get_ml!(m, specs, marginal_approximation, cache)
        m₁s = sample_supermodels(m)
        for m₁ in m₁s
            B = ml - get_ml!(m₁, specs, marginal_approximation, cache)
            if B < Oₗ
                delete!(accepted_models, m)
                if !(m₁ ∈ candidate_models)
                    push!(candidate_models, m₁)
                end
            elseif Oₗ <= B <= Oᵣ && !(m₁ ∈ candidate_models)
                push!(candidate_models, m₁)
            end
        end
    end # Up pass

    length(accepted_models) == 0 &&
        error("The model search did not accept any model.")

    out_bits = collect(accepted_models)
    out_modelset = WeightedModelSet(out_bits, cache)

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
        coef_weights)
end

function starting_models(startup::Symbol, specs::AbstractModelSpecs)
    no_params = length(param_names(specs))
    if startup == :saturated
        bits₀ = (BitVector(fill(true, no_params)),)
    elseif startup == :random
        max_int = 1 << no_params - 1
        max_int == -1 &&
            throw(OverflowError("The number of possible models is too big to sample without replacement."))
        ints = sample(1:max_int, 500, replace = false)
        bits₀ = BitVector.(digits.(ints, base = 2, pad = no_params))
    elseif startup == :singlerandom
        bits₀ = (BitVector(randombits(no_params)),)
    elseif startup == :leaps
        supportsleaps(specs) ||
            throw(ErrorException(string("Unsupported model specification for the leaps-and-bounds algorithm: ",
                typeof(specs))))
        #TODO: re-use AIC calculations from leaps()
        R"library(leaps)"
        model_x, model_y = leaps_data(specs)
        @rput model_x
        @rput model_y
        R"startup_bits <- leaps(model_x, model_y, int = TRUE)$which"
        @rget startup_bits
        bits₀ = BitVector.(collect(eachrow(startup_bits)))
    else
        throw(ArgumentError(string("Unrecognized startup option: ", startup)))
    end
    return bits₀
end
