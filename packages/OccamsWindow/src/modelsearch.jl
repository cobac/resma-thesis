
function model_search(saturated_model::StatisticalModel, marginal_approximation::AbstractMarginalApproximation; hyperparams::OccamsWindowParams = OccamsWindowParams())
    @debug "General model_search() called."
    saturated_bits = get_coef_bits(saturated_model)
    model_specs = get_model_specs(saturated_model)
    (; Oᵣ, Oₗ, startup) = hyperparams

    # 𝓐  from Madigan & Raftery (1994)
    accepted_models = emptyModelSet(ModelAndMarginal{get_model_type(saturated_model),
        typeof(marginal_approximation)})

    if startup == :saturated
        bits₀ = (fill(true, length(saturated_bits)),)
    elseif startup == :random
        no_bits = length(saturated_bits)
        max_int = 1 << no_bits - 1
        max_int == -1 && error("err... buffer overflow calculating the no. of possible models")
        ints = sample(1:max_int, 500, replace = false)
        bits₀ = BitVector.(digits.(ints, base = 2, pad = length(saturated_bits)))
    elseif startup == :singlerandom
        bits₀ = (randombits(length(saturated_bits)),)
    elseif startup == :leaps
        R"library(leaps)"
        model_x, has_intercept = predictors(model_specs)
        model_y = response(model_specs)
        @rput model_x
        @rput model_y
        @rput has_intercept
        R"startup_bits <- leaps(model_x, model_y, int = has_intercept)$which"
        @rget startup_bits
        bits₀ = BitVector.(collect(eachrow(startup_bits)))
        map!(bits₀, bits₀) do bits
            append!([true], bits)
        end
    else
        error("Unrecognized startup option: ", startup)
    end

    # 𝒞 from Madigan & Raftery (1994)
    candidate_models = ModelSet([(BitVector(bit₀),
        ModelAndMarginal(fit(model_specs, bit₀),
            marginal_approximation)) for bit₀ in bits₀])

    # Down pass
    down_iter = 0
    while !isempty(candidate_models)
        down_iter += 1
        @debug "Down pass iter: $down_iter"
        m = pop_rand!(candidate_models)
        accepted_models[first(m)] = last(m)
        sum(first(m)) == 1 && continue
        m₀s = sample_submodels(m, model_specs, marginal_approximation)
        for m₀ in m₀s
            B = last(m₀).value - last(m).value
            if B > Oᵣ
                delete!(accepted_models, first(m))
                if !haskey(candidate_models, first(m₀))
                    candidate_models[first(m₀)] = last(m₀)
                end
            elseif Oₗ <= B <= Oᵣ && !haskey(candidate_models, first(m₀))
                candidate_models[first(m₀)] = last(m₀)
            end
        end
        # @debug accepted_models
        # @debug candidate_models
    end # Down pass

    # Up pass
    up_iter = 0
    candidate_models = accepted_models
    accepted_models = emptyModelSet(ModelAndMarginal{get_model_type(saturated_model),
        typeof(marginal_approximation)})
    while !isempty(candidate_models)
        up_iter += 1
        @debug "Up pass iter: $up_iter"
        m = pop_rand!(candidate_models)
        accepted_models[first(m)] = last(m)
        m₁s = sample_supermodels(m, model_specs, marginal_approximation)
        for m₁ in m₁s
            B = last(m).value - last(m₁).value
            if B < Oₗ
                delete!(accepted_models, first(m))
                if !haskey(candidate_models, first(m₁))
                    candidate_models[first(m₁)] = last(m₁)
                end
            elseif Oₗ <= B <= Oᵣ && !haskey(candidate_models, first(m₁))
                candidate_models[first(m₁)] = last(m₁)
            end
        end
        @debug accepted_models
        @debug candidate_models
    end # Up pass

    length(accepted_models) == 0 &&
        error("The model search did not accept any model.")

    out_bits = collect(keys(accepted_models))
    out_models = [accepted_models[k] for k in out_bits]
    out_modelset = WeightedModelSet(out_bits, out_models)

    coef_weights = zeros(length(saturated_bits))
    for bit in eachindex(saturated_bits.bits)
        for model in eachindex(out_models)
            if out_bits[model][bit]
                coef_weights[bit] += out_modelset.weights[model]
            end
        end
    end

    return OccamsWindowSolution(out_modelset,
        saturated_model,
        marginal_approximation,
        hyperparams,
        down_iter,
        up_iter,
        coef_weights)
end
