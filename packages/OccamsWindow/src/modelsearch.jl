
Base.@kwdef struct OccamsWindowParams{F<:AbstractFloat}
    Oᵣ::F = 0.0
    Oₗ::F = log(20.0)
end

function model_search(saturated_model::StatisticalModel, marginal_approximation::AbstractMarginalApproximation, params::OccamsWindowParams = OccamsWindowParams())
    @debug "General model_search() called."
    saturated_bits = get_coef_bits(saturated_model)
    model_specs = get_model_specs(saturated_model)
    (; Oᵣ, Oₗ) = params

    # 𝓐  from Madigan & Raftery (1994)
    accepted_models = emptyModelSet(ModelAndMarginal{get_model_type(saturated_model),
        typeof(marginal_approximation)})

    bits₀ = randombits(length(saturated_bits))
    # 𝒞 from Madigan & Raftery (1994)
    candidate_models = ModelSet([(BitVector(bits₀),
        ModelAndMarginal(fit(model_specs, bits₀),
            marginal_approximation))])

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
        down_iter,
        up_iter,
        coef_weights)
end 
