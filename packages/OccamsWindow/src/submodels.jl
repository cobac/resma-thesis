struct NamedBits{N}
    bits::BitArray{N}
    names::Array{String,N}
    # TODO: Implement size check
    # function NamedBits(bits, names)
    #     size(bits) != size(names) && error("Mismatch dimensions between bits and names.")
    #     return new{typeof(bits)}(bits, names)
    # end
end

Base.length(bs::NamedBits) = length(bs.bits)

Base.size(bs::NamedBits{N}) where {N} = N

function get_coef_bits(model::StatisticalModel)
    names = StatsAPI.coefnames(model)
    return NamedBits(BitVector(ones(Int, length(names))), names)
end

function get_coef_bits(saturated_bits::NamedBits, model::StatisticalModel)
    all_names = saturated_bits.names
    model_names = StatsAPI.coefnames(model)
    return all_names .∈ (model_names,)
end

function issubmodel(sub_bits::BitArray{N}, super_bits::BitArray{N}) where {N}
    sub_bits == super_bits && return false
    sub_params = foldl(+, sub_bits)
    super_params = foldl(+, super_bits)
    sub_params > super_params && return false
    sum_and = mapreduce(x -> x[1] && x[2], +, zip(sub_bits, super_bits))
    return sub_params == sum_and
end

function sample_submodels(model_el, model_specs, marginal_approximation)
    bits = first(model_el)
    trues = findall(bits)
    out = ModelSet(length(size(bits)), ModelAndMarginal{typeof(last(model_el).model),typeof(marginal_approximation)})
    for dropped_edge in trues
        submodel_bits = copy(bits)
        submodel_bits[dropped_edge] = 0
        m = fit(model_specs, submodel_bits)
        out[submodel_bits] = ModelAndMarginal(m, marginal_approximation)
    end
    return out
end

function sample_supermodels(model_el, model_specs, marginal_approximation)
    bits = first(model_el)
    falses = findall(.!bits)
    out = ModelSet(length(size(bits)),
        ModelAndMarginal{typeof(last(model_el).model),typeof(marginal_approximation)})
    for added_edge in falses
        supermodel_bits = copy(bits)
        supermodel_bits[added_edge] = 1
        m = fit(model_specs, supermodel_bits)
        out[supermodel_bits] = ModelAndMarginal(m, marginal_approximation)
    end
    return out
end 
function randombits(n)
    bits = rand(Bool, n)
    if sum(bits) == 0
        return randombits(n)
    end
    return bits
end
