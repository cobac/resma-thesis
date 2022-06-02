# function get_coef_bits(model::StatisticalModel)
#     names = #     return NamedBits(BitVector(ones(Int, length(names))), names)
# end

# function get_coef_bits(saturated_bits::NamedBits, model::StatisticalModel)
#     all_names = saturated_bits.names
#     model_names = StatsAPI.coefnames(model)
#     return all_names .∈ (model_names,)
# end

function issubmodel(sub_bits::BitVector, super_bits::BitVector)
    sub_bits == super_bits && return false
    sub_params = foldl(+, sub_bits)
    super_params = foldl(+, super_bits)
    sub_params > super_params && return false
    sum_and = mapreduce(x -> x[1] && x[2], +, zip(sub_bits, super_bits))
    return sub_params == sum_and
end

function sample_submodels(bits)
    trues = findall(bits)
    out = ModelSet()
    for dropped_edge in trues
        submodel_bits = copy(bits)
        submodel_bits[dropped_edge] = 0
        push!(out, submodel_bits)
    end
    return out
end

function sample_supermodels(bits)
    falses = findall(.!bits)
    out = ModelSet()
    for added_edge in falses
        supermodel_bits = copy(bits)
        supermodel_bits[added_edge] = 1
        push!(out, supermodel_bits)
    end
    return out
end

function randombits(n::Int)
    bits = rand(Bool, n)
    if sum(bits) == 0
        return randombits(n)
    end
    return bits
end
