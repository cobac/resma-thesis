struct GGMModelSpecs{M<:AbstractMatrix,S<:AbstractString} <: AbstractModelSpecs
    x::M
    names::Vector{S}
end

function model_specs(model::SimpleGGM.GGM)
    x_ggm = model.x
    R"options(warn = -1)"
    R"library(psychonetrics)"
    @rput x_ggm
    GGMModelSpecs(x_ggm, model.names)
end

"""
    get_indices(n, no_vars)

Generate the indices of the precision matrix represented by the `n`th bit out of `no_vars`.
"""
function get_indices(n, no_vars)
    no_param = binomial(no_vars, 2)
    n > no_param &&
        throw(ArgumentError("n is larger than the no. of parameters. n = $n, no_param = $no_param."))
    substracting = no_vars - 1
    leftover = n
    first_index = 1
    while (true)
        if leftover > substracting
            leftover -= substracting
        else
            break
        end
        substracting -= 1
        first_index += 1
    end
    second_index = first_index + leftover
    return (first_index, second_index)
end

function set_zeros!(target, indices)
    target[indices...] = 0
    target[reverse(indices)...] = 0
end


function marginal_likelihood(specs::GGMModelSpecs, bits::BitVector, approximation::BIC)
    target = ones(size(specs.x, 2), size(specs.x, 2))
    ommited_edges = findall(.!bits)
    for edge in ommited_edges
        pcors = get_indices(edge, size(specs.x, 2))
        set_zeros!(target, pcors)
    end

    @rput target

    R"model <- ggm(x_ggm, omega = target) |> runmodel()"
    R"bic <- model@fitmeasures$bic"
    @rget bic

    return -bic / 2
end

function marginal_likelihood(specs::GGMModelSpecs, bits::BitVector, approximation::FakeMarginal)
    R"""if (!exists("x_ggm")) stop("Data matrix not available in the R environment.")"""
    rand(append!(fill(0.2, 10), 0.8))
end


param_names(specs::GGMModelSpecs) = specs.names
