module SimpleGGM
using Distributions, LinearAlgebra, StatsBase, StatsAPI, DataFrames, PDMats
import StatsAPI: bic, fit, coefnames

export ggm, bic, coefnames, fit

# Epskamp et al. (2017). doi: 10.1007/s11336-017-9557-x

struct GGM{M<:AbstractMatrix,D<:AbstractMvNormal} <: StatisticalModel
    x::M
    dist::D # Precision matrix
    names::Vector{String}
end

function ggm(df::AbstractDataFrame, target::Union{Nothing,AbstractMatrix} = nothing)
    x = hcat(eachcol(df)...)
    if isnothing(target)
        target = ones(size(x, 2), size(x, 2))
    end
    return fit(GGM, x, target, expand_names(names(df)))
end

function StatsAPI.fit(::Type{GGM},
    x::AbstractMatrix,
    target::AbstractMatrix,
    names::AbstractVector{<:AbstractString})
    issymmetric(target) || throw(ArgumentError("Target matrix is not symmetric."))
    size(target) == (size(x, 2), size(x, 2)) ||
        throw(ArgumentError("Mismatched dimensions between covariance and target matrices."))
    S = inv(PDMat(cov(x)))
    Κ = S .* target
    dist = MvNormalCanon(Κ)
    return GGM(x, dist, names)
end

function generate_dist(Κ::AbstractMatrix, target::AbstractMatrix)
    MvNormalCanon(Κ.*target)

end

StatsAPI.bic(model::GGM) = length(model.names) * log(size(model.x, 1)) -
                           2 * loglikelihood(model.dist, model.x')

function expand_names(names::Vector{<:AbstractString})
    it = Iterators.product(names, names) |> collect
    is = Tuple.(vec(collect(CartesianIndices(size(it)))))
    return [string(it[i, j][1], "--", it[i, j][2]) for (j, i) in is if i < j]
end

StatsAPI.coefnames(model::GGM) = model.names

end
