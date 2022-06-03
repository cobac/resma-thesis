module SimpleGGM
using Distributions, LinearAlgebra, StatsBase, StatsAPI, DataFrames
import StatsAPI: fit, bic, coefnames

export ggm, fit, bic, coefnames

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
    return fit(GGM, x, target, names(df))
end

function StatsBase.fit(::Type{GGM},
    x::AbstractMatrix,
    target::AbstractMatrix,
    names::AbstractVector{<:AbstractString})
    issymmetric(target) || error("Target matrix is not symmetric.")
    size(target) == (size(x, 2), size(x, 2)) ||
        error("Mismatched dimensions between covariance and target matrices.")
    Κ = inv(Hermitian(cov(x, corrected = false))) .* target
    # TODO: Assumes centered variables
    dist = MvNormalCanon(Κ)
    return GGM(x, dist, names)
end

StatsAPI.bic(model::GGM, x::AbstractMatrix) = length(model.names) * log(size(model.x, 1)) -
                                              2 * loglikelihood(model.dist, x')

function StatsAPI.coefnames(model::GGM)
    it = Iterators.product(model.names, model.names)
    return [string(i, "--", j) for (j, i) in it if i != j]
end

end # module
