module SimpleGGM
using Distributions, LinearAlgebra, StatsBase, StatsAPI, DataFrames

export ggm

struct GGM{M<:AbstractMatrix, D<:AbstractMvNormal} <: StatisticalModel
    x::M
    dist::D
    names::Vector{String}
end 

function ggm(df::AbstractDataFrame)
    x = hcat(eachcol(df)...)
    return fit(GGM, x, names(df))
end 

function StatsBase.fit(::Type{GGM},
                       x::AbstractMatrix,
                       names::AbstractVector{<:AbstractString})
    Κ = Symmetric(inv(cov(x)))
    # TODO: Assumes centered variables
    dist = MvNormalCanon(Κ)
    return GGM(x, dist, names)
end 

function StatsAPI.coefnames(model::GGM)
    it = Iterators.product(model.names, model.names)
    return [string(i, "--", j) for (j, i) in it if i != j]
end 

end # module
