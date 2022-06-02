struct OccamsWindowSolution{F<:AbstractFloat,S<:AbstractModelSpecs,A<:AbstractMarginalApproximation}
    modelset::WeightedModelSet{F}
    specs::S
    approximation::A
    hyperparams::OccamsWindowParams{F}
    iters::Tuple{Int, Int}
    coef_weights::Vector{F}
 end
 
function Base.show(io::IO, solution::OccamsWindowSolution)
    (; modelset, specs, approximation, iters, coef_weights) = solution
    println(io, "Occam's Window executed for $iters[1] + $iters[2] = $(sum(iters)) iterations, using the $approximation approximation to the marginal likelihood.")
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
