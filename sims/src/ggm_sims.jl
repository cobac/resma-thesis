import Pkg
Pkg.activate(".")
using Distributed

@everywhere include("ggm_include.jl")
@info "Distributed environments ready."

"""
    simulate!(; no_sims, run_all = true)

Runs the simulations.
`no_sims` is the number of simulations per condition.
`run_all = true` forces to run all the conditions. 
When `run_all = false` the program does not run conditions that have already produced an output file.
"""
function simulate!(; no_sims, run_all = true)
    (length(ARGS) != 1 ||
     !(ARGS[1] ∈ string.(subtypes(GGMSimType)))) &&
        throw(ArgumentError("Invalid argument(s): $ARGS"))

    model_type = @parse_type ARGS[1]
    ps = [5, 10] # No. of variables
    ns = [500, 2000] # Sample sizes
    sparses = [0.25, 0.75] # Proportion of sparseness

    conditions = Set(Iterators.product(1:no_sims, ps, ns, sparses))

    if !run_all
        filenames = readdir("output", join = true)
        model_name = string(typeof(model_type))
        rx = Regex("$(model_name)_(\\d+)_(\\d+)_(\\d+)_(\\d+)")
        for file in filenames
            m = match(rx, file)
            if !isnothing(m)
                p, n, sparse, sim_id = parse.(Int, m.captures)
                iszero(sparse) ? sparse = 1 : sparse /= 100
                condition = (sim_id, p, n, sparse)
                if condition ∈ conditions
                    conditions = delete!(conditions, condition)
                end
            end
        end
    end

    @showprogress @distributed for (sim_id, p, n, sparse) in collect(conditions)
        true_is, models, models_p, param_p, timeout =
            one_sim(sim_id, n, p, sparse, model_type)
        save_sim!(model_type, sim_id, p, n, sparse, true_is, models, models_p, param_p, timeout)
    end
    return nothing
end

stats = @timed simulate!(no_sims = 15, run_all = false)
jldsave("output/$(ARGS[1]).jld2"; stats)
@info "Completed in $(stats.time)s"
