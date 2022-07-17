import Pkg
Pkg.activate(".")
using Distributed

@everywhere include("lm_include.jl")
@info "Distributed environments ready."

function simulate!(; no_sims)
    (length(ARGS) != 1 ||
     !(ARGS[1] ∈ string.(subtypes(SimType)))) &&
        throw(ArgumentError("Invalid argument(s): $ARGS"))

    model_type = @parse_type ARGS[1]
    vars = [5, 10, 20] # No. of variables
    obs_per_param = [10, 20, 100] # No. of observations **per variable**
    prop_vars = [4, 2, 1] # Proportion of variables that conform the true data-generating model.

    @showprogress @distributed for (sim_id, p, obs, prop_p) in
                                   collect(Iterators.product(1:no_sims, vars, obs_per_param, prop_vars))
        no_vars = div(p, prop_p)
        n = p * obs
        true_is, models, models_p, param_p, timeout =
            one_sim(sim_id, n, p, no_vars, model_type)
        save_sim!(sim_id, p, obs, prop_p, model_type, true_is, models, models_p, param_p, timeout)
    end
    return nothing
end

stats = @timed simulate!(no_sims = 20)
jldsave("output/$(ARGS[1]).jld2"; stats)
@info "Completed in $(stats.time)s"

