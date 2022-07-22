include("lm_include.jl")
include("ggm_include.jl")

using CairoMakie

# Takes data points, saves plot
function generate_plots!(model_class::Type{SimType})
		data_dict = generate_data_points(model_class)
		rx = Regex("(\\d+)_(\\d+)_(\\d+)")

		vars = string.([5, 10, 20]) # No. of variables
		obs_per_param = string.([10, 20, 100]) # No. of observations **per variable**
		prop_vars = string.([4, 2, 1]) # Proportion of variables that conform the true data-generating model.


		f = Figure(resolution = (1000, 1200))

		g_p_max = f[1, 1:3]
		g_p_mid = f[2, 1:3]
		g_p_min = f[3, 1:3]
		p_grids = [g_p_max, g_p_mid, g_p_min]

		axis_args = (leftspinevisible = false,
				rightspinevisible = false,
				bottomspinevisible = false,
				topspinevisible = false,
				yticks = ([-1, 0, 1], ["1", "0", "1"]),
				yticksvisible = false)

		axis_p_max = [Axis(g_p_max[row, col]; ylabel = obs_per_param[row], axis_args...) for row in 1:3, col in 1:3]
		axis_p_mid = [Axis(g_p_mid[row, col]; ylabel = obs_per_param[row], axis_args...) for row in 1:3, col in 1:3]
		axis_p_min = [Axis(g_p_min[row, col]; ylabel = obs_per_param[row], axis_args...) for row in 1:3, col in 1:3]
		p_axis = [axis_p_max, axis_p_mid, axis_p_min]

		for (label, g) in zip(string.("Total no. of variables = ", vars), p_grids)
				Label(g[0, :], label)
		end

		colors = Makie.wong_colors()

		for key in keys(data_dict)
				p, n, prop = match(rx, key).captures
				p_i = findfirst(x -> x == p, vars)
				n_i = findfirst(x -> x == n, obs_per_param)
				prop_i = findfirst(x -> x == prop, prop_vars)

				sens = get_sens(data_dict[key])
				specs = -1 .* get_specs(data_dict[key])

				barplot!(p_axis[p_i][n_i, prop_i], 1:6, sens, fillto = specs, color = colors[1:6])
				hlines!(p_axis[p_i][n_i, prop_i], 0, color = :black)

				prop_i != 1 && hideydecorations!.(p_axis[p_i][n_i, prop_i], grid = false)
		end

		for a in p_axis
				linkyaxes!(a[1, 1], a[1, 2], a[1, 3])
				linkyaxes!(a[2, 1], a[2, 2], a[2, 3])
				linkyaxes!(a[3, 1], a[3, 2], a[3, 3])
				hidexdecorations!.(a, grid = true)
		end

		model_names = ["BAS",
				"BMA",
				"Occam's+\nLeaps",
				"Occam's+\nLeaps+\nNo window",
				"Occam's+\nSaturated",
				"Occam's+\nSaturated+\nNo window"]

		for block in 1:3
				Label(f[block, 0], "Observations per variable:", rotation = pi / 2, tellheight = false, tellwidth = true)
		end

		Label(f[4, 1], "1/4 of variables included")
		Label(f[4, 2], "1/2 of variables included")
		Label(f[4, 3], "All variables included")
		Legend(f[:, 4],
				[PolyElement(color = col) for col in colors[1:6]],
				model_names,
				tellheight = false,
				framevisible = false, orientation = :vertical)
		resize_to_layout!(f)
		save("figures/linear_results.svg", f)
		@info "Linear regression plot saved!"
end

function generate_plots!(model_class::Type{GGMSimType})
		data_dict = generate_data_points(model_class)
		rx = Regex("(\\d+)_(\\d+)_(\\d+)")


		ps = string.([5, 10]) # No. of variables
		ns = string.([500, 2000]) # Sample sizes
		sparses = string.([25, 75]) # Proportion of sparseness

		f = Figure(resolution = (1000, 1200))

		axis_args = (leftspinevisible = false,
				rightspinevisible = false,
				bottomspinevisible = false,
				topspinevisible = false,
				yticks = ([-1, 0, 1], ["1", "0", "1"]),
				yticksvisible = false)

		axis_p_max = [Axis(f[1, col]; axis_args...) for col in 1:4]
		axis_p_min = [Axis(f[2, col]; axis_args...) for col in 1:4]
		p_axis = [axis_p_max, axis_p_min]

		for (label, col) in zip(string.("Total no. of variables = ", ps), [1:2, 3:4])
				Label(f[0, col], label)
		end

		colors = Makie.wong_colors()

		for key in keys(data_dict)
				p, n, sparse = match(rx, key).captures
				p_i = findfirst(x -> x == p, ps)
				n_i = findfirst(x -> x == n, ns)
				sparse_i = findfirst(x -> x == sparse, sparses)

				sens = get_sens(data_dict[key])
				specs = -1 .* get_specs(data_dict[key])

				i = 0
				if p_i == 2
						i = 2
				end
				labels = nothing
				timed_out_keys = ["10_2000_75", "10_2000_25", "10_500_25", "10_500_75"]
				if key ∈ timed_out_keys
						labels = ["", "", "*"]
				end

				barplot!(p_axis[n_i][i + sparse_i], 1:3, sens, fillto = specs, color = colors[1:3], bar_labels = labels)
				hlines!(p_axis[n_i][i + sparse_i], 0, color = :black)

				sparse_i != 1 && hideydecorations!.(p_axis[n_i][i + sparse_i], grid = false)
		end

		for a in p_axis
				linkyaxes!(a[1], a[2], a[3], a[4])
				hidexdecorations!.(a, grid = true)
		end

		model_names = ["BDgraph", "BGGM", "Occam's"]

		Label(f[1, 0], "500 observations", rotation = pi / 2, tellheight = false, tellwidth = true)
		Label(f[2, 0], "2000 observations", rotation = pi / 2, tellheight = false, tellwidth = true)

		 Label(f[3, 1], "25% sparsity")
		 Label(f[3, 2], "75% sparsity")
		 Label(f[3, 3], "25% sparsity")
		 Label(f[3, 4], "75% sparsity")
		 Legend(f[:, 5],
				 [PolyElement(color = col) for col in colors[1:3]],
				 model_names,
				 tellheight = false,
				 framevisible = false, orientation = :vertical)
		resize_to_layout!(f)
		save("figures/ggm_results.svg", f)
		@info "GGM plot saved!"
end

get_sens(xs) = map(first, xs)
get_specs(xs) = map(last, xs)

# Generates data points per plot
function generate_data_points(model_class)
		model_types = subtypes(model_class)
		# data_dict = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
		data_dict = Dict()

		for model_type in model_types
				model_type = model_type()
				data = load_data(model_type)
				for key in keys(data)
						if !haskey(data_dict, key)
								data_dict[key] = []
						end
						data_dict[key] = push!(data_dict[key], sens_spec(data[key], model_type))
				end
		end
		return data_dict
end

# For a model type, loads its data as a dictionary
function load_data(model_type)
		model_name = string(typeof(model_type))
		dict = Dict{String,Vector{Tuple{Vector{Int},Vector{AbstractFloat},Bool}}}()
		filenames = readdir("output", join = true)
		rx = Regex("$(model_name)_(\\d+)_(\\d+)_(\\d+)_(\\d+)")
		for file in filenames
				m = match(rx, file)
				if !isnothing(m)
						p, obs, prop_p, _ = m.captures
						data = load(file)
						key = join([p, obs, prop_p], "_")
						get!(dict, key, [])
						dict[key] = push_dict!(dict, key, data, model_type)
				end
		end
		return dict
end

function push_dict!(dict, key, data, _)
		push!(dict[key], (data["true_is"],
				data["param_p"],
				data["timeout"]))
end

function push_dict!(dict, key, data, ::Union{BDgraph,BGGM})
		push!(dict[key], (data["true_is"],
				data["param_p"],
				false))
end


# For each vector of sims get values
function sens_spec(sims, model_type)
		true_pos = 0
		all_pos = 0
		true_neg = 0
		all_neg = 0
		for sim in sims
				_true_pos, _all_pos, _true_neg, _all_neg = _sens_spec(sim[1], sim[2], model_type)
				true_pos += _true_pos
				all_pos += _all_pos
				true_neg += _true_neg
				all_neg += _all_neg
		end

		sens = true_pos / all_pos
		spec = true_neg / all_neg

		isnan(sens) ? sens = 0 : nothing
		isnan(spec) ? spec = 0 : nothing
		# isinf(sens) ? sens = 1 : nothing
		isinf(spec) ? spec = 1 : nothing
		return sens, spec
end

function _sens_spec(true_is, probs, model_type; threshold = 0.5)
		true_pos = sum(probs[true_is] .> threshold)
		all_pos = sum(probs .> threshold)
		false_is = filter(!in(true_is), 1:length(probs))
		if !isempty(false_is)
				true_neg = sum(probs[false_is] .< threshold)
		else
				true_neg = 0
		end
		all_neg = sum(probs .< threshold)

		return true_pos, all_pos, true_neg, all_neg
end

# This is horrible design
_sens_spec(true_is, probs, model_type::BGGM) = _sens_spec(true_is, probs, nothing, threshold = 3)

occams_data = load_data(Occams())

for key in keys(occams_data)
		bool = Bool[]
		for sim in occams_data[key]
				push!(bool, sim |> last)
		end
		@show sum(bool) ∈ (0, 15)
end
# When there is a timeout, all simulations of that condition time-out

for key in (keys(occams_data))
		if occams_data[key] |> first |> last
				@show key
		end
end

# Time-outs:
# key = "10_2000_75"
# key = "10_2000_25"
# key = "10_500_25"
# key = "10_500_75"

generate_plots!(SimType)
generate_plots!(GGMSimType)
