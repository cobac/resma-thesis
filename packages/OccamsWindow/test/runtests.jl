using OccamsWindow, Test
using DataFrames, GLM, Distributions, Random, StatsAPI
using OccamsWindow.BayesianLinearRegression

const N_TEST = 500
const p = 10 # no. params
const n = 500 # no. obs

Random.seed!(1)
x = rand(Float64, (n, p))
β = rand(Normal(0, 10), p + 1)
y = β[1] .+ x * β[2:end] .+ rand(Normal(), n)

df = DataFrame(x, [Symbol("x" * string(i)) for i in 1:p])
df[!, :y] = y

saturated_lm = lm(term(:y) ~ term(1) + foldl(+, [term(Symbol("x" * string(i)))
                                                 for i in 1:p]), df)

saturated_blm = blm(term(:y) ~ term(1) + foldl(+, [term(Symbol("x" * string(i)))
                                                   for i in 1:p]), df)

saturated_bits = BitVector(fill(true, p))

function generate_data(no_vars, x, β)
    is = sample(1:size(x, 2), no_vars, replace = false)
    y = β[1] .+ x[:, is] * β[is.+1] .+ rand(Normal(), n)
    return y, sort(is)
end


@testset "Marginal approximations" begin
    @test OccamsWindow.marginal_likelihood(saturated_lm, BIC()) ≈ -765.5577622644875
    @test_throws ErrorException OccamsWindow.marginal_likelihood(saturated_lm, OccamsWindow.TestLaplace())
end

@testset "Random bits" begin
    for i in 1:div(N_TEST, 20)
        n = rand(1:30)
        bits = OccamsWindow.randombits(n)
        @test length(bits) == n
    end
end

@testset "Modelsets" begin
    max_set_size = div(N_TEST, 20)
    m_set = OccamsWindow.ModelSet([OccamsWindow.randombits(p) for i in 1:max_set_size])
    set_size = length(m_set)
    @test set_size <= max_set_size
    @test all(length.(m_set) .== p)
    @test show(IOBuffer(), m_set) == nothing
    for i in 1:div(max_set_size, 2)
        rand_set = OccamsWindow.pop_rand!(m_set)
        @test length(m_set) == set_size - i
        @test (rand_set ∈ m_set) == false
    end
end

@testset "Submodels" begin
    sub_bits = BitVector([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    super_bits = saturated_bits
    @test OccamsWindow.issubmodel(sub_bits, sub_bits) == false

    @test OccamsWindow.issubmodel(sub_bits, super_bits) == true
    @test OccamsWindow.issubmodel(super_bits, sub_bits) == false

    bits_1 = BitVector([0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    @test OccamsWindow.issubmodel(bits_1, super_bits) == true
    @test OccamsWindow.issubmodel(super_bits, bits_1) == false
    @test OccamsWindow.issubmodel(bits_1, sub_bits) == false
    @test OccamsWindow.issubmodel(sub_bits, bits_1) == false

    bits_2 = BitVector([1, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    @test OccamsWindow.issubmodel(bits_2, super_bits) == true
    @test OccamsWindow.issubmodel(super_bits, bits_2) == false
    @test OccamsWindow.issubmodel(bits_2, sub_bits) == false
    @test OccamsWindow.issubmodel(sub_bits, bits_2) == true

    bits_3 = BitVector([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    @test OccamsWindow.issubmodel(bits_3, super_bits) == true
    @test OccamsWindow.issubmodel(super_bits, bits_3) == false
    @test OccamsWindow.issubmodel(bits_3, sub_bits) == false
    @test OccamsWindow.issubmodel(sub_bits, bits_3) == false
    @test OccamsWindow.issubmodel(bits_3, bits_2) == true
    @test OccamsWindow.issubmodel(bits_2, bits_3) == false
end


@testset "Marginal approximations" begin end

@testset "Caching" begin end

@testset "Solutions" begin end

@testset "Supported models" begin

    @testset "lm" begin

        specs_lm = OccamsWindow.model_specs(saturated_lm)
        @testset "Spec generation" begin
            @test length(OccamsWindow.param_names(specs_lm)) == p
            @test length(specs_lm.y) == n
            @test size(specs_lm.X, 1) == n
            @test size(specs_lm.X, 2) == p + 1

            for _ in 1:div(N_TEST, 5)
                bits_lm = OccamsWindow.randombits(p) |> BitVector
                model_lm = OccamsWindow.fit(specs_lm, bits_lm)
                # Intercept always included
                @test all(modelmatrix(model_lm)[:, 1] .== 1)
                @test !any(eachcol(modelmatrix(model_lm)[:, 2:end]) .== (ones(n),))
            end
        end

        @testset "Starting models" begin
            @test_throws ArgumentError OccamsWindow.starting_models(:not_defined, specs_lm)
            init_saturated_lm = OccamsWindow.starting_models(:saturated, specs_lm)[1]
            @test length(init_saturated_lm) == p
            @test all(init_saturated_lm .== 1)

            init_random_lm = OccamsWindow.starting_models(:random, specs_lm)
            @test length(init_random_lm) == 500
            @test all(length.(init_random_lm) .== p)

            init_leaps_lm = OccamsWindow.starting_models(:leaps, specs_lm)
            @test all(length.(init_leaps_lm) .== p)

        end

        @testset "Model search" begin
            true_pos = BitVector(undef, N_TEST)
            true_neg = BitVector(undef, N_TEST)
            for i in 1:N_TEST
                x = rand(Float64, (n, p))
                β = rand(Normal(0, 10), p + 1)
                y_lm, true_is = generate_data(rand(2:p), x, β)

                df = DataFrame(x, [Symbol("x" * string(i)) for i in 1:p])
                df[!, :y] = y_lm
                saturated_lm = lm(term(:y) ~ term(1) + foldl(+, [term(Symbol("x" * string(i)))
                                                                 for i in 1:p]), df)
                solution_lm = model_search(saturated_lm, BIC(),
                    hyperparams = OccamsWindowParams(startup = :leaps))

                true_pos[i] = all(solution_lm.coef_weights[true_is] .> 0.5)
                false_bits = trues(p)
                false_bits[true_is] .= false
                false_is = findall(false_bits)
                true_neg[i] = all(solution_lm.coef_weights[false_is] .< 0.5)
                @test true
            end
            @test mean(true_pos) > 0.9
            @test mean(true_neg) > 0.1
        end
    end #lm

    # @testset "blm" begin
    #     @testset "Spec generation" begin
    #         specs_blm = OccamsWindow.model_specs(saturated_blm)
    #         @test length(OccamsWindow.param_names(specs_blm)) == p
    #         @test length(specs_blm.y) == n
    #         @test size(specs_blm.X, 1) == n
    #         @test size(specs_blm.X, 2) == p + 1

    #         for _ in 1:N_TEST
    #             bits_blm = OccamsWindow.randombits(p) |> BitVector
    #             model_blm = OccamsWindow.fit(specs_blm, bits_blm)
    #             # Intercept always included
    #             @test all(modelmatrix(model_blm)[:, 1] .== 1)
    #             @test !any(eachcol(modelmatrix(model_blm)[:, 2:end]) .== (ones(n),))
    #         end
    #     end
    # end #blm
end
