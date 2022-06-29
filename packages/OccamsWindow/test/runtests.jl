using OccamsWindow, Test
using DataFrames, GLM, Distributions, Random, StatsAPI
using OccamsWindow.BayesianLinearRegression

const N_TEST = 1000
const p = 10 # no. params
const n = 500 # no. obs

Random.seed!(1)
x = rand(Float64, (n, p))
X = [ones(n) x]
β = rand(Normal(0, 10), p + 1)
y = X * β .+ rand(Normal(), n)

df = DataFrame(x, :auto)
df[!, :y] = y

saturated_lm = lm(term(:y) ~ term(1) + foldl(+, [term(Symbol("x" * string(i)))
                                                 for i in 1:p]), df)

saturated_blm = blm(term(:y) ~ term(1) + foldl(+, [term(Symbol("x" * string(i)))
                                                   for i in 1:p]), df)

saturated_bits = BitVector(fill(true, p))

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

@testset "Supported models" begin

    @testset "unsupported" begin end

    @testset "lm" begin
        specs_lm = OccamsWindow.model_specs(saturated_lm)
        @test length(OccamsWindow.param_names(specs_lm)) == p
        @test length(specs_lm.y) == n
        @test size(specs_lm.X, 1) == n
        @test size(specs_lm.X, 2) == p + 1

        for _ in 1:N_TEST
            bits_lm = OccamsWindow.randombits(p) |> BitVector
            model_lm = OccamsWindow.fit(specs_lm, bits_lm)
            # Intercept always included
            @test all(modelmatrix(model_lm)[:, 1] .== 1)
            @test !any(eachcol(modelmatrix(model_lm)[:, 2:end]) .== (ones(n),))
        end
    end #lm

    @testset "blm" begin
        specs_blm = OccamsWindow.model_specs(saturated_blm)
        @test length(OccamsWindow.param_names(specs_blm)) == p
        @test length(specs_blm.y) == n
        @test size(specs_blm.X, 1) == n
        @test size(specs_blm.X, 2) == p + 1

        for _ in 1:N_TEST
            bits_blm = OccamsWindow.randombits(p) |> BitVector
            model_blm = OccamsWindow.fit(specs_blm, bits_blm)
            # Intercept always included
            @test all(modelmatrix(model_blm)[:, 1] .== 1)
            @test !any(eachcol(modelmatrix(model_blm)[:, 2:end]) .== (ones(n),))
        end
    end #blm
end
@testset "Model search base" begin
    for i in 1:N_TEST
        model_search(saturated_lm, BIC())
        @test true == true
    end
end 
