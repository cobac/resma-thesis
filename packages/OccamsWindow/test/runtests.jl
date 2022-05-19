using OccamsWindow, Test
using DataFrames, GLM, Distributions, Random, StatsAPI

const N_TEST = 1000
p = 10 # no. params
n = 500 # no. obs

Random.seed!(1)
x = rand(Float64, (n, p))
X = [ones(n) x]
β = rand(Normal(0, 10), p + 1)
y = X * β .+ rand(Normal(), n)

df = DataFrame(x, :auto)
df[!, :y] = y

saturated_model = lm(term(:y) ~ term(1) + foldl(+, [term(Symbol("x" * string(i)))
                                                    for i in 1:p]), df)

saturated_bits = OccamsWindow.get_coef_bits(saturated_model)

@testset "Marginal approximations" begin
    @test OccamsWindow.marginal_likelihood(saturated_model, BIC()) ≈ -765.5577622644875
    @test_throws ErrorException OccamsWindow.marginal_likelihood(saturated_model, OccamsWindow.TestLaplace())
end

@testset "Modelsets" begin
    # struct DummyModel <: StatsAPI.StatisticalModel end
    # OccamsWindow.WeightedModelSet(fill(DummyModel(), 5), fill(0.2, 5))
end

@testset "Submodels" begin
    model = lm(@formula(y ~ x1 + x2), df)

    sub_bits = OccamsWindow.get_coef_bits(saturated_bits, model)
    super_bits = OccamsWindow.get_coef_bits(saturated_bits, saturated_model)

    @test OccamsWindow.issubmodel(sub_bits, sub_bits) == false

    @test OccamsWindow.issubmodel(sub_bits, super_bits) == true
    @test OccamsWindow.issubmodel(super_bits, sub_bits) == false

    bits_1 = BitVector([0, 1, 0, 0, 0, 1, 0, 0, 0, 1,])
    @test OccamsWindow.issubmodel(bits_1, super_bits) == true
    @test OccamsWindow.issubmodel(super_bits, bits_1) == false
    @test OccamsWindow.issubmodel(bits_1, sub_bits) == false
    @test OccamsWindow.issubmodel(sub_bits, bits_1) == false

    bits_2 = BitVector([1, 1, 0, 0, 0, 0, 0, 1, 0, 0,])
    @test OccamsWindow.issubmodel(bits_2, super_bits) == true
    @test OccamsWindow.issubmodel(super_bits, bits_2) == false
    @test OccamsWindow.issubmodel(bits_2, sub_bits) == false
    @test OccamsWindow.issubmodel(sub_bits, bits_2) == false

    bits_3 = BitVector([0, 0, 0, 0, 0, 0, 0, 1, 0, 0,])
    @test OccamsWindow.issubmodel(bits_3, super_bits) == true
    @test OccamsWindow.issubmodel(super_bits, bits_3) == false
    @test OccamsWindow.issubmodel(bits_3, sub_bits) == false
    @test OccamsWindow.issubmodel(sub_bits, bits_3) == false
    @test OccamsWindow.issubmodel(bits_3, bits_2) == true
    @test OccamsWindow.issubmodel(bits_2, bits_3) == false
end

@testset "Model search base" begin
    for i in 1:N_TEST
        model_search(saturated_model, BIC())
        @test true == true
    end
end 
