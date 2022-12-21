using TitanicClassifier
using Test
using Random
using DataFrames

@testset "TitanicClassifier.jl" begin
    Random.seed!(1)

    @test computeKernel([1, 2], [3, 4], LinearKernel()) == [3 4; 6 8]
    @test computeKernel([1 2; 3 4], [2 3; 4 5], LinearKernel()) == [8 14; 18 32]

    @test computeKernel([1, 2], [3, 4], LinearKernel()) .+ 1 == computeKernel([1, 2], [3, 4], PolynomialKernel(1))
    @test computeKernel([1, 2], [3, 4], PolynomialKernel(2)) == [16 25; 49 81]
    @test computeKernel([1 2; 3 4], [1 2; 2 3], PolynomialKernel(2)) == [36 81; 144 361]

    @test round.(computeKernel([1 2; 3 4], [1 2; 2 3], RBFKernel(1)); digits=4) == [1 0.3679; 0.0183 0.3679]
    @test round.(computeKernel([1 2; 3 4], [1 2; 2 3], RBFKernel(20)); digits=4) == [1 0.9975; 0.99 0.9975]


    @test name_preprocessing(DataFrame(Name = [
        "Cumings, Mrs. John Bradley",
        "Heikkinen, Miss. Laina",
        "Allen, Mr. William Henry",
        "McCarthy, Mr. Timothy J"
        ])) == DataFrame(Name = ["Mrs", "Miss", "Mr", "Mr"])

    # Write your tests here.

end
