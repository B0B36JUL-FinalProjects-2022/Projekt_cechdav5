using TitanicClassifier
using Test
using DataFrames

@testset "TitanicClassifier.jl" begin

    @testset "compute_kernel" begin
        Mi = [1 2; 3 4]
        Mj = [1 2; 2 3]
        @test compute_kernel(Mi, Mj, LinearKernel()) == [5 8; 11 18]
        @test compute_kernel(Mi, Mj, PolynomialKernel(2)) == [36 81; 144 361]
        @test round.(compute_kernel(Mi, Mj, RBFKernel(1)); digits=4) == [1 0.3679; 0.0183 0.3679]
        @test round.(compute_kernel(Mi, Mj, RBFKernel(20)); digits=4) == [1 0.9975; 0.99 0.9975]
    end

    @testset "titanic feature preprocessing" begin
        @test name_preprocessing(DataFrame(Name = [
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
            "Allen, Mr. William Henry",
            "McCarthy, Sir. Timothy J"
            ])) == DataFrame(Name = ["Mrs", "Miss", "Mr", "Royalty"])

        @test cabin_preprocessing(DataFrame(Cabin = [
            "C85", "E46", missing])) == DataFrame(Cabin = ['C', 'E', 'U'])

        @test ticket_preprocessing(DataFrame(Ticket = ["A/5 21171",
        "STON/O2. 3101282", "373450", "LINE"])) == DataFrame(Ticket = [2, 4, 3, 1])
    end

    @testset "SVM solution" begin
        X = [-1 1; 0 1; 1 1; 2 1; 3 1; 4 1]
        y = [1, 1, 1, -1, -1, -1]
        K = compute_kernel(X, X, LinearKernel())
        C = 10000

        z = solve_SVM_dual(K, y, C)
        @test z ≈ [0, 0, 2, 2, 0, 0] atol=1e-5
        @test compute_bias(K, y, z, C) ≈ 3 atol=1e-5

        model = solve_SVM(X, y, C)
        @test classify_SVM([-4 1; -3 1; 5 1; 6 1], model) == [1, 1, -1, -1]
    end

    @testset "prepare_data_for_SVM" begin
        X, y = prepare_data_for_SVM([-1; 0; 3; 4], [1,1,0,0])
        @test X ≈ [-1.05021 1.0; -0.63012 1.0; 0.63012 1.0; 1.05021 1.0] atol=1e-5
        @test y == [1, 1, -1, -1]
    end

    @testset "replace missing values" begin
        df = DataFrame(A = [1, 1, missing, 2, 3, 4], B = [0, 0, 6, 1, 2, 3],
        C = [2, 2, missing, 3, 4, 5], D = [2, 2, 4, 3, 4, 5])

        @test replace_missing_with_most_common(df, "A")[!, "A"] == [1, 1, 1, 2, 3, 4]
        @test replace_missing_with_median(df, "A")[!, "A"] == [1, 1, 2, 2, 3, 4]
        @test replace_missing_with_linreg(df, "A", "B")[!, "A"] ≈ [1, 1, 8.57142, 2, 3, 4] atol=1e-5

        @test replace_missing_with_most_common(df, ["A", "C"])[!, ["A",
        "C"]] == DataFrame(A = [1, 1, 1, 2, 3, 4], C = [2, 2, 2, 3, 4, 5])
        @test replace_missing_with_median(df, ["A", "C"])[!, ["A",
        "C"]] == DataFrame(A = [1, 1, 2, 2, 3, 4], C=[2, 2, 3, 3, 4, 5])
        @test replace_missing_with_linreg(df, "A", ["B", "D"])[!, "A"] ≈ [1, 1, 5, 2, 3, 4] atol=1e-5
    end

    @testset "categorical_to_int" begin
        df = DataFrame(A = ["Hello", "World", "Hello", "dlroW"])
        @test categorical_to_int(df[!, "A"]) == [1, 2, 1, 3]
    end

    @testset "standardize_data" begin
        X = standardize_data([-2 1; -1 2; 0 3; 1 4; 2 5; 3 6; 4 7; 5 8])
        @test standardize_data(X) ≈ [-1.428 -1.428; -1.020 -1.020; -0.612 -0.612;
         -0.204 -0.204; 0.204 0.204; 0.612 0.612; 1.020 1.020; 1.428 1.428] atol=1e-2
    end
end
