module TitanicClassifier

using CSV
using DataFrames

# Write your package code here.
export title_frequencies, get_title_token, solve_SVM, computeKernel, LinearKernel,
compute_bias, classify_SVM, hyperparamCrossValidation, categorical_to_int!,
CSV_to_df, cabin_preprocessing!, replace_missing_with_median!, categorical_to_int

include("svm.jl")
include("data_preparation.jl")


extract_title(name) = String(last(split(split(name, ".")[1], ", ")))

function title_frequencies(df)
    freq = Dict()
    for row in eachrow(df)
        title = extract_title(row["Name"])
        freq[title] = get!(freq, title, 0) + 1
    end

    return freq
end

remove_article(name) = Base.replace(Base.replace(name, "the" => ""), "The" => "")
trim_whitespace(name) = lstrip(rstrip(name))

function tokenize_titles(str, rules)
    for r in rules
        for j in first(r)
            if str == j
                return last(r)
            end
        end
    end
    return str
end

function get_title_token(str, replace_rules)
    title = trim_whitespace(remove_article(extract_title(str)))
    token = tokenize_titles(title, replace_rules)
    return token
end

function cabin_to_categorical(cabin)
    if !ismissing(cabin)
        return cabin[1]
    end
    return 'U'
end

cabin_preprocessing!(df) = transform!(df, :Cabin => ByRow(c -> cabin_to_categorical(c)) => :Cabin)

end