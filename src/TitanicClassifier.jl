module TitanicClassifier

using CSV
using DataFrames

export name_preprocessing, name_preprocessing!, cabin_preprocessing, cabin_preprocessing!,
    ticket_preprocessing, ticket_preprocessing!, titanic_preprocessing, ticket_preprocessing!,
    compute_kernel, LinearKernel, PolynomialKernel, RBFKernel, solve_SVM_dual, compute_bias,
    solve_SVM, classify_SVM, prepare_data_for_SVM, hyperparam_cross_validation, categorical_to_int,
    categorical_to_int!, replace_missing_with_median, replace_missing_with_most_common,
    replace_missing_with_linreg, standardize_data, CSV_to_df, random_data_split


include("svm.jl")
include("data_preparation.jl")

function get_normalized_title(name::AbstractString)
    title = String(last(split(split(name, ".")[1], ", ")))
    title_without_article = Base.replace(Base.replace(title, "the" => ""), "The" => "")
    normalized_title = lstrip(rstrip(title_without_article))
    return normalized_title
end

function name_to_title(titanic_df::DataFrame)
    titanic_cpy = copy(titanic_df)
    name_to_title!(titanic_cpy)
    return titanic_cpy
end

name_to_title!(titanic_df::DataFrame) = transform!(titanic_df, :Name => ByRow(get_normalized_title) => :Name)

function get_title_token(title::AbstractString)
    replacement_rules = [[["Dr", "Rev", "Col", "Major", "Capt"], "Officer"],
        [["Jonkheer", "Countess", "Sir", "Lady", "Don", "Dona"], "Royalty"],
        [["Mlle"], "Miss"], [["Ms"], "Miss"], [["Mme"], "Mrs"]]

    for (title_group, token) in replacement_rules
        if title in title_group
            return token
        end
    end

    return title
end

function title_to_title_token(titanic_df::DataFrame)
    titanic_cpy = copy(titanic_df)
    title_to_title_token!(titanic_cpy)
    return titanic_cpy
end

title_to_title_token!(titanic_df::DataFrame) = transform!(titanic_df, :Name => ByRow(get_title_token) => :Name)

function name_preprocessing(titanic_df::DataFrame)
    titanic_cpy = copy(titanic_df)
    name_preprocessing!(titanic_cpy)
    return titanic_cpy
end

function name_preprocessing!(titanic_df::DataFrame)
    name_to_title!(titanic_df)
    title_to_title_token!(titanic_df)
end

function cabin_to_categorical(cabin::Union{AbstractString,Missing})
    if !ismissing(cabin)
        return cabin[1]
    end
    return 'U'
end

function cabin_preprocessing(titanic_df::DataFrame)
    titanic_cpy = copy(titanic_df)
    cabin_preprocessing!(titanic_cpy)
    return titanic_cpy
end

cabin_preprocessing!(titanic_df::DataFrame) = transform!(titanic_df, :Cabin => ByRow(c -> cabin_to_categorical(c)) => :Cabin)

function extract_ticket_num(ticket::AbstractString)
    temp = split(ticket, " ")
    if (length(temp) == 1 && temp[1] == "LINE")
        return -1
    else
        return parse(Int64, last(temp))
    end
end

function get_ticket_mappig(titanic_df::DataFrame)
    tickets = Set()

    for row in eachrow(titanic_df)
        push!(tickets, extract_ticket_num(row["Ticket"]))
    end

    sorted_ticket_nums = sort(collect(tickets))

    ticket_idx_mapping = Dict{Integer,Integer}()
    for i in eachindex(sorted_ticket_nums)
        ticket_idx_mapping[sorted_ticket_nums[i]] = i
    end

    return ticket_idx_mapping
end

function ticket_preprocessing(titanic_df::DataFrame)
    titanic_cpy = copy(titanic_df)
    ticket_preprocessing!(titanic_cpy)
    return titanic_cpy
end

function ticket_preprocessing!(titanic_df::DataFrame)
    mapping = get_ticket_mappig(titanic_df)

    transform!(titanic_df, :Ticket => ByRow(t -> mapping[extract_ticket_num(t)]) => :Ticket)
end

"""
    titanic_preprocessing(titanic_df)

Accepts DataFrame with the data from the Titanic dataset which is included
in the `data` directory of the TitanicClassifier module, or one with the same structure.
Returns a copy of this DataFrame with no missing values, and with all features converted to numeric
values. For more information about preprocessing of individual features see `features.ipynb`
in the `examples` directory.  
"""
function titanic_preprocessing(titanic_df::DataFrame)
    titanic_cpy = copy(titanic_df)
    titanic_preprocessing!(titanic_cpy)
    return titanic_cpy
end

function titanic_preprocessing!(titanic_df::DataFrame)
    cabin_preprocessing!(titanic_df)
    name_preprocessing!(titanic_df)
    ticket_preprocessing!(titanic_df)
    replace_missing_with_median!(titanic_df, [:Fare])
    replace_missing_with_most_common!(titanic_df, [:Embarked])
    categorical_to_int!(titanic_df, [:Sex, :Name, :Embarked, :Cabin])
    replace_missing_with_linreg!(titanic_df, :Age, [:Pclass, :Name])
end

end