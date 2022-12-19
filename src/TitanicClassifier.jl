module TitanicClassifier

using CSV
using DataFrames

# Write your package code here.
export title_frequencies, get_title_token, solve_SVM, computeKernel, LinearKernel,
compute_bias, classify_SVM, hyperparamCrossValidation, categorical_to_int!,
CSV_to_df, cabin_preprocessing!, replace_missing_with_median!, categorical_to_int,
replace_missing_with_most_common!, name_to_title!, title_to_title_token!, name_preprocessing!,
ticket_preprocessing!

include("svm.jl")
include("data_preparation.jl")

#=
extract_title(name) = String(last(split(split(name, ".")[1], ", ")))
remove_article(name) = Base.replace(Base.replace(name, "the" => ""), "The" => "")
trim_whitespace(name) = lstrip(rstrip(name))
get_normalized_title(name) = trim_whitespace(remove_article(extract_title(str))) =#

function get_normalized_title(name)
    title = String(last(split(split(name, ".")[1], ", ")))
    title_without_article = Base.replace(Base.replace(title, "the" => ""), "The" => "")
    normalized_title = lstrip(rstrip(title_without_article))
    return normalized_title
end

name_to_title!(titanic_df) = transform!(titanic_df, :Name => ByRow(get_normalized_title) => :Name)

function get_title_token(title)
    replacement_rules = [[["Dr", "Rev", "Col", "Major", "Capt"], "Officer"],
    [["Jonkheer", "Countess", "Sir", "Lady", "Don", "Dona"], "Royalty"], 
    [["Mlle"], "Miss"], [["Ms"], "Miss"], [["Mme"],"Mrs"]]

    for (title_group, token) in replacement_rules
        if title in title_group
            return token
        end
    end

    return title
end

title_to_title_token!(titanic_df) = transform!(titanic_df, :Name => ByRow(get_title_token) => :Name)

function name_preprocessing!(titanic_df)
    name_to_title!(titanic_df)
    title_to_title_token!(titanic_df)
end

function cabin_to_categorical(cabin)
    if !ismissing(cabin)
        return cabin[1]
    end
    return 'U'
end

cabin_preprocessing!(titanic_df) = transform!(titanic_df, :Cabin => ByRow(c -> cabin_to_categorical(c)) => :Cabin)

function extract_ticket_num(ticket) 
    temp = split(ticket, " ")
    if (length(temp) == 1 && temp[1]=="LINE")
        return -1
    else 
        return parse(Int64, last(temp))
    end
end

function get_ticket_mappig(titanic_df)
    tickets = Set()

    for row in eachrow(titanic_df)
        push!(tickets, extract_ticket_num(row["Ticket"]))
    end

    sorted_ticket_nums = sort(collect(tickets))

    ticket_idx_mapping = Dict{Integer, Integer}()
    for i in eachindex(sorted_ticket_nums)
        ticket_idx_mapping[sorted_ticket_nums[i]] = i
    end

    return ticket_idx_mapping
end

function ticket_preprocessing!(titanic_df)
    mapping = get_ticket_mappig(titanic_df)

    transform!(titanic_df, :Ticket => ByRow(t -> mapping[extract_ticket_num(t)]) => :Ticket)
end

end