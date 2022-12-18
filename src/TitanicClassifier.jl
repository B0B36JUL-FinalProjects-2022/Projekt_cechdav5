module TitanicClassifier

using CSV

# Write your package code here.
export load_data, title_frequencies, get_title_token, solve_SVM, computeKernel, LinearKernel, compute_bias

include("svm.jl")

function load_data()
    csv_reader = CSV.File("./data/train.csv")
    i = 0
    for row in csv_reader
        print(row.Survived, "|", row.Pclass, "|", row.Name, "|",  row.Sex, "|",
         row.Age, "|", row.SibSp, "|", row.Parch, "|", row.Ticket, "|", row.Fare, "|", row.Cabin, "|", row.Embarked, "\n")
        i+=1
        if i == 5
            break
        end
    end
end

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

end