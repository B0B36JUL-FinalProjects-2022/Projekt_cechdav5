using Statistics

function get_categorical_int_mapping(categorical)
    c = unique(categorical)
    mapping = Dict()

    i = 1
    for val in c
        if(!haskey(mapping, val))
            get!(mapping, val, i)
            i+=1
        end
    end

    return mapping
end

function categorical_to_int(categorical)
    mapping = get_categorical_int_mapping(categorical)
    return map(x -> mapping[x], categorical)
end

function categorical_to_int!(df, cols)
    for col in cols
        transform!(df, col => categorical_to_int => col)
    end
end

function replace_missing_with_median!(df, cols)
    for col in cols
        med = median(skipmissing(df[!, col]))
        print(med)
        replace!(df[!, col], missing => med);
    end
end

function replace_missing_with_most_common!(df, cols)
    for col in cols
        frequencies = combine(groupby(dropmissing(select(df, col)), [col]), nrow)
        most_common = first(frequencies[!, col], 1)[1]
        replace!(df[!, col], missing => most_common);
    end
end

function CSV_to_df(path)
    if isfile(path)
        return CSV.read(path, DataFrame)
    else 
        throw(DomainError(path, "Such file doesn't exist"))
    end
end