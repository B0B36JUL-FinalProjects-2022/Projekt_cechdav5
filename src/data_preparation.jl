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
        no_missing_vec = getproperty(dropmissing(select(df, col)), col)
        vec_int = categorical_to_int(no_missing_vec)
        med = median(vec_int)
    end
end

function CSV_to_df(path)
    if isfile(path)
        return CSV.read(path, DataFrame)
    end
    return nothing
end