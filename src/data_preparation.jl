using Statistics

function CSV_to_df(path)
    if isfile(path)
        return CSV.read(path, DataFrame)
    else 
        throw(DomainError(path, "Such file doesn't exist"))
    end
end

function get_categorical_int_mapping(categorical::AbstractVector)
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

function categorical_to_int(categorical::AbstractVector)
    mapping = get_categorical_int_mapping(categorical)
    return map(x -> mapping[x], categorical)
end


categorical_to_int!(df::DataFrame, col) = categorical_to_int!(df, [col])

function categorical_to_int!(df::DataFrame, cols::AbstractVector)
    for col in cols
        transform!(df, col => categorical_to_int => col)
    end
end

function categorical_to_int(df::DataFrame, cols)
    cp_df = copy(df)
    categorical_to_int!(cp_df, cols)
    return cp_df
end

replace_missing_with_median!(df::DataFrame, col) = replace_missing_with_median!(df, [col])

function replace_missing_with_median!(df::DataFrame, cols::AbstractVector)
    for col in cols
        med = median(skipmissing(df[!, col]))
        replace!(df[!, col], missing => med);
    end
end

function replace_missing_with_median(df::DataFrame, cols)
    cp_df = copy(df)
    replace_missing_with_median!(cp_df, cols)
    return cp_df
end

replace_missing_with_most_common!(df::DataFrame, col) = replace_missing_with_most_common!(df, [col])

function replace_missing_with_most_common!(df::DataFrame, cols::AbstractArray)
    for col in cols
        frequencies = combine(groupby(dropmissing(select(df, col)), [col]), nrow)
        most_common = first(frequencies[!, col], 1)[1]
        replace!(df[!, col], missing => most_common);
    end
end

function replace_missing_with_most_common(df::DataFrame, cols)
    cp_df = copy(df)
    replace_missing_with_most_common!(cp_df, cols)
    return cp_df
end

function get_y(w::AbstractVector{<: Real}, dat)
    y, xs... = dat

    if ismissing(y)
        y = dot(w, reshape(collect(xs), (length(xs), 1)))
    end

    return y
end

replace_missing_with_linreg!(df::DataFrame, y_col, x_col) = replace_missing_with_linreg!(df, y_col, [x_col])

function replace_missing_with_linreg!(df::DataFrame, y_col, x_cols::AbstractArray)
    no_mising_df = dropmissing(df, y_col)

    X = Matrix{Float64}(no_mising_df[!, x_cols])
    y = Vector{Float64}(no_mising_df[!, y_col])

    w = (X'*X) \ (X'*y)
    
    combined = copy(x_cols)
    prepend!(combined, [y_col])

    transform!(df, combined => ByRow((row...) -> get_y(w, row)) => y_col)
end

function replace_missing_with_linreg(df::DataFrame, y_col, x_cols)
    cp_df = copy(df)
    replace_missing_with_linreg!(cp_df, y_col, x_cols)
    return cp_df
end

standardize_data(X::Vector{<: Real}) = standardize_data(reshape(X, (length(X), 1)))[:, 1]

function standardize_data(X::Matrix{<: Real})
    X = Matrix{Float64}(X)
    num_cols = size(X, 2)
    means = reshape(mean.(eachcol(X)), (1, num_cols))
    stds = reshape(std.(eachcol(X)), (1, num_cols))
    
    X .-= means
    X ./= stds;

    return X
end