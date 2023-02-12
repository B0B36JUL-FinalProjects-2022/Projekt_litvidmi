using CSV, DataFrames, Plots, Random

export get_movielens_data!, read_movielens_to_df, train_test_split

function get_movielens_data!(dir)
    mkpath(dir)
    path = download("http://files.grouplens.org/datasets/movielens/ml-100k.zip")
    run(`unzip -x $path -d $dir`)
end


function read_movielens_to_df(filename)
    df = CSV.read(filename, DataFrame; header=false)
    rename!(df, [:user_id, :item_id, :rating, :timestamp])
    num_users = length(unique(df.user_id))
    num_items = length(unique(df.item_id))
    return df, num_users, num_items
end


function train_test_split(X, y::AbstractVector; dims=1, ratio_train=0.8, kwargs...)
    n = length(y)
    size(X, dims) == n || throw(DimensionMismatch("..."))

    n_train = round(Int, ratio_train*n)
    i_rand = randperm(n)
    i_train = i_rand[1:n_train]
    i_test = i_rand[n_train+1:end]

    return selectdim(X, dims, i_train), y[i_train], selectdim(X, dims, i_test), y[i_test]
end