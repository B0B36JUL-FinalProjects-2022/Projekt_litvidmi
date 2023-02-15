using CSV, DataFrames, Plots, Random
const dir = joinpath(dirname(@__FILE__), "..", "movie_data")

export get_movielens_data!,
       read_movielens_to_df, 
       train_test_split,
       ui_matrix_to_movielens_form,
       restore_item_vector!

function get_movielens_data!(dir::String = dir)
    isdir(joinpath(dir, "ml-100k")) && return
    mkpath(dir)
    path = download("http://files.grouplens.org/datasets/movielens/ml-100k.zip")
    run(`unzip -x $path -d $dir`)
end


function read_movielens_to_df(filename::String="../../movie_data/ml1-100k/u.data")
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


function ui_matrix_to_movielens_form(X)
    m, n = size(X)
    X1 = [j for i in 1:n, j in 1:m]
    X2 = [j for i in 1:m, j in 1:n]'

    x1 = collect(Iterators.flatten(X1))
    x2 = collect(Iterators.flatten(X2))
    y = collect(Iterators.flatten(X'))
    return hcat(x1, x2), y
end

function restore_item_vector!(vec::Vector, X::Matrix, y::Vector, item_id::Integer)
    inds_in_sparse = findall(x -> x == 1, X[:, 2] .== item_id)
    inds_in_dense = X[inds_in_sparse, 1]
    for (ind_s, ind_d) in zip(inds_in_sparse, inds_in_dense)
        vec[ind_d] = y[ind_s]
    end
end
