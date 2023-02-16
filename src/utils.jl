using CSV, DataFrames, Plots, Random
const dir = joinpath(dirname(@__FILE__), "..", "movie_data")

export get_movielens_data!,
       read_movielens_to_df, 
       train_test_split,
       ui_matrix_to_movielens_form,
       restore_item_vector!,
       restore_item_vector,
       split_into_batches


"""
    get_movielens_data!(dir::String = dir)
If movielens-100k dataset is not present in the given directory,
this function downloads and unpacks it.
   """
function get_movielens_data!(dir::String = dir)
    isdir(joinpath(dir, "ml-100k")) && return
    mkpath(dir)
    path = download("http://files.grouplens.org/datasets/movielens/ml-100k.zip")
    run(`unzip -x $path -d $dir`)
end


function read_movielens_to_df(
    filename::String = joinpath(dir,"../../movie_data/ml1-100k/u.data")
)
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

"""
    ui_matrix_to_movielens_form(R::Matrix) -> Matrix, Vector

R - a matrix of orbitrary size of rating for user-item pairs
X - Matrix of user-item pairs (with non-zero entries in R)
y - vector of non-zero ratings for each pair from y
"""
function ui_matrix_to_movielens_form(R::Matrix)
    m, n = size(R)
    X1 = [j for i in 1:n, j in 1:m]
    X2 = [j for i in 1:m, j in 1:n]'

    x1 = collect(Iterators.flatten(X1))
    x2 = collect(Iterators.flatten(X2))
    X = hcat(x1, x2)
    y = collect(Iterators.flatten(R'))
    
    mask = (y .!= 0)
    y = y[mask]
    X = X[mask, :]

    return X, y
end


"""
    restore_item_vector!(vec::Vector, X::Matrix, y::Vector, item_id::Integer) -> Vector

vec (number of users, )  - vector of zeros (length one item (column) from user-item matrix)
X   (n, 2)               - user-item pairs
y   (y, )                - (non-zero) rating for each pair

The function fills input vector <vec> with ratings each user (index in that vector) gave to
the item <item_id> creating representation of the item (column) in user-item matrix.
For a detailed exapmle see <../tests> directory.
"""
function restore_item_vector!(vec::Vector, X::Matrix, y::Vector, item_id)
    inds_in_sparse = findall(x -> x == 1, X[:, 2] .== item_id)
    inds_in_dense = X[inds_in_sparse, 1]
    for (ind_s, ind_d) in zip(inds_in_sparse, inds_in_dense)
        vec[ind_d] = y[ind_s]
    end
end


function restore_item_vector!(mat, X, y, item_ids)
    for i in eachindex(item_ids)
        inds_in_sparse = findall(x -> x == 1, X[:, 2] .== item_ids[i])
        inds_in_dense = X[inds_in_sparse, 1]
        for (ind_s, ind_d) in zip(inds_in_sparse, inds_in_dense)
            mat[ind_d, i] = y[ind_s]
        end
    end
end


function restore_item_vector(vec::Vector, X::Matrix, y::Vector, item_id::Integer)
    vec = zeros(length(vec))
    inds_in_sparse = findall(x -> x == 1, X[:, 2] .== item_id)
    inds_in_dense = X[inds_in_sparse, 1]
    for (ind_s, ind_d) in zip(inds_in_sparse, inds_in_dense)
        vec[ind_d] = y[ind_s]
    end

    return vec
end

"""
    split_into_batches(n::Integer, batch_size::Integer; shuffle::Bool = false)

returns a vector of vectors with indices of elements to be put into each batch.
Last batch may contain less indices.
"""
function split_into_batches(n::Integer, batch_size::Integer; shuffle::Bool = false)
    indexes = collect(1:n)
    if !shuffle
        batch_indexes = [indexes[collect(((i-1) * batch_size + 1):(i * batch_size))] for i in 1:(n ÷ batch_size)] 
    else
        shuffle!(indexes)
        batch_indexes = [indexes[collect(((i-1) * batch_size + 1):(i * batch_size))] for i in 1:(n ÷ batch_size)] 
    end
    n_remaining_indexes = n % batch_size
    (n_remaining_indexes != 0) && push!(batch_indexes, indexes[(n - n_remaining_indexes + 1):end])

    return batch_indexes
end