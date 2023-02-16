export rmse_movielens,
       right_predicted_movielens,
       predict_rating

abstract type Recommender end

abstract type RecLoss end


"""
function run_train_eval_loop!(
    loss<:RecLoss,
    model<:Recommender,
    X_train::Matrix, 
    y_train::Vector;
    X_test::Union{Matrix, Nothing} = nothing, 
    y_test::Union{Matrix, Nothing} = nothing,
    num_epochs::Integer = 10, 
    α::Real = 1e-3, 
    λ::Real = 0,
    verbose::Bool = true,
    shuffle::Bool = true
) -> Vector, Vector
where
X_train, X_test have shape (n, 2)
y_train, y_test have shape (n, ) 
α - learning rate
λ - l2 regularization coefficient
"""
function run_train_eval_loop! end


"""
    rmse_movielens(model<:Recommender, X::Matrix, y::Vector) -> Real
Returns RMSE for the specific model and data in form of
matrix X (nx2) of user-item pairs and vector y (n,) of ratings.
"""
function sgd_step! end


"""
    rmse_movielens(model<:Recommender, X::Matrix, y::Vector) -> Real
Returns RMSE for the specific model and data in form of
matrix X (nx2) of user-item pairs and vector y (n,) of ratings.
"""
function rmse_movielens end


"""
    precision_movielens(model<:Recommender, X::Matrix, y::Vector; tolerance = 0.5) -> Real

Return the percent of movies for which
abs(rating prediction - real) doesn't surpass some tolerance value.
X (nx2)
y (n,)
"""
function precision_movielens end


"""
       predict_rating(user_id, item_id) -> Real
"""
function predict_rating end

include("matrix_factorization.jl")
include("autorec.jl")
include("losses.jl")
include("evaluators.jl")
