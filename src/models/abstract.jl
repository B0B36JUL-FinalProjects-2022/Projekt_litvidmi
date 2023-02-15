export rmse_movielens,
       right_predicted_movielens,
       predict_rating

abstract type Recommender end

abstract type RecLoss end

function rmse_movielens end

function right_predicted_movielens end

function predict_rating end

include("matrix_factorization.jl")
include("autorec.jl")
include("losses.jl")
include("evaluators.jl")
