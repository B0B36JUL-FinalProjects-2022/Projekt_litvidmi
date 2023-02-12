using Flux
using Statistics
using LinearAlgebra

export MF, MF_MSE_LOSS, opt_step!, run_train_eval_loop!,
       rmse_movielens, rmse_movielens_int,
       right_predicted_movielens, ui_matrix_to_movielens_form

struct MF
    num_factors::Integer
    num_users::Integer
    num_items::Integer
    P
    Q
    function MF(num_factors, num_users, num_items)
        P = Dense(num_factors => num_users)
        Q = Dense(num_factors => num_items)
        return new(num_factors, num_users, num_items, P, Q)
    end
end


function (m::MF)(user_id, item_id)
    return m.P.weight[user_id, :]' * m.Q.weight[item_id, :]
end

(m::MF)((user_id, item_id)) = m(user_id, item_id)
(m::MF)(x::Vector) = m(x[1], x[2])
(m::MF)(x1::Vector, x2::Vector) = m.(x1, x2)
(m::MF)(X::Matrix) = m(X[:, 1], X[:, 2])

Flux.@functor MF

struct MF_MSE_LOSS
end

# function (l::MF_RMSE_LOSS)(model::MF, user_id, item_id, y::Real)
#     abs2(model(user_id, item_id) - y)
# end


(l::MF_MSE_LOSS)(model::MF, user_id::Int, item_id::Int, R::Matrix{<:Real}) = abs2(model(user_id, item_id) .- R[user_id, item_id])
(l::MF_MSE_LOSS)(model::MF, user_id, item_id, y::Real) = abs2(model(user_id, item_id) - y)
(l::MF_MSE_LOSS)(model::MF, (user_id, item_id), y::Real) = l(model, user_id, item_id, y)
(l::MF_MSE_LOSS)(model::MF, x::Vector, y::Vector) = mean(abs2.(model.(x) .- y))
(l::MF_MSE_LOSS)(model::MF, x::Matrix, y::Vector) = mean(abs2.(model(x) .- y))
(l::MF_MSE_LOSS)(model::MF, x::Vector, y::Real) = (abs2(model(x[1], x[2]) - y))

# Flux.@functor MF_RMSE_LOSS

function opt_step!(loss, m::MF, data, α)
    x, y = data[1]
    grad = gradient(loss, m, x, y)[1]
    dLdP, dLdQ = grad.P.weight, grad.Q.weight

    @. m.P.weight -= α * dLdP
    @. m.Q.weight -= α * dLdQ
end

function run_train_eval_loop!(loss, model::MF, X_train, y_train, X_test=nothing, y_test=nothing; num_epochs = 100, α = 1e-3)
    train_error = zeros(num_epochs)
    test_error = X_test !== nothing ? zeros(num_epochs) : nothing
    for epoch in 1:num_epochs
        for i in eachindex(y_train)
            user_id, item_id = X_train[i, :]
            data = [((user_id, item_id), y_train[i])] 
            opt_step!(loss, model, data, 1e-2)
        end
        train_error[epoch] = rmse_movielens(model, X_train, y_train)
        if X_test !== nothing test_error[epoch] = rmse_movielens(model, X_test, y_test) end
    end

    return train_error, test_error
end

function rmse_movielens(model::MF, X, y)
    cum_loss = 0 
    for i in eachindex(y)
        user_id, item_id = X[i, :]
        cum_loss += norm(model(user_id, item_id) - y[i], 2)
    end
    return cum_loss / length(y)
end

function rmse_movielens_int(model::MF, X, y)
    cum_loss = 0 
    for i in eachindex(y)
        user_id, item_id = X[i, :]
        cum_loss += norm(convert(Int, round(model(user_id, item_id))) - y[i], 2)
    end
    return cum_loss / length(y)
end

function right_predicted_movielens(model::MF, X, y; precision = 0.5)
    cum_loss = 0 
    for i in eachindex(y)
        user_id, item_id = X[i, :]
        cum_loss += abs(model(user_id, item_id) - y[i]) <= precision
    end
    return cum_loss / length(y)
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