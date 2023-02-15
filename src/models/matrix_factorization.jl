export MF, sgd_step!, run_train_eval_loop!

struct MF <: Recommender
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


function sgd_step!(loss::RecLoss, m::MF, data, α)
    x, y = data[1]
    grad = gradient(loss, m, x, y)[1]
    dLdP, dLdQ = grad.P.weight, grad.Q.weight

    @. m.P.weight -= α * dLdP
    @. m.Q.weight -= α * dLdQ
end

function run_train_eval_loop!(
    loss::RecLoss,
    model::MF,
    X_train, 
    y_train, 
    X_test=nothing, 
    y_test=nothing; 
    num_epochs = 10, 
    α = 1e-3, 
    verbose = true
)
    testmode!(model, true)
    train_error = zeros(num_epochs + 1)
    test_error = X_test !== nothing ? zeros(num_epochs + 1) : nothing
    train_error[1] = rmse_movielens(model, X_train, y_train)
    X_test !== nothing && (test_error[1] = rmse_movielens(model, X_test, y_test))

    for epoch in 1:num_epochs
        trainmode!(model, true)
        for i in eachindex(y_train)
            user_id, item_id = X_train[i, :]
            data = [((user_id, item_id), y_train[i])] 
            sgd_step!(loss, model, data, 1e-2)
        end
        testmode!(model, true)
        train_error[epoch + 1] = rmse_movielens(model, X_train, y_train)
        X_test !== nothing && ((test_error[epoch + 1] = rmse_movielens(model, X_test, y_test)))
        if verbose
            print("epoch :$epoch, train_loss: $(train_error[epoch + 1])")
            (X_test !== nothing) ? println("; test_loss: $(test_error[epoch + 1])") : println()
        end
    end

    return train_error, test_error
end