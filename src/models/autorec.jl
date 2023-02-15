export AutoRec, 
       sgd_step!,
       run_train_eval_loop!

struct AutoRec <: Recommender
    num_users::Integer
    num_hidden::Integer
    encoder::Dense
    decoder::Dense
    dropout::Dropout
    function AutoRec(num_users::Integer, num_hidden::Integer; dropout_rate=0.05)
        encoder = Dense(num_users => num_hidden)
        decoder = Dense(num_hidden => num_users)
        dropout = Dropout(dropout_rate)
        return new(num_users, num_hidden, encoder, decoder, dropout)
    end
end

# input x is a column of R matrix: all users 
function (m::AutoRec)(x)
    hidden = m.encoder(x)
    hidden = m.dropout(sigmoid.(hidden))
    pred   = m.decoder(hidden)
    return pred
end

Flux.trainable(model::AutoRec) = (model.encoder, model.decoder)


function sgd_step!(loss::RecLoss, model::AutoRec, data, α; λ=0.1)
    x, y = data[1]
    grad = gradient(loss, model, x, y, λ)[1]
    dLdW, dLdV = grad.encoder.weight, grad.decoder.weight
    dldWb, dldVb = grad.encoder.bias, grad.decoder.bias

    @. model.encoder.weight -= α * dLdW
    @. model.decoder.weight -= α * dLdV
    @. model.encoder.bias   -= α * dldWb
    @. model.decoder.bias   -= α * dldVb
end;


function run_train_eval_loop!(
    loss::RecLoss,
    model::AutoRec,
    X_train,
    y_train,
    X_test     = nothing,
    y_test     = nothing;
    num_epochs = 10,
    α          = 1e-3,
    λ          = 0,
    verbose    = true
)
    train_error = zeros(num_epochs + 1)
    test_error = X_test !== nothing ? zeros(num_epochs + 1) : nothing
    
    testmode!(model, true)
    train_error[1] = rmse_movielens(model, X_train, y_train)
    (X_test !== nothing) && (test_error[1] = rmse_movielens(model, X_test, y_test))

    num_items = size(unique(X_train[:, 2]))[1]
    λ_normed = λ / num_items
    vec_to_learn = zeros(model.num_users)
    for epoch in 1:num_epochs
        trainmode!(model, true)
        for i in 1:num_items
            restore_item_vector!(vec_to_learn, X_train, y_train, i)
            x = y = vec_to_learn
            data = [(x, y)] 
            sgd_step!(loss, model, data, α; λ = λ_normed)
            vec_to_learn *= 0
        end
        testmode!(model, true)
        train_error[epoch + 1] = rmse_movielens(model, X_train, y_train)
        X_test !== nothing && (test_error[epoch + 1] = rmse_movielens(model, X_test, y_test))
        if verbose
            print("epoch :$epoch, train_loss: $(train_error[epoch + 1])")
            (X_test !== nothing) ? println("; test_loss: $(test_error[epoch + 1])") : println()
        end
    end
    return train_error, test_error
end

