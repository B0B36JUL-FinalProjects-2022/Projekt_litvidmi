using Flux
using LinearAlgebra

export AutoRec, AUTOREC_MSE_LOSS, opt_step!,
       run_train_eval_loop!, restore_item_vector!, mask_from_vector

struct AutoRec
    num_users::Integer
    num_hidden::Integer
    encoder
    decoder
    dropout
    function AutoRec(num_users::Integer, num_hidden::Integer; dropout_rate=0.05)
        encoder = Dense(num_users => num_hidden)
        decoder = Dense(num_hidden => num_users)
        dropout = Dropout(dropout_rate)
        return new(num_users::Integer, num_hidden::Integer, encoder, decoder, dropout::Dropout)
    end
end

# input x is a column of R matrix: all users 
function (m::AutoRec)(x)
    hidden = m.encoder(x)
    hidden = m.dropout(sigmoid.(hidden))
    pred   = m.decoder(hidden)
    return pred
end

struct AUTOREC_MSE_LOSS
end

(l::AUTOREC_MSE_LOSS)(m::AutoRec, x, y, λ = 0) = norm(y - m(x), 2) + λ * (norm(m.encoder.weight, 2) + norm(m.decoder.weight, 2))
(l::AUTOREC_MSE_LOSS)(m::AutoRec, x, y; λ = 0) = norm(y - m(x), 2) + λ * (norm(m.encoder.weight, 2) + norm(m.decoder.weight, 2))

function opt_step!(loss, model::AutoRec, data, α; λ=0.1)
    x, y = data[1]
    grad = gradient(loss, model, x, y, λ)[1]
    dLdW, dLdV = grad.encoder.weight, grad.decoder.weight

    @. model.encoder.weight -= α * dLdW
    @. model.decoder.weight -= α * dLdV
end;

# test it 
function restore_item_vector!(vec, X, y, item_id)
    inds_in_sparse = findall(x -> x == 1, X[:, 2] .== item_id)
    inds_in_dense = X[inds_in_sparse, 1]
    for (ind_s, ind_d) in zip(inds_in_sparse, inds_in_dense)
        vec[ind_d] = y[ind_s]
    end
end


function mask_from_vector(vec, X, item_id)
    vec = zeros(length(vec))
    inds_in_sparse = findall(x -> x == 1, X[:, 2] .== item_id)
    inds_in_dense = X[inds_in_sparse, 1]
    for ind_d in inds_in_dense
        vec[ind_d] = 1
    end
    return vec
end

function run_train_eval_loop!(loss, model::AutoRec, X_train, y_train,
                     X_test = nothing, y_test = nothing;
                     num_epochs = 10, α = 1e-3, λ = 0)
    train_error = zeros(num_epochs + 1)
    test_error = X_test !== nothing ? zeros(num_epochs + 1) : nothing
    
    testmode!(model, true)
    train_error[1] = rmse_movielens(model, X_train, y_train)
    if X_test !== nothing test_error[1] = rmse_movielens(model, X_test, y_test) end

    num_items = size(unique(X_train[:, 2]))[1]
    λ_normed = λ / num_items
    vec_to_learn = zeros(model.num_users)
    for epoch in 1:num_epochs
        trainmode!(model, true)
        for i in 1:num_items
            restore_item_vector!(vec_to_learn, X_train, y_train, i)
            x = y = vec_to_learn
            data = [(x, y)] 
            opt_step!(loss, model, data, α; λ = λ_normed)
            vec_to_learn *= 0
        end
        testmode!(model, true)
        train_error[epoch + 1] = rmse_movielens(model, X_train, y_train)
        if X_test !== nothing test_error[epoch + 1] = rmse_movielens(model, X_test, y_test) end
    end
    return train_error, test_error
end

# work on it a bit more
function rmse_movielens(model::AutoRec, X, y)
    cum_loss = 0 
    num_elems = 0
    num_items = length(unique(X[:, 2]))
    vec_to_restore = zeros(model.num_users)
    for i in 1:num_items
        restore_item_vector!(vec_to_restore, X, y, i)
        mask = mask_from_vector(vec_to_restore, X, i)
        masked_vec = mask .* (model(vec_to_restore) - vec_to_restore)
        cum_loss += sum(masked_vec.^2)
        num_elems += sum(mask)
        vec_to_restore *= 0
    end
    return sqrt(cum_loss / num_elems)
end

function rmse_movielens_int(model::AutoRec, X, y)
    cum_loss = 0 
    num_elems = 0
    num_items = length(unique(X[:, 2]))
    vec_to_restore = zeros(model.num_users)
    for i in 1:num_items
        restore_item_vector!(vec_to_restore, X, y, i)
        mask = mask_from_vector(vec_to_restore, X, i)
        masked_vec = mask .* (model(vec_to_restore) - vec_to_restore)
        cum_loss += sum(masked_vec.^2)
        num_elems += sum(mask)
        vec_to_restore *= 0
    end
    return sqrt(cum_loss / num_elems)
end

function right_predicted_movielens(model::AutoRec, X, y; precision = 0.5)
    cum_loss = 0 
    num_items = length(unique(X[:, 2]))
    vec_to_restore = zeros(model.num_users)
    for i in 1:num_items
        restore_item_vector!(vec_to_restore, X, y, i)
        cum_loss += sum(abs.(vec_to_restore - model(vec_to_restore)) .<= precision .* (vec_to_restore .!= 0))
        vec_to_restore *= 0
    end
    return cum_loss / (length(y))
end