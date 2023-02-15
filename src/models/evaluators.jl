export rmse_movielens,
       right_predicted_movielens


function rmse_movielens(model::AutoRec, X, y;)
    cum_loss = 0 
    num_elems = 0
    num_items = length(unique(X[:, 2]))
    vec_to_restore = zeros(model.num_users)
    for i in 1:num_items
        restore_item_vector!(vec_to_restore, X, y, i)
        mask = vec_to_restore .!= 0
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

function rmse_movielens(model::MF, X, y)
    cum_loss = 0 
    for i in eachindex(y)
        user_id, item_id = X[i, :]
        cum_loss += sum((model(user_id, item_id) - y[i])^2)
    end
    return sqrt(cum_loss / length(y))
end

function right_predicted_movielens(model::MF, X, y; precision = 0.5)
    cum_loss = 0 
    for i in eachindex(y)
        user_id, item_id = X[i, :]
        cum_loss += abs(model(user_id, item_id) - y[i]) <= precision
    end
    return cum_loss / length(y)
end

function prediction_distribution end