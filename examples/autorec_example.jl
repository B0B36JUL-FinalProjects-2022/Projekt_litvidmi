using Revise
using SysRec
using Plots


R = [3 1 1 3 1;
     1 2 4 1 3;
     3 1 1 3 1;
     4 3 5 3 3]


model = AutoRec(4, 3; dropout_rate = 0)
mse_loss = AUTOREC_MSE_LOSS()

RX, Ry = ui_matrix_to_movielens_form(R)
num_epochs = 50000
lr = 1e-3


# x = RX[1, :]
# y = Ry[1]
# data = [(x, y)] 
x = R[:, 1]
data = [(x, x)]

model(x)

opt_step!(mse_loss, model, data, 0.01; λ = 0.001)
gradient(mse_loss, model, R[:, 1], R[:, 1], 0.1)[1]
mse_loss(model, R[:, 1], R[:, 1])

train_error = run_train_eval_loop!(mse_loss, model, RX, Ry; num_epochs = num_epochs, α = lr, λ = 1)

plot(collect(1:num_epochs), train_error, label="mse", xlabel="epoch", ylabel="train_error", linewidth = 2)

model.P.weight * model.Q.weight'
convert.(Int, round.(model.P.weight * model.Q.weight'))


plot(collect(1:num_epochs), train_error, label="train error", xlabel="epoch", ylabel="mse", linewidth = 2)





const dir = joinpath(dirname(@__FILE__), "..", "movie_data")
# get_movielens_data!(dir)

data_file = joinpath(dir, "ml-100k/u.data")
df, num_users, num_items = read_movielens_to_df(data_file)
sparsity = 1 - length(df.user_id) / (num_users * num_items)
first(df, 5)

plot(histogram(df.rating, bins=5), xlabel="rating", ylabel="count")


X, y = [df.user_id df.item_id], df.rating
X_train, y_train, X_test, y_test = train_test_split(X, y)
# restore_item_vector!(zeros(1000), X_train, y_train, 1)

hidden_categories = 500
model = AutoRec(num_users, hidden_categories, dropout_rate = 0.0)
mse_loss = AUTOREC_MSE_LOSS()
num_epochs = 30
lr = 1e-1


train_error, test_error = run_train_eval_loop!(mse_loss, model, X_train, y_train, X_test, y_test; num_epochs = num_epochs, α = lr, λ = 0.0)

errors = hcat(train_error, test_error)
plot(
    collect(1:num_epochs + 1), errors,
    label=["train error" "test_error"],
    xlabel="epoch", 
    ylabel="rmse", 
    linewidth = [2 2]
    )

rmse_movielens(model, X_train, y_train)
rmse_movielens_int(model, X_train, y_train)
right_predicted_movielens(model, X_train, y_train)


rmse_movielens(model, X_test, y_test)
rmse_movielens_int(model, X_test, y_test)
right_predicted_movielens(model, X_test, y_test)


using Flux
using LinearAlgebra
m = Dense(4 => 4)
# @. m.weight = Matrix(I, 4, 4) * 1.0
# m(R[:, 1])
@. m.weight *= 0
m.weight[1, 1] = m.weight[2, 2] = m.weight[3, 3] = m.weight[4, 4] = 1
@. m.bias = [-1, 0, 0, 1]

# function rmse_movielens(model::Dense, X, y)
#     cum_loss = 0 
#     num_elems = 0
#     num_items = length(unique(X[:, 2]))
#     vec_to_restore = zeros(4)
#     for i in 1:num_items
#         restore_item_vector!(vec_to_restore, X, y, i)
#         mask = mask_from_vector(vec_to_restore, X, i)
#         println(mask)
#         masked_vec = mask .* (model(vec_to_restore) - vec_to_restore)
#         println(masked_vec)
#         cum_loss += sum(masked_vec.^2)
#         num_elems += sum(mask)
#         vec_to_restore *= 0
#     end
#     return sqrt(cum_loss / num_elems)
# end
#test it and restore and mask 


R1 = copy(R)
R1[1, 1] = R[2, 3] = R[3, 2] = R[4, 3] = 0
R1X, R1y = ui_matrix_to_movielens_form(R1)
rmse_movielens(m, RX, Ry)
norm(m(R) - R, 2) / sqrt(prod(size(R)))

norm(R - R, 2)