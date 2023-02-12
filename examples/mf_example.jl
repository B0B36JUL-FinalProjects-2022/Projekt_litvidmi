using Revise
using SysRec
using Plots


R = [3 1 1 3 1;
     1 2 4 1 3;
     3 1 1 3 1;
     4 3 5 3 3]


model = MF(3, 4, 5)
mse_loss = MF_MSE_LOSS()

# mse_loss(model, 1, 2, 1)
# opt_step!(mse_loss, model, [((1, 1), 1)], 0.1)
# using Flux
# gradient(mse_loss, model, (1, 1), 1)


model.P.weight * model.Q.weight'
RX, Ry = ui_matrix_to_movielens_form(R)
num_epochs = 200
lr = 1e-2
train_error = run_train_eval_loop!(mse_loss, model, RX, Ry; num_epochs = num_epochs, α = lr)

plot(collect(1:num_epochs), train_error, label="mse", xlabel="epoch", ylabel="train_error", linewidth = 2)

model.P.weight * model.Q.weight'
convert.(Int, round.(model.P.weight * model.Q.weight'))


plot(collect(1:num_epochs), train_error, label="train error", xlabel="epoch", ylabel="mse", linewidth = 2)




# download and unpack the dataset to the parent directory
const dir = joinpath(dirname(@__FILE__), "..", "movie_data")
get_movielens_data!(dir)

data_file = joinpath(dir, "ml-100k/u.data")
df, num_users, num_items = read_movielens_to_df(data_file)
sparsity = 1 - length(df.user_id) / (num_users * num_items)
first(df, 5)

plot(histogram(df.rating, bins=5), xlabel="rating", ylabel="count")


X, y = [df.user_id df.item_id], df.rating
X_train, y_train, X_test, y_test = train_test_split(X, y)

hidden_categories = 20
model = MF(hidden_categories, num_users, num_items)

num_epochs = 50
lr = 1e-2
train_error, test_error = run_train_eval_loop!(mse_loss, model, X_train, y_train, X_test, y_test; num_epochs = num_epochs, α = lr)



errors = hcat(train_error, test_error)
plot(
    collect(1:num_epochs), errors,
    label=["train error" "test_error"],
    xlabel="epoch", 
    ylabel="mse", 
    linewidth = [2 2]
    )


rmse_movielens(model, X_train, y_train)
rmse_movielens_int(model, X_train, y_train)
right_predicted_movielens(model, X_train, y_train)


rmse_movielens(model, X_test, y_test)
rmse_movielens_int(model, X_test, y_test)
right_predicted_movielens(model, X_test, y_test)