using Revise
using SysRec
using Flux: gradient
using Plots

R = [3 1 1 3 1;
     1 2 4 1 3;
     3 1 1 3 1;
     4 3 5 3 3]

model = MF(3, 4, 5)
mse_loss = MF_MSE_LOSS()


# look what internal functions of training loop return
#----------------------------
mse_loss(model, 1, 2, 1)
gradient(mse_loss, model, (1, 1), 1)
opt_step!(mse_loss, model, [((1, 1), 1)], 1e-5)
model(1, 1)
#----------------------------

model.P.weight * model.Q.weight'
RX, Ry = ui_matrix_to_movielens_form(R)
num_epochs = 200
lr = 1e-2
train_error = run_train_eval_loop!(mse_loss, model, RX, Ry; num_epochs = num_epochs, α = lr)

plot(collect(1:num_epochs + 1), train_error, label="mse", xlabel="epoch", ylabel="train_error", linewidth = 2)

model.P.weight * model.Q.weight'
convert.(Int, round.(model.P.weight * model.Q.weight'))


# example on movielens
# load and preprocess data
#----------------------------
# download and unpack the dataset to the parent directory if needed
const dir = joinpath(dirname(@__FILE__), "..", "movie_data")
get_movielens_data!()

data_file = joinpath(dir, "ml-100k/u.data")
df, num_users, num_items = read_movielens_to_df(data_file)
sparsity = 1 - length(df.user_id) / (num_users * num_items)
first(df, 5)

plot(histogram(df.rating, bins=5), xlabel="rating", ylabel="count")

X, y = [df.user_id df.item_id], df.rating
X_train, y_train, X_test, y_test = train_test_split(X, y)
#----------------------------

# create, train and evaluate a model
#----------------------------
hidden_categories = 25
model = MF(hidden_categories, num_users, num_items)
mse_loss = MF_MSE_LOSS()
num_epochs = 10
lr = 1e-2

train_error, test_error = run_train_eval_loop!(mse_loss, model, X_train, y_train, X_test, y_test; num_epochs = num_epochs, α = lr)
errors = hcat(train_error, test_error)

plot(
    collect(1:num_epochs + 1), errors,
    label=["train error" "test_error"],
    xlabel="epoch", 
    ylabel="mse", 
    linewidth = [2 2]
    )

rmse_movielens(model, X_train, y_train)
right_predicted_movielens(model, X_train, y_train)

rmse_movielens(model, X_test, y_test)
right_predicted_movielens(model, X_test, y_test)




real_item_vec = [1 0 3 0 7 9 0 0 6 5]
user_item_pairs = copy([1 3 5 6 9 10; 1 1 1 1 1 1]')
item_rating = [1, 3, 7, 9, 6, 5]
vec = zeros(length(real_item_vec))
restore_item_vector!(vec, user_item_pairs, item_rating, 1)


R = rand(15, 15)
dummy_autorec = AutoRec(15, 5; dropout_rate = 0)
mse_loss = AUTOREC_MSE_LOSS()
RX, Ry = ui_matrix_to_movielens_form(R)
rmse_movielens(dummy_autorec, RX, Ry)
norm(dummy_autorec(R) - R, 2) / sqrt(prod(size(R)))