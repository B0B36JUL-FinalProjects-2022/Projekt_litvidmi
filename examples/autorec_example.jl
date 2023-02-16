using Revise
using SysRec
using Plots
using Flux
using LinearAlgebra
using BSON: @save, @load



# simple example 
#----------------------------
# R is matrix which the model should recreate
R = [3 1 1 3 1;
     1 2 4 1 3;
     3 1 1 3 1;
     4 3 5 3 3]

model = AutoRec(4, 3; dropout_rate = 0)
mse_loss = AUTOREC_MSE_LOSS()

# look what internal functions of training loop return
#----------------------------
x = R[:, 1]
data = [(x, x)]
model(x)

sgd_step!(mse_loss, model, data, 1e-5; λ = 0.001)
mse_loss(model, R[:, 1], R[:, 1])
gradient(mse_loss, model, R[:, 1], R[:, 1], 0.1)[1]
#----------------------------


RX, Ry = ui_matrix_to_movielens_form(R)
num_epochs = 100
lr = 1e-2

train_error, _ = run_train_eval_loop!(
    mse_loss,
    model, 
    RX, Ry; 
    num_epochs = num_epochs, 
    α = lr, 
    λ = 0.1, 
    verbose = false,
    batch_size = 8
)

plot(
    collect(1:num_epochs + 1), 
    train_error, 
    label="mse", 
    xlabel="epoch", 
    ylabel="train_error", 
    linewidth = 2
)


# example on movielens
# load and preprocess data
#----------------------------
const dir = joinpath(dirname(@__FILE__), "..", "movie_data")
get_movielens_data!(dir)

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
hidden_categories = 500
model = AutoRec(num_users, hidden_categories, dropout_rate = 0.05)
mse_loss = AUTOREC_MSE_LOSS()
num_epochs = 50
lr = 1e-2

train_error, test_error = run_train_eval_loop!(
    mse_loss, 
    model, 
    X_train, 
    y_train;
    X_test = X_test, 
    y_test = y_test,
    num_epochs = num_epochs, 
    α = lr, 
    λ = 1e-2,
    batch_size = 32
)

errors = hcat(train_error, test_error)
plot(
    collect(1:num_epochs + 1), errors,
    label=["train error" "test_error"],
    xlabel="epoch", 
    ylabel="rmse", 
    linewidth = [2 2]
)

rmse_movielens(model, X_train, y_train)
precision_movielens(model, X_train, y_train)

rmse_movielens(model, X_test, y_test)
precision_movielens(model, X_test, y_test)


@save "ar_weights_091.bson" model
