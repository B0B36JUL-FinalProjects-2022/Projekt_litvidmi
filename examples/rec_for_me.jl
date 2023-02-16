using SysRec
using CSV
using DataFrames
using Tables

const dir = joinpath(dirname(@__FILE__), "..", "movie_data")
get_movielens_data!(dir)

data_file = joinpath(dir, "ml-100k/u.data")
df, num_users, num_items = read_movielens_to_df(data_file)
sparsity = 1 - length(df.user_id) / (num_users * num_items)


#change items and users so I train a user-based AutoRec 
X, y = [df.item_id df.user_id], df.rating
X_train, y_train, X_test, y_test = train_test_split(X, y)

#----------------------------
# create, train and evaluate a model
#----------------------------

hidden_categories = 500
model = AutoRec(num_items, hidden_categories, dropout_rate = 0.1)
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
    λ = 1e-2 * 2,
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



me_X = [1 1;2 1;3 1;12 1;17 1;23 1;33 1;42 1;50 1;51 1;55 1;56 1;64
      1;67 1;69 1;73 1;96 1;98 1;127 1;128 1;131 1;134 1;177 1;178
      1;188 1;194 1;210 1;226 1;231 1;237 1;241 1;249 1;257 1;318
      1;395 1;452 1;475 1]

me_y = [4, 5, 5, 5, 4, 5, 4, 5, 4, 4, 5, 5, 5, 2, 5, 4, 3, 4,
        5, 1, 4, 4, 5, 5, 5, 5, 5, 2,2, 5, 4, 3, 4, 5, 5, 2, 5]

me_restored = zeros(num_items)
restore_item_vector!(me_restored, X_train, y_train, 1)
me_restored = model(me_restored)

movies_df = CSV.read("./movie_data/ml-100k/u.item", DataFrame; header=false)
indices = movies_df.Column1
names = movies_df.Column2

my_recommendations = names[reverse(sortperm(me_restored))]

CSV.write("recs_for_me.csv",  Tables.table(my_recommendations), writeheader=false)