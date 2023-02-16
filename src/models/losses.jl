export MF_MSE_LOSS, AUTOREC_MSE_LOSS

struct MF_MSE_LOSS <: RecLoss end

# (l::MF_MSE_LOSS)(
#     model::MF,
#     args...; 
#     λ = 0
# ) = _mf_mse_loss(model, args...) + λ * (norm(model.P.weight, 2) + norm(model.Q.weight, 2))

# _mf_mse_loss(
#     model::MF, 
#     user_id::Int, 
#     item_id::Int, 
#     R::Matrix{<:Real}
# ) = abs2(model(user_id, item_id) .- R[user_id, item_id])

# _mf_mse_loss(model::MF, user_id, item_id, y) = abs2(model(user_id, item_id) - y)
# _mf_mse_loss(model::MF, (user_id, item_id), y) = abs2(model(user_id, item_id) - y)
# _mf_mse_loss(model::MF, x::Vector, y::Vector) = mean(abs2.(model.(x) .- y))
# _mf_mse_loss(model::MF, x::Matrix, y::Vector) = mean(abs2.(model(x) .- y))
# _mf_mse_loss(model::MF, x::Vector, y::Real) = (abs2(model(x[1], x[2]) - y))

(l::MF_MSE_LOSS)(
    model::MF, 
    user_id::Int, 
    item_id::Int, 
    R::Matrix{<:Real}
) = abs2(model(user_id, item_id) .- R[user_id, item_id]) + 
λ * (norm(model.P.weight, 2) + norm(model.Q.weight, 2))

(l::MF_MSE_LOSS)(model::MF, user_id::Integer, item_id::Integer, y, λ = 0) = 
abs2(model(user_id, item_id) - y) + λ * (norm(model.P.weight, 2) + norm(model.Q.weight, 2))
(l::MF_MSE_LOSS)(model::MF, (user_id, item_id), y, λ = 0) =
l(model, item_id, user_id, y, λ)
(l::MF_MSE_LOSS)(model::MF, x::Vector, y::Vector, λ = 0) = 
mean(abs2.(model.(x) .- y))  + λ * (norm(model.P.weight, 2) + norm(model.Q.weight, 2))
(l::MF_MSE_LOSS)(model::MF, x::Matrix, y::Vector, λ = 0) = 
mean(abs2.(model(x) .- y)) + λ * (norm(model.P.weight, 2) + norm(model.Q.weight, 2))
(l::MF_MSE_LOSS)(model::MF, x::Vector, y::Real, λ = 0) = 
(abs2(model(x[1], x[2]) - y)) + λ * (norm(model.P.weight, 2) + norm(model.Q.weight, 2))


struct AUTOREC_MSE_LOSS <: RecLoss end
(l::AUTOREC_MSE_LOSS)(m::AutoRec, x, y, λ = 0) = norm(y - (y .!= 0) .* m(x), 2) + λ * (norm(m.encoder.weight, 2) + norm(m.decoder.weight, 2))
