export MF_MSE_LOSS, AUTOREC_MSE_LOSS

struct MF_MSE_LOSS <: RecLoss
end

(l::MF_MSE_LOSS)(model::MF, user_id::Int, item_id::Int, R::Matrix{<:Real}) = abs2(model(user_id, item_id) .- R[user_id, item_id])
(l::MF_MSE_LOSS)(model::MF, user_id, item_id, y::Real) = abs2(model(user_id, item_id) - y)
(l::MF_MSE_LOSS)(model::MF, (user_id, item_id), y::Real) = l(model, user_id, item_id, y)
(l::MF_MSE_LOSS)(model::MF, x::Vector, y::Vector) = mean(abs2.(model.(x) .- y))
(l::MF_MSE_LOSS)(model::MF, x::Matrix, y::Vector) = mean(abs2.(model(x) .- y))
(l::MF_MSE_LOSS)(model::MF, x::Vector, y::Real) = (abs2(model(x[1], x[2]) - y))


struct AUTOREC_MSE_LOSS <: RecLoss end
(l::AUTOREC_MSE_LOSS)(m::AutoRec, x, y, λ = 0) = norm(y - (y .!= 0) .* m(x), 2) + λ * (norm(m.encoder.weight, 2) + norm(m.decoder.weight, 2))
