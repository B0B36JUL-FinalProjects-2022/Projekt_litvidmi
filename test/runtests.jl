using SysRec
using Test
using LinearAlgebra
using Random

@testset "SysRec.jl" begin
    @testset "Utils" begin
        @testset "Restore vector" begin
            real_item_vec = [1, 0, 3, 0, 7, 9, 0, 0, 6, 5]
            user_item_pairs = copy([1 3 5 6 9 10; 1 1 1 1 1 1]')
            item_rating = [1, 3, 7, 9, 6, 5]
            vec = zeros(length(real_item_vec))
            restore_item_vector!(vec, user_item_pairs, item_rating, 1)
            @test real_item_vec == convert.(Int, vec)
        end
    end

    @testset "Model" begin
        @testset "RMSE" begin
            R = rand(15, 15) * 100
            dummy_autorec = AutoRec(15, 5; dropout_rate = 0)
            mse_loss = AUTOREC_MSE_LOSS()
            RX, Ry = ui_matrix_to_movielens_form(R)
            n_ratings = sqrt(prod(size(R)))

            my_rmse = rmse_movielens(dummy_autorec, RX, Ry)
            norm_frob = norm(dummy_autorec(R) - R, 2)

            @test abs(my_rmse - norm_frob / n_ratings) < 1e-8
        end
    end
end

### todo :  ui_matrix_to_movielens_form?