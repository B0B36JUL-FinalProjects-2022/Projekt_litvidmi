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
        @testset "Matrix to movielens format" begin
            R = [1 0 3 0; 7 9 0 0; 8 6 2 3]
            X = copy([1 1 2 2 3 3 3 3; 1 3 1 2 1 2 3 4]')
            y = [1, 3, 7, 9, 8, 6, 2, 3]
            RX, Ry = ui_matrix_to_movielens_form(R)

            @test (X, y) == (RX, Ry)
        end
        @testset "Split into batches" begin
            n = 40
            batch_size = 3
            original = collect(1:40)
            splitted = split_into_batches(n, batch_size)
            reconstructed = reduce(vcat, splitted)

            @test length(splitted) == ceil(n / batch_size) && reconstructed == original
        end
    end


    @testset "Models" begin
        @testset "RMSE for AutoRec" begin
            R = rand(15, 15) * 100
            dummy_autorec = AutoRec(15, 5; dropout_rate = 0)
            mse_loss = AUTOREC_MSE_LOSS()
            RX, Ry = ui_matrix_to_movielens_form(R)
            n_ratings = sqrt(prod(size(R)))
            my_rmse = rmse_movielens(dummy_autorec, RX, Ry)
            norm_frob = norm(dummy_autorec(R) - R, 2)

            @test abs(my_rmse - norm_frob / n_ratings) < 1e-8
        end
        @testset "MF gradients sanity" begin
            R = [3 1 1 3 1;
                 1 2 4 1 3;
                 3 1 1 3 1;
                 4 3 5 3 3]
            RX, Ry = ui_matrix_to_movielens_form(R)
            model = MF(3, 4, 5)
            mse_loss = MF_MSE_LOSS()
            cum_grad_P = copy(model.P.weight) .* 0
            cum_grad_Q = copy(model.Q.weight) .* 0
            for i in 1:4
                for j in 1:5
                    cum_grad_P += gradient(mse_loss, model, (i, j), R[i, j])[1].P.weight
                    cum_grad_Q += gradient(mse_loss, model, (i, j), R[i, j])[1].Q.weight
                end
            end

            tol = 6
            @test round.(gradient(mse_loss, model, RX, Ry)[1].P.weight; sigdigits=tol) == round.(cum_grad_P ./ prod(size(R)); sigdigits=tol)
            @test round.(gradient(mse_loss, model, RX, Ry)[1].Q.weight; sigdigits=tol) == round.(cum_grad_Q ./ prod(size(R)); sigdigits=tol)
        end
    end
end
