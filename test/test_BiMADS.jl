@testset "BiMADS" begin
    function test(x)
        f1(x) = (x[1] + 1) .^ 2 + (x[2] - 1) .^ 2 - 10
        f2(x) = (x[1] + 12) .^ 2 + (x[2] - 3) .^ 2 + 20.0
        return f1, f2
    end

    @testset "B_points" begin
        cost = [1.0, 2.0]
        coordinates = [3.0, 4.0]
        points = DS.B_points(cost, coordinates)
        @test points.cost == cost
        @test points.x_now == coordinates
        @test points.weight == 0
    end

    @testset "BiMADS_status" begin
        s = DS.BiMADS_status()
        @test s.iteration == 0
        @test s.func_evaluation == 0
        @test s.total_time == 0.0
        @test s.hypervolume == 0.0
        @test s.opt_status == DS.Unoptimized
        @test s.opt_string == "Unoptimized"
        @test typeof(s.start_time) == Float64
    end

    @testset "Pareto Evaluation" begin
        p = DSProblem(2; objective = test)
        result = Optimize!(p)
        hv1 = hvIndicator(result)
        p = DSProblem(2; objective = test)
        SetIterationLimit(p, 100)
        result = Optimize!(p)
        hv2 = hvIndicator(result)
        @test typeof(DS.pareto_front(result)) == Vector{DS.B_points}
        @test hv1 >= hv2
    end

    @testset "Initialization" begin
        @testset "p_split" begin
            p = DSProblem(2; objective = test)
            @test p_dim(p) == 2
            # if successfully splitting BOP into two SOPs
            p1, p2 = DS.p_split(p)
            x = zeros(Float64, p.N)
            p.objective(x)
            f1, f2 = p.objective(x)
            @test p1.objective == f1
            @test p2.objective == f2
        end
        @testset "Optimization for p1 and P2" begin
            SetIterationLimit(p1, 1)
            @test_throws ErrorException DS.get_Bpoints(p1, p2, 1)
            SetFunctionEvaluationLimit(p2, 1)
            @test_throws ErrorException DS.get_Bpoints(p2, p1, 2)
            #reset p1 p2 for other tests
            p1, p2 = DS.p_split(DSProblem(2; objective = test))
            @test length(DS.get_Bpoints(p1, p2, 1)) > 0
            @test length(DS.get_Bpoints(p2, p1, 2)) > 0
        end
        p1, p2 = DS.p_split(DSProblem(2; objective = test))
        @test typeof(DS.initial_X_L(p1, p2)) == Vector{DS.B_points}
    end
    @testset "Main Iteration" begin
        @testset "Reference Point Determination" begin
            undominated_points = Vector{DS.B_points}()
            # one undominated point
            push!(undominated_points, DS.B_points([0.1, 0.2], [0.3, 0.4]))
            j, δ, ref_point = DS.ReferencePointDetermination(undominated_points)
            @test j == 1.0 && δ == 1.0 && ref_point == [0.1, 0.2]
            # two undominated point
            push!(undominated_points, DS.B_points([0.5, 0.6], [0.7, 0.8]))
            j, δ, ref_point = DS.ReferencePointDetermination(undominated_points)
            @test j == 1.0 &&
                  isapprox(0.32, δ, atol = 0.1) &&
                  ref_point == [0.5, 0.2]
        end

        @testset "Auxiliary function phi" begin
            p1, p2 = DS.p_split(DSProblem(2; objective = test))
            f1 = p1.objective
            f2 = p2.objective
            @test DS.phi(f1, f2, [16.0, 61.0], [-6.0, 1.0]) == -1.0
            @test DS.phi(f1, f2, [15.0, 61.0], [-6.0, 1.0]) ==
                  DS.phi(f1, f2, [16.0, 60.0], [-6.0, 1.0]) ==
                  0.0
            @test DS.phi(f1, f2, [14.0, 61.0], [-6.0, 1.0]) == 1.0
        end

        @testset "Stopping Conditions" begin
            @testset "Iteration Limit" begin
                p = DSProblem(2; objective = test)
                SetIterationLimit(p, 62)
                @test_throws ErrorException Optimize!(p)
                status = DS.BiMADS_status()
                p.status.optimization_status = DS.IterationLimit
                undominated_points = Vector{DS.B_points}()
                @test DS.checkBiMADSStopping(p, status, undominated_points) ==
                      false
            end
            @testset "Function Evalution Limit" begin
                p = DSProblem(2; objective = test)
                SetFunctionEvaluationLimit(p, 234)
                @test_throws ErrorException Optimize!(p)
                status = DS.BiMADS_status()
                p.status.optimization_status = DS.FunctionEvaluationLimit
                undominated_points = Vector{DS.B_points}()
                @test DS.checkBiMADSStopping(p, status, undominated_points) ==
                      false
            end
            @testset "Run time Limit" begin
                p = DSProblem(2; objective = test)
                AddStoppingCondition(p, RuntimeStoppingCondition(0.001))
                @test_throws ErrorException Optimize!(p)
                status = DS.BiMADS_status()
                p.status.optimization_status = DS.RuntimeLimit
                undominated_points = Vector{DS.B_points}()
                @test DS.checkBiMADSStopping(p, status, undominated_points) ==
                      false
            end

            @testset "Hyper-Volume Limit" begin
                p = DSProblem(2; objective = test)
                AddStoppingCondition(p, HypervolumeStoppingCondition(1.0))
                @test typeof(Optimize!(p)) == Vector{DS.B_points}
                status = DS.BiMADS_status()
                p.status.optimization_status = DS.HypervolumeLimit
                undominated_points = Vector{DS.B_points}()
                @test DS.checkBiMADSStopping(p, status, undominated_points) ==
                      false
            end

        end
    end

end
