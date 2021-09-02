using Statistics
export p_dim, hvIndicator,p_MADS, BiMADS_status, paretoCoverage,plot_Bpoint
#TODO BiMADS
logocolors = Colors.JULIA_LOGO_COLORS
"""
points for BiMADS Algo in objective space
"""
mutable struct B_points
    cost::Vector{Float64}  #f1 and f2 value in objective space
    x_now::Vector{Float64}
    weight::Int            #weight for each undominated point to determine the δ
    function B_points(cost::Vector{Float64},x::Vector{Float64}, w::Int=0)
        p = new()
        p.cost = cost
        p.x_now=x
        p.weight = w
        return p
    end
end
"""
Status for BiMADS main iteration, also contain stop conditions

"""
mutable struct BiMADS_status
    iteration::Int64                #Total iteration spent
    func_evaluation::Int64          #Total function evaluation spent
    total_time::Float64             #Totlal running time
    hypervolume::Float64            #Hypervolume for current Pareto set
    opt_status::OptimizationStatus
    opt_string::String
    start_time::Float64
    function BiMADS_status()
        s=new()
        s.iteration=0
        s.func_evaluation=0
        s.total_time=0.0
        s.hypervolume=0.0
        s.opt_status=Unoptimized
        s.opt_string="Unoptimized"
        s.start_time=time()
        return s
    end
end

"""
    plot_Bpoint(points::Vector{B_points})

Plot the Pareto front
"""
function plot_Bpoint(points::Vector{B_points})
    fig = Plots.scatter()
    for i = 1:length(points)
        fig = plot!([points[i].cost[1]],[points[i].cost[2]],seriestype = :scatter,aspect_ratio=1,legend = false,color=logocolors.red,show = true)
    end
    plot!(fig,xlabel="f1 cost",ylabel="f2 cost")
return fig
end

function get_adj_dis(points::Vector{B_points})
    println("-----------------")
    for i=1:length(points)-1
        display(LinearAlgebra.norm(points[i+1].cost - points[i].cost)^2)
    end
    println("-----------------")
end

function paretoCoverage(paretoSet::Vector{B_points})::Tuple{Float64,Float64}
    points=Vector{Float64}()
    for i=1:length(paretoSet)-1
        push!(points,LinearAlgebra.norm(paretoSet[i+1].cost - paretoSet[i].cost)^2)
    end
    return mean(points), std(points)
end


"""
    hvIndicator(paretoSet::Vector{B_points},factor=1.1)::Float64

Hyper-Volume indicator for evalating the Pareto Front
factor is used for choosing the referencepoint for hypervolume calculation
"""
function hvIndicator(paretoSet::Vector{B_points},factor=1.1)::Float64
    points=Vector{Vector{Float64}}()
    for i=1:length(paretoSet)-1
        push!(points,paretoSet[i].cost)
    end
    length(points)<2 && return 0.
     ref=factor.*[last(points)[1],first(points)[2]]
     # normalize_factor=(ref[1]-first(points)[1]).*(ref[2]-last(points)[2])./2
     normalize_factor=(last(points)[1]-first(points)[1]).*(first(points)[2]-last(points)[2])./2
     hv_volume=0.
     area(x::Vector{Float64},y::Vector{Float64})=(abs(x[1]-y[1])).*(abs(x[2]-y[2]))
     for p in points
         hv_volume+=area(ref,p)
         ref[2]=p[2]
     end
     return hv_volume/normalize_factor
end


"""
    p_split(p::DSProblem)::Tuple{DSProblem,DSProblem}

To split Bi-obj problem into two single obj problems
"""
function p_split(p::DSProblem)::Tuple{DSProblem,DSProblem}
    x = zeros(Float64, p.N)
    f1, f2 = p.objective(x)
    p1::DSProblem = deepcopy(p::DSProblem)
    p2::DSProblem = deepcopy(p::DSProblem)
    SetObjective(p1, f1)
    SetObjective(p2, f2)
    return p1, p2
end

"""
    get_Bpoints(p1::DSProblem,p2::DSProblem,flag::Int)

Get the points to determine the reference point and Pareto front
flag for identifying optimize p1 or p2
"""
function get_Bpoints(p1::DSProblem, p2::DSProblem, flag::Int)
    temp_Bpoints = Vector{B_points}()
    Setup(p1)
    # push B_point for initial point
    flag == 1 &&
        push!(temp_Bpoints, B_points([p1.objective(p1.x), p2.objective(p1.x)],p1.x))

    while _check_stoppingconditions(p1)
        p1.full_output && OutputIterationDetails(p1)
        if OptimizeLoop(p1) == Dominating
            flag == 1 &&
                push!(temp_Bpoints, B_points([p1.x_cost, p2.objective(p1.x)],p1.x))
            flag == 2 &&
                push!(temp_Bpoints, B_points([p2.objective(p1.x), p1.x_cost],p1.x))
        end
    end

    Finish(p1)

    if p1.status.optimization_status != MeshPrecisionLimit && p1.status.optimization_status != PollPrecisionLimit && p1.status.optimization_status != KeyInterrupt
        error("The $(p1.status.optimization_status_string) is not enough for BiMADS, please increase it")
    end
    return temp_Bpoints
end

"""
    pareto_front(points::Vector{B_points}, minimize = true)

find the Pareto points, only for this BiMADS.jl, not general methods
"""
function pareto_front(points::Vector{B_points}, minimize = true)
    cmp_strict = minimize ? (<) : (>)
    cmp = minimize ? (<=) : (>=)
    dims = 2 #number of objective functions, needs modification for multi-MADS
    strictly_dominates(u::B_points, v::B_points) =
        all(cmp(u.cost[i], v.cost[i]) for i = 1:dims) &&
        any(cmp_strict(u.cost[i], v.cost[i]) for i = 1:dims)
    undominated(p::B_points) =
        !any(
            strictly_dominates(p2::B_points, p::B_points) for
            p2 ∈ points if p2.cost != p.cost
        )
    filter!(undominated, points)
    sort!(points, by = v -> v.cost, rev = false)
    return points
end

"""
    initial_X_L(p1, p2)::Vector{Float64}

get the initial undominated points from p1 and p2
"""
function initial_X_L(p1, p2)::Vector{B_points}
    x_L = Vector{B_points}()
    # optmization for p1
    append!(x_L, get_Bpoints(p1, p2, 1))
    x_L = reverse(x_L)
    # optmization for p2
    append!(x_L, get_Bpoints(p2, p1, 2))
    return x_L
end

"""
    get_ref(points::Vector{B_points})::Int

get the reference point for L>2
"""
function get_ref(points::Vector{B_points})::Int
    dis(x::Int) = LinearAlgebra.norm(points[x].cost - points[x-1].cost)^2+LinearAlgebra.norm(points[x].cost - points[x+1].cost)^2
    max_index = 2 #point with largest distance to its adjacent points
    max_distance=0.

    for i = 2:length(points)-1
        if dis(i)/(points[i].weight+1) >= max_distance
            max_distance = dis(i)/(points[i].weight+1)
            max_index = i
        end
    end
    return max_index
end


"""
    ReferencePointDetermination(undominated_points::Vector{B_points})::Tuple{Int,Float64,Vector{Float64}}

get the reference point for any case
"""
function ReferencePointDetermination(undominated_points::Vector{B_points})::Tuple{Int,Float64,Vector{Float64}}
    L=length(undominated_points)
    if L==1
        j=1
        ref_point=undominated_points[1].cost
        δ= 1.
    elseif L==2
        j=1 #different definition in paper(j=2) and books(j=1)
        ref_point=[undominated_points[2].cost[1],undominated_points[1].cost[2]]
        δ=(LinearAlgebra.norm(undominated_points[1].cost - undominated_points[2].cost)^2)/(undominated_points[2].weight+1)
    else
        j = get_ref(undominated_points)
        ref_point=[undominated_points[j+1].cost[1],undominated_points[j-1].cost[2]]
        δ=(LinearAlgebra.norm(undominated_points[j].cost - undominated_points[j-1].cost)^2
        +LinearAlgebra.norm(undominated_points[j].cost - undominated_points[j+1].cost)^2)/(undominated_points[j].weight+1)
    end
    undominated_points[j].weight+=1
    return j, δ, ref_point
end

"""
    function phi(f1::Function,f2::Function, r::Vector{Float64}, x::Vector{Float64})

Auxiliary Function from BiMADS. Aim to refomulate the Bi-obj problem into
single-obj problem. Return the cost of the refomulated objective function

x is the point to be evaluated, r is the reference point
"""
function phi(f1::Function,f2::Function, r::Vector{Float64}, x::Vector{Float64})
    cost1=f1(x)
    cost2=f2(x)
    if (cost1 <= r[1]) && (cost2 <= r[2]) #if x dominant r
        return -(r[1] - cost1)^2 * (r[2] - cost2)^2
    elseif (cost1 == r[1]) || (cost2 == r[2])
        return 0
    else
        return (max(cost1 - r[1], 0))^2 + (max(cost2 - r[2], 0))^2
    end
end

"""
    update_p(p::DSProblem)
reset the reformulated problem for new MADS iteration
Including reset the Mesh/Poll size, IterationLimit, FunctionEvaluationLimit, RuntimeLimit, etc.

"""
# update p before the main iteration
function update_p(p,p1::DSProblem,p2::DSProblem,status::BiMADS_status)
    for c in p.stoppingconditions
        i=_get_conditionindexes(p,typeof(c))[1]
        if c isa IterationStoppingCondition
            status.iteration=p1.status.iteration+p2.status.iteration
            p.stoppingconditions[i].limit-=status.iteration
            p.stoppingconditions[i].limit<=1 && error("The Iteration limit is not enough for BiMADS, please increase it")
        elseif c isa FunctionEvaluationStoppingCondition
            status.func_evaluation=p1.status.function_evaluations+p2.status.function_evaluations
            p.stoppingconditions[i].limit-=status.func_evaluation
            p.stoppingconditions[i].limit<=1 && error("The Function evaluation limit is not enough for BiMADS, please increase it")
        elseif c isa RuntimeStoppingCondition
            status.total_time=p1.status.runtime_total+p2.status.runtime_total
            p.stoppingconditions[i].limit -=status.total_time
            p.stoppingconditions[i].limit<=0 && error("The Runtime limit is not enough for BiMADS, please increase it")
        end
    end
end

# update p during the main iteration
function update_p(p_init::DSProblem,p_reform::DSProblem,status::BiMADS_status)
    status.iteration+=p_reform.status.iteration
    status.func_evaluation+=p_reform.status.function_evaluations
    status.total_time=p_reform.status.runtime_total
    p_reform=deepcopy(p_init)

    for c in p_reform.stoppingconditions
        i=_get_conditionindexes(p_reform,typeof(c))[1]
        if c isa IterationStoppingCondition
            p_reform.stoppingconditions[i].limit=p_init.stoppingconditions[i].limit-status.iteration
        elseif c isa FunctionEvaluationStoppingCondition
            p_reform.stoppingconditions[i].limit=p_init.stoppingconditions[i].limit-status.func_evaluation
        end
    end
    return p_reform
end
# update p after the main iteration
# function update_p(p_reform::DSProblem,status::BiMADS_status)
#     for c in p.stoppingconditions
#         i=_get_conditionindexes(p,typeof(c))[1]
#         if c isa IterationStoppingCondition
#             status.iteration+=p_reform.status.iteration
#         elseif c isa FunctionEvaluationStoppingCondition
#             status.func_evaluation+=p_reform.status.function_evaluations
#         end
#     end
# end

"""
    checkKeyInterrupt(status::BiMADS_status)::Bool
Check the input from REPL and stop the optimization

"""
# for BiMADS
function checkKeyInterrupt(status::BiMADS_status)::Bool
    bb = bytesavailable(stdin)
    data = read(stdin, bb)
    if bb>0 && data==UInt8[0x71] #q for quit
        println("quit")
        status.opt_status=KeyInterrupt
        return false
    end
    return true
end

"""
    checkBiMADSStopping(p::DSProblem,status::BiMADS_status,undominated_points::Vector{B_points})::Bool
Check the stopping condition for BiMADS

"""
function checkBiMADSStopping(p::DSProblem,status::BiMADS_status,undominated_points::Vector{B_points})::Bool
    p.status.runtime_total = time() - status.start_time
    if p.status.optimization_status!=PollPrecisionLimit && p.status.optimization_status!=MeshPrecisionLimit
        # add the numbers from last iteration
        status.opt_status=p.status.optimization_status
        status.iteration+=p.status.iteration
        status.func_evaluation+=p.status.function_evaluations
        return false
    end

    for c in p.stoppingconditions
        if c isa RuntimeStoppingCondition
            status.opt_status=RuntimeLimit
            i=_get_conditionindexes(p,RuntimeStoppingCondition)[1]
            p.status.runtime_total>p.stoppingconditions[i].limit && return false
        end
        if c isa HypervolumeStoppingCondition
            status.opt_status=HypervolumeLimit
            i=_get_conditionindexes(p,HypervolumeStoppingCondition)[1]
            diff=hvIndicator(undominated_points)-status.hypervolume
            status.hypervolume=hvIndicator(undominated_points)
            status.hypervolume>=p.stoppingconditions[i].limit && return false #if the HV smaller than a certain value
            # diff<=p.stoppingconditions[i].limit && return false #if the change of HV smaller than threshold
        end
    end

    !checkKeyInterrupt(status) && return false
    return true
end

"""
    Optimize_Bi!(p::DSProblem)

Run the BiMADS algorithm on problem `p`.

`p` must have had its initial point and objective function set. If extreme
barrier constraints have been set then the initial point must be value for
those constraints.
"""
function Optimize_Bi!(p::DSProblem)
    println("BiMADS")
    status=BiMADS_status()
    p1::DSProblem, p2::DSProblem = p_split(p)
    f1 = p1.objective
    f2 = p2.objective

# SetIterationLimit(p1,10)
# SetIterationLimit(p2,2)


    # Initialization
    undominated_points = pareto_front(initial_X_L(p1, p2))
# display(undominated_points)
# display(plot_Bpoint(undominated_points))

    p_reform=deepcopy(p)
    update_p(p_reform,p1,p2,status)

    iteration_count=0
    ref_point=Vector{Float64}()
    j = 0     #initial point for the refomulated sigle-obj problem
    δ = 0.0 # quantify the quality of the coverage of the undominated points
            # here is the coverage of the point with largest distance to its adjacent points
# fig1=plot_Bpoint(undominated_points)
# fig2=plot_Bpoint(undominated_points)
    # Main iteration
    while true
        count=0
# count2=0
# eva=500
# tti=10000
# length(undominated_points)>200 && break
        #Reference point determination
        j, δ, ref_point=ReferencePointDetermination(undominated_points)

        #Single-objective formulation minimization
        f_reform(x)=phi(f1,f2, ref_point, x)
        SetObjective(p_reform, f_reform)
        SetInitialPoint(p_reform,undominated_points[j].x_now)
# SetIterationLimit(p_reform,10)
# fig1=plot_Bpoint(undominated_points)
# p_reform.full_output=true
        #run MADS for refomulated problem and add new undominated points
        Setup(p_reform)

        while _check_stoppingconditions(p_reform)
            p_reform.full_output && OutputIterationDetails(p_reform)
            if OptimizeLoop(p_reform) == Dominating
                    push!(undominated_points, B_points([p1.objective(p_reform.x), p2.objective(p_reform.x)],p_reform.x))
                    count+=1
# count2+=1



# p_reform.status.function_evaluations>eva && break
# p_reform.status.optimization_status=PollPrecisionLimit
# count>3000 && break
# plot!(fig2,[p1.objective(p_reform.x)],[p2.objective(p_reform.x)],seriestype = :scatter,color=logocolors.blue)
            end
        end
        # p_reform.status.runtime_total = time() - status.start_time
# fig2=plot_Bpoint(undominated_points)

# display(fig2)

        #update X_L
        pareto_front(undominated_points)
        println("===============================")
        println("Start of iteration ",iteration_count)
        println("j=",j,"  ref=",ref_point)
        println("Add new points:",count)
        println("Hyper-Volume:",hvIndicator(undominated_points))
        println("Total undominated points: ", length(undominated_points))
        println("===============================")
        iteration_count+=1
# iteration_count>tti && break
# @show p_reform.status.optimization_status
# p_reform=update_p(p,p_reform,status)
        !checkBiMADSStopping(p_reform,status,undominated_points) && break
        p_reform=update_p(p,p_reform,status)
# length(undominated_points)>=474 && break
        # n_iteration>=p.stoppingconditions[1].limit ? (n_iteration+=1) : (n_iteration+=1)
        # display(undominated_points)
        # fig3=plot_Bpoint(undominated_points)
        # figall=plot(fig2,fig3,aspect_ratio=1)
        # savefig(figall, "/Users/zyy/Desktop/XJTLU/MSc_Project/Julia/test_julia/Results/DS_result_$(iteration_count-1).pdf");

    end
    # update_p(p_reform,status)
    println("===============================")
    println("Total undominated points:", length(undominated_points))
    println("Total Iterations: ",status.iteration)
    println("Total Function Evaluations: ",status.func_evaluation)
    println("Total Run Time: ",p_reform.status.runtime_total)
    println("Hyper-Volume: ",hvIndicator(undominated_points))
    println("Optimization Status: ",status.opt_status)
    println("===============================")
    return undominated_points
end

"""
    p_dim(obj::Function,p.N)::Int

The dimension of the given objective function
If the dimension==1, reset the objective function for MADS
"""
function p_dim(p::DSProblem)::Int
    dim=length(p.objective(ones(p.N)))
    if dim==1
        x = zeros(Float64, p.N)
        f= p.objective(x)
        SetObjective(p, f[1])
        return dim
    end
    return dim
end
