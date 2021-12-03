using Pkg
#First move to correct directory with cd("......./BenchmarkModelsForDQNProject")
Pkg.activate(".")

using POMDPs
using BenchmarkModelsForDQNProject
using StaticArrays
using Distributions: Uniform
using Flux
using DeepQLearning
using Mill
using POMDPPolicies
using POMDPSimulators


mdp = Cont_Collection_HPH_MDP(step_size=0.5f0)

# test if it works without errors
# get_particles(s::StateCollection) = s.particles
#
# s = rand(initialstate(mdp))
#
# sp = transition(mdp, s, Direction(1., 1.))
#
# get_position(s)
# a = Direction(1., 1.)
# get_position(s) + a + SMatrix{2, 3, Float32}(rand(Uniform(0., 0.25), (2, 3)))
#
# AN = ArrayNode(hcat(POMDPs.convert_s.(Array, sp.particles, mdp)...))
# im = ArrayModel(Dense(3, 3))
#
# agg = mean_aggregation(1)
#
# BN = BagNode(AN, [1:Mill.nobs(AN)])
#
# bm = ArrayModel(Dense(4, 3))
#
# BM = BagModel(ArrayModel(Dense(3, 3, Flux.tanh)), mean_aggregation(1), ArrayModel(Dense(4, 3)))
#
# A = ArrayNode(vcat([[[1, 2], [1, 2]], [1, 2]]...))
#
# hcat([[1 2; 3 4], [5; 6]]...)

function DeepQLearning.isrecurrent(m::BagModel)
    return false
end

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.5, steps=120000))

model = BagModel(ArrayModel(Dense(3, 3)), mean_aggregation(3),
                ArrayModel(Dense(4, length(POMDPs.actions(mdp)))))

solver = DeepQLearningSolver(qnetwork = model, max_steps=150000,
                             exploration_policy = exploration,
                             learning_rate=0.005,log_freq=1000,
                             recurrence=false,double_q=true, dueling=false, prioritized_replay=true,
                             verbose=true, max_episode_length=150)

model.im.m.weight == solver.qnetwork.im.m.weight
policy = solve(solver, mdp)


using MCTS

solver2 = DPWSolver()
policy2 = solve(solver2, mdp)

# # velikosti jednotlivych bagu
# last.(size.(r._s_batch))
#
# # kumulaticni soucty ukladajici posledni vyskyt daneho bagu
# cumsum(last.(size.(r._s_batch)))
#
#
# vcat(collect.([(csum[i-1] + 1):csum[2], (csum[4] + 1):csum[5]])...)
#
# # pole ktery ma o jeden index vic a ukazuje zacatky bagu
# (pushfirst(cumsum(last.(size.(r._s_batch))), 0) .+ 1)
#
#
# # vytvoreni listu generatoru
# (:).((pushfirst(cumsum(last.(size.(r._s_batch))), 0) .+ 1)[1:end-1], cumsum(last.(size.(r._s_batch))))


sim = RolloutSimulator(max_steps=200)
r_tot = simulate(sim, mdp, policy)
sim2 = RolloutSimulator(max_steps=200)
r_tot = simulate(sim2, mdp, policy2)

# policy = RandomPolicy(mdp)
println("\nstart\n")
for (s, a, r) in stepthrough(mdp, policy2, "s,a,r", max_steps=200)
    println("in state $s")
    println("took action $a")
    println("received reward $r")
end

###SARSOP NEFUNGUJE PROTOZE MDP, DISCRETE VALUE ITERATION NEFUNGUJE PROTOZE CONTINUOUS STATE SPACE


using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration

VERTICES_PER_AXIS = 61 # Controls the resolutions along the grid axis
grid = RectangleGrid(range(0.0f0, stop=6.0f0, length=VERTICES_PER_AXIS), range(0.0f0, stop=6.0f0, length=VERTICES_PER_AXIS), [0.0f0, 1.0f0, 2.0f0]) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid


approx_solver = LocalApproximationValueIterationSolver(interp, verbose=true, max_iterations=1000, is_mdp_generative=false)
approx_policy = solve(approx_solver, mdp)
asd
