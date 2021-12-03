# Discrete version of Heaven, Priest and Hell World from Th(r)un 99
# In order to receive the reward, the agent has to arrive to Heaven, but he does not know where it is.
# To obtain the position of heaven, he must ask the priest, denominated with P.
#
# #########
# #S     ?#
# #       #
# #       #
# #?     P#
# #########
#
# World is 6 X 6 Grid with Heaven location incorporated (1323 total states possible) - more like 108 states
# Agent can go into 8 directions -  1 - North-East, 2 - East, 3 - South-East, 4 - South,
#                                   5 - South-West, 6 - West, 7 - North-West, 8 - North,
# Each Action is one long in max norm
# Each action can have approx 25% error - represented by normal distribution Normal(0, 0.25)
# Agent can obtain 3 observations that stand for the Heaven position
# 0 - Unknown, 1 - Lower-Left corner, 2 - Upper-Right corner
#
# world state is represented by Int according to following formula
# state = (Heaven position) * 6 ^ 2 + (y) * 6 + (x) + 1
# state = state_index

using POMDPs
using Distributions
using POMDPModelTools
using POMDPSimulators
using DeepQLearning
using Flux
using POMDPPolicies
using MCTS
using D3Trees
using Random
using POMDPModels
using ParticleFilters
using DataStructures
using StaticArrays

using Pipe: @pipe

import POMDPModelTools: Uniform





struct State
    x::Int
    y::Int
    heaven::Int
    State(x, y, heaven) = new(max(min(5, x), 0), max(min(5, y), 0), heaven)
end

function State(coord::Tuple{Int64, Int64})
    State(coord[1], coord[2], 0)
end|


function State(coord::Tuple{Int64, Int64, Int64})
    State(coord...)
end

function State(x, y)
    State(x, y, 0)
end

Base.max(a::State) = a

Base.isless(a::State, b::State) = a.heaven * 36 + min(5, a.y) * 6 + min(5, a.x) + 1 <
                                b.heaven * 36 + min(5, b.y) * 6 + min(5, b.x) + 1

Base.broadcastable(ss::State) = Ref(ss)

import Base.:+
import Base: ==

+(ss::State, step::Tuple{Int, Int})::State = State(ss.x + step[1], ss.y + step[2], ss.heaven)
==(ss1::State, ss2::State)::Bool = ss1.x == ss2.x && ss1.y == ss2.y && ss1.heaven == ss2.heaven
==(ss1::State, ss2::Tuple{Int, Int})::Bool = ss1.x == first(ss2) && ss1.y == last(ss2)


struct MCPOMDPWorld <: POMDP{State, Int, State}
    heaven::Int
end

function MCPOMDPWorld()
    MCPOMDPWorld(rand(Categorical([0.5, 0.5])))
end




POMDPs.states(m::MCPOMDPWorld) = vcat(vec(State.(Iterators.product(0:5, 0:5, 0:2))), State(5, 0, 3)) # mozna bude potreba transponovat
POMDPs.actions(m::MCPOMDPWorld) = 1:8
# POMDPs.observations(m::MCPOMDPWorld) = 1:4


POMDPs.stateindex(m::MCPOMDPWorld,  ss::State)::Int = ss == State(5, 0, 3) ? 109 : ss.heaven * 36 + min(5, ss.y) * 6 + min(5, ss.x) + 1
POMDPs.actionindex(m::MCPOMDPWorld, a::Int)::Int = a
# POMDPs.obsindex(m::MCPOMDPWorld, o::Int)::Int = o


POMDPs.initialstate(m::MCPOMDPWorld) = SparseCat([State(0, 0, 0)], [1.])
# POMDPs.initialstate(m::MCPOMDPWorld) = Deterministic(1)
POMDPs.discount(m::MCPOMDPWorld)::Float64 = .99

POMDPs.isterminal(m::MCPOMDPWorld, ss::State)::Bool = ss == (5, 0) || ss == (0, 5)
POMDPs.isterminal(m::MCPOMDPWorld, ss::SparseCat)::Bool = pdf(ss, (5, 0)) > 0.9 || pdf(ss, (0, 5)) >  .9
POMDPs.isterminal(m::MCPOMDPWorld, ss::Array)::Bool = pdf(ss, (5, 0)) > 0.9 || pdf(ss, (0, 5)) >  .9


function POMDPs.reward(m::MCPOMDPWorld, ss::State, a::Int, sp::State)::Float64
    sp == State(5,0,3) && return 0.
    !isterminal(m, sp) && return -1.
    sp.heaven == m.heaven && return 1000
    if sp.heaven == 0
        (sp == (0, 5) && m.heaven == 1 || sp == (5, 0) && m.heaven == 2) && return 0
        return -500
    end
    return -5000
end


function aggregate_dist_to_sparsecat(pomdp::MCPOMDPWorld, target_states)
    probs = (0.025, 0.025, 0.025, 0.025, 0.8, 0.025, 0.025, 0.025, 0.025)
    vals = Dict{State, Float64}()
    for (id, state) in enumerate(target_states)
        if state == (5, 5)
            state = State(5, 5, pomdp.heaven)
        end
        if haskey(vals, state)
            vals[state] += probs[id]
        else
            vals[state] = probs[id]
        end
    end
    # println("Aggregaating distances: $(keys(vals))")
    return SparseCat(keys(vals), values(vals))
end


function POMDPs.transition(m::MCPOMDPWorld, ss::State, a::Int)::SparseCat
    isterminal(m, ss) && return SparseCat([State(5, 0, 3)], [1.])
    if a % 2 != 0
        if a < 4
            if a == 1
                target_states = ss .+ Iterators.product((1:3), -(1:3)) # NE
            else
                target_states = ss .+ Iterators.product((1:3), (1:3)) #  SE
            end
        else
            if a == 7
                target_states = ss .+ Iterators.product(-(1:3), -(1:3)) # NW
            else
                target_states = ss .+ Iterators.product(-(1:3), (1:3)) # SW
            end
        end
    else
        if a < 5
            if a == 2
                target_states = ss .+
                    ((1, -1), (2, -1), (3, -1), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)) # E
            else
                target_states = ss .+
                    ((-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2), (-1, 3), (0, 3), (1, 3)) # S
            end
        else
            if a == 6
                target_states = ss .+
                    ((-3, -1), (-2, -1), (-1, -1),  (-3, 0), (-2, 0), (-1, 0), (-3, 1), (-2, 1), (-1, 1)) # W
            else
                target_states = ss .+
                    ((-1, -3), (0, -3), (1, -3), (-1, -2), (0, -2), (1, -2), (-1, 1), (0, 1), (1, 1)) # N
            end
        end
    end
    # println("SPARSE CAT: $(aggregate_dist_to_sparsecat(m, target_states))")
    return aggregate_dist_to_sparsecat(m, target_states)
end



# chybi observace doteku se zdi
# function POMDPs.observation(m::MCPOMDPWorld, a::Int, sp::State)::Uniform
#     sp == (5, 5) && return Uniform(stateindex(m, State(5, 5, m.heaven)))
#     sp.heaven == 0 && return Uniform(1:stateindex(m, State(4, 5, 0)))
#     sp.heaven == 1 && return Uniform(stateindex(m, State(0, 0, 1)):stateindex(m, State(4, 5, 1)))
#     sp.heaven == 2 && return Uniform(stateindex(m, State(0, 0, 2)):stateindex(m, State(4, 5, 2)))
# end
# POMDPs.initialobs(m::MCPOMDPWorld, sp::State) = observation(m, 0, sp)

# function POMDPs.obsindex(m::MCPOMDPWorld, o::Uniform)::Int
#     36 in support(o) && return 1
#     1 in support(o) && return 2
#     37 in support(o) && return 3
#     73 in support(o) && return 4
# end


# function POMDPs.observation(m::MCPOMDPWorld, a::Int, sp::State)::Uniform
#     println("Getting observation: $(sp), $(sp == (5,5))")
#     sp == (5, 5) && return Uniform([State(5, 5, m.heaven)])
#     sp.heaven == 1 && return Uniform([State(x, y, 1) for x in 0:4, y in 0:5])
#     sp.heaven == 2 && return Uniform([State(x, y, 2) for x in 0:4, y in 0:5])
#     sp.heaven == 0 && return Uniform([State(x, y) for x in 0:4, y in 0:5])
# end
function POMDPs.observation(m::MCPOMDPWorld, a::Int, sp::State)::Uniform
    # println("Getting observation: $(sp), $(sp == (5,5))")
    sp.heaven == 0 && return Uniform([State(x, y) for x in 0:5, y in 0:4])
    sp.heaven == 1 && return Uniform(vcat(vec([State(x, y, 1) for x in 0:5, y in 0:5]), State(5, 0, 3)))
    sp.heaven == 2 && return Uniform(vcat(vec([State(x, y, 2) for x in 0:5, y in 0:5]), State(5, 0, 3)))
    Uniform(vcat(vec([State(x, y, m.heaven) for x in 0:5, y in 0:5]), State(5, 0, 3)))
end
POMDPs.initialobs(m::MCPOMDPWorld, sp::State) = observation(m, 0, sp)
POMDPs.initialobs(m::MCPOMDPWorld) = observation(m, 0, State(0,0))
# function POMDPs.obsindex(m::MCPOMDPWorld, o::Uniform)::Int
#     any(s -> s == (5, 5), collect(support(env.o))) && return 1
#     State(0, 0, 0) in support(o) && return 2
#     State(0, 0, 1) in support(o) && return 3
#     State(0, 0, 2) in support(o) && return 4
#     println("CO SE DEJE?!")
#     println("SUPPORT: $(o)")
# end
function POMDPs.obsindex(m::MCPOMDPWorld, o::Uniform)::Int
    State(0, 0, 0) in support(o) && return 1
    # println(o.set)
    State(0, 0, 1) in support(o) && return 2
    State(0, 0, 2) in support(o) && return 3
    # println("CO SE DEJE?!")
    # println("SUPPORT: $(o)")
end


# function POMDPs.obsindex(m::MCPOMDPWorld, o::State)::Int
#     return stateindex(m, o)
# end



convert_s(::Type{A}, s::State, mdp::MCPOMDPWorld) where A<:AbstractArray = Float64[s.x, s.y, s.heaven]
convert_s(::Type{State}, s::AbstractArray, mdp::MCPOMDPWorld) = State(s[1], s[2], s[3])


function convert_s(::Type{A}, s::SparseCat, mdp::MCPOMDPWorld) where A<:AbstractArray
    ss = zeros(Float32, 109)
    for (state, prob) in s
        ss[stateindex(mdp, state)] = prob
    end
    return ss
end

import POMDPs:convert_o
# function convert_o(::Type{A}, o::Uniform, mdp::MCPOMDPWorld) where A<:AbstractArray
#     oi = obsindex(mdp, o)
#     println("Converting o: $(o), index: $(oi)")
#     o_array = zeros(109)
#     oi == 1 && (o_array[36] = 1.)
#     oi == 2 && (o_array[1:35] .= 1/35)
#     oi == 3 && (o_array[37:72] .= 1/36)
#     oi == 4 && (o_array[73:109] .= 1/36)
#     println("New observation array: $(o_array)")
#     return o_array
# end
function convert_o(::Type{A}, o::Uniform, mdp::MCPOMDPWorld) where A<:AbstractArray
    oi = obsindex(mdp, o)
    # println("Converting o: $(o), index: $(oi)")
    o_array = zeros(Float32, 109)
    oi == 1 && (o_array[1:36] .= 1/36)
    oi == 2 && (o_array[vcat(collect(37:72), 109)] .= 1/37)
    oi == 3 && (o_array[73:109] .= 1/37)
    # println("New observation array: $(o_array)")
    return o_array
end

Base.broadcastable(m::MCPOMDPWorld) = Ref(m)


# using POMDPPolicies
# using POMDPSimulators

# pomdp = MCPOMDPWorld()
# policy = RandomPolicy(pomdp)

# println("\nstart\n")
# for (s, a, o, r) in stepthrough(pomdp, policy, "s,a,o,r", max_steps=100)
#     println("in state $s")
#     println("took action $a")
#     println("received observation $o and reward $r")
# end





# @requirements_info MCTSSolver() MCPOMDPWorld() 1
# show_requirements(get_requirements(POMDPs.solve, (MCTSSolver(), MCPOMDPWorld())))




# # This defines how nodes in the tree view are labeled.
# function MCTS.node_tag(s::State)
#     if s.done
#         return "done"
#     else
#         return "[$(s.x),$(s.y), $(s.heaven)]"
#     end
# end

# ParticleCollection{State}
# function MCTS.node_tag(s::ParticleCollection{State})
#     return "[$(counter(particles(s)))]"
# end

# MCTS.node_tag(a::Int) = "[$a]"


# n_iter = 1000
# depth = 1000
# ec = 10000.0

# pomdp = MCPOMDPWorld()
# pomdp = LegacyGridWorld()
# pomdp = MiniHallway()


# solver = DPWSolver(n_iterations=n_iter,
#                    depth=depth,
#                    exploration_constant=ec,
#                    tree_in_info=true,
#                    max_time = 100.)

# #POMDP
# updater = SIRParticleFilter(pomdp, 200)
# solver = BeliefMCTSSolver(solver, updater)

# #MDP
# solver = MCTSSolver(n_iterations=n_iter,
#     depth=depth,
#     exploration_constant=ec,
#     enable_tree_vis=true
# )


# planner = solve(solver, pomdp)

# state = initialstate(pomdp, Random.MersenneTwister(4))
# state = initialstate(pomdp)

# a = action(planner, initialize_belief(updater, initialstate_distribution(pomdp)))
# a, info = action_info(planner, initialize_belief(updater, initialstate_distribution(pomdp)))
# inchrome(D3Tree(info[:tree], init_expand=10)) # click on the node to expand it


# no_simulations = 10_000
# mean([simulate(RolloutSimulator(max_steps = 100), pomdp, planner, updater(planner), initialstate(pomdp)) for i in 1:no_simulations])
# mean([simulate(planner, updater(planner), initialstate(pomdp)) for i in 1:no_simulations])


# planner



# pomdp = MCPOMDPWorld()
#
# # Define the Q network (see Flux.jl documentation)
# # the gridworld state is represented by a 2 dimensional vector.
# model = Chain(Dense(3, 32), Dense(32, length(actions(pomdp))))
# model = Chain(Dense(3, 32), Dense(32, 8))
#
# exploration = EpsGreedyPolicy(pomdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))
#
# solver = DeepQLearningSolver(qnetwork = model, max_steps=10,
#                              exploration_policy = exploration,
#                              learning_rate=0.005,log_freq=500,
#                              recurrence=false,double_q=false, dueling=false, prioritized_replay=false)
# solver
# policy = solve(solver, pomdp)


# solver = DeepQLearningSolver(max_steps=10000,
#                              exploration_policy = exploration,
#                              learning_rate=0.005,log_freq=500,
#                              recurrence=false,double_q=false, dueling=false, prioritized_replay=false)
# solver

# model
# solver.qnetwork = model
# solver

# sim = RolloutSimulator(max_steps=30)
#
# r_tot = simulate(sim, pomdp, policy)
# println("Total discounted reward for 1 simulation: $r_tot")

using CommonRLInterface

const RL = CommonRLInterface


POMDPCommonRLEnv{RLO}(m, s=initialstate(m), o=initialobs(m)) where {RLO} =
                    POMDPCommonRLEnv{RLO, typeof(m), SparseCat, Uniform}(m, s, o)


function RL.reset!(env::POMDPCommonRLEnv)
    env.s = initialstate(env.m)
    env.o = initialobs(env.m)
    return nothing
end

function RL.act!(env::POMDPCommonRLEnv, a, belief::AbstractArray)
    # sp, o, r = @gen(:sp, :o, :r)(env.m, env.s, a)


    # compute belief of new simulated state
    new_belief = sum(map((s, p) -> convert_s(Array, transition(env.m, s, a), env.m) .* p,
                states(env.m), belief))

    all_states = Dict(states(env.m) .=> new_belief)
    filtered_states = filter(r -> last(r) > 0., all_states)
    # println("Filtered Values: $(filtered_states)")
    # println("Sum: $(sum(values(filtered_states)))")
    env.s = SparseCat(collect(keys(filtered_states)), collect(values(filtered_states)))

    # println("Max State: $(max(collect(keys(filtered_states))...))")
    obs_state = collect(keys(filtered_states))

    # best_states = partialsortperm(collect(keys(filtered_states)), 1:2, rev=true)
    # print(best_states)

    env.o = observation(env.m, a, max(collect(keys(filtered_states))...))
    # println("Real observation: $(observation(env.m, a, max(collect(keys(filtered_states))...)))")

    # println("New S: $(env.s)")
    # println("New O: $(obsindex(env.m, env.o))")
    r = sum(map((sp, p) -> reward(env.m, State(0, 0), a, sp) * p, collect(support(env.s)), env.s.probs))
    return r
end

Deterministic

function RL.observe(env::POMDPCommonRLEnv{RLO}) where {RLO}
    # return convert_o(RLO, env.o, env.m)

    # convert states and obs to array representations
    s_new = convert_s(Array, env.s, env.m)
    # @show env.o
    o_new = convert_o(Array, env.o, env.m)

    # @show s_new
    # @show o_new

    # compute the game state
    updated_belief = @pipe o_new .* s_new |> _ ./ sum(_)

    return updated_belief
    # filter out the zero values
    # all_states = Dict(states(env.m) .=> updated_belief)
    # return all_states
    # filtered_states = filter(r -> last(r) > 0., all_states)

    # return filtered_states
end


RL.terminated(env::POMDPCommonRLEnv) = isterminal(env.m, env.s)

mdp = SimpleGridWorld();


mdp = MCPOMDPWorld()
env = POMDPCommonRLEnv{AbstractArray{Float32}}(mdp)

using POMDPModels
mdp = TigerPOMDP()
POMDPCommonRLEnv{RLO}(m, s=initialstate(m), o=initialobs(m)) where {RLO} =
        POMDPCommonRLEnv{RLO, typeof(m), BoolDistribution, BoolDistribution}(m, s, o)
POMDPs.initialobs(p::TigerPOMDP) = observation(p, 0, rand(initialstate(p)))
function convert_s(::Type{A}, s::Bool, mdp::TigerPOMDP) where A<:AbstractArray
    # return Float32[1 - s, s]
    return SVector{2,Float32}(1-s, s)
end
function convert_s(::Type{A}, s::BoolDistribution, mdp::TigerPOMDP) where A<:AbstractArray
    # return Float32[pdf(s, false), pdf(s, true)]
    return SVector{2,Float32}(pdf(s, false), pdf(s, true))
end
function convert_o(::Type{A}, o::BoolDistribution, mdp::TigerPOMDP) where A<:AbstractArray
    # return Float32[pdf(o, false), pdf(o, true)]
    return SVector{2,Float32}(pdf(o, false), pdf(o, true))
end
initialstate(mdp)
convert_s(Array, initialstate(mdp), mdp)
convert_o(Array, initialobs(mdp), mdp)






# Define the Q network (see Flux.jl documentation)
# the gridworld state is represented by a 2 dimensional vector.
model = Chain(Dense(109, 32), Dense(32, length(POMDPs.actions(mdp))))

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.1, steps=100000/2))

solver = DeepQLearningSolver(qnetwork = model, max_steps=1000,
                             exploration_policy = exploration,
                             learning_rate=0.01,log_freq=500,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true,
                             verbose=true)
# @enter
policy = solve(solver, mdp)

Array{Float64, 1} <: AbstractArray{<:Real, Number}

Array<: AbstractArray

RL.reset!(env)

env

sim = RolloutSimulator(max_steps=30 )
r_tot = simulate(sim, mdp, policy)
println("Total discounted reward for 1 simulation: $r_tot")




using Revise
using POMDPs
using SubHunt
using CommonRLInterface
using DeepQLearning
using Flux
using StaticArrays


exploration = EpsGreedyPolicy(SubHuntPOMDP(), LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))
solver = DeepQLearningSolver(qnetwork= Chain(Dense(8, 32, relu), Dense(32,32,relu), Dense(32, 6)),
                             max_steps=100_000, exploration_policy=exploration, verbose=true, prioritized_replay=false)
solve(solver, SubHuntPOMDP())

import SubHunt: Pos

Pos(1., 1.)

POMDPs.initialobs(::SubHuntPOMDP, ::SubState) = SubHunt.SubObsDist(SubHunt.active_beam(Pos(1., 1.)), Normal(3., 0.5), Normal(1., 0.5))




xs = [3. 1. 1.; 4. 1. 1.]
a = fit(MvNormal, xs)
rand(a, 10



(5, 5) in support(env.o)
