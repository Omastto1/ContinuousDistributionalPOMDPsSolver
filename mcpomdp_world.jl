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
# World is 21 X 21 Grid with Heaven location incorporated (1323 total states possible)
# Agent can go into 8 directions -  1 - North-East, 2 - East, 3 - South-East, 4 - South,
#                                   5 - South-West, 6 - West, 7 - North-West, 8 - North,
# Each Action is one long in max norm
# Each action can have approx 25% error - represented by normal distribution Normal(0, 0.25)
# Agent can obtain 3 observations that stand for the Heaven position
# 0 - Unknown, 1 - Lower-Left corner, 2 - Upper-Right corner
#
# world state is represented by Int according to following formula
# state = (Heaven position) * 21 ^ 2 + (y) * 21 + (x) + 1
# state = state_index

using POMDPs
using Distributions
using POMDPModelTools

struct State
    x::Int
    y::Int
    heaven::Int
    State(x, y, heaven) = new(max(min(20, x), 0), max(min(20, y), 0), heaven)
end

function State(coord::Tuple{Int64, Int64})
    State(coord[1], coord[2], 0)
end


function State(coord::Tuple{Int64, Int64, Int64})
    State(coord...)
end

function State(x, y)
    State(x, y, 0)
end


Base.broadcastable(ss::State) = Ref(ss)

import Base.:+
import Base: ==

+(ss::State, step::Tuple{Int, Int})::State = State(ss.x + step[1], ss.y + step[2])
==(ss1::State, ss2::State)::Bool = ss1.x == ss2.x && ss1.y == ss2.y && ss1.heaven == ss2.heaven
==(ss1::State, ss2::Tuple{Int, Int})::Bool = ss1.x == first(ss2) && ss1.y == last(ss2)


struct MCPOMDPWorld <: POMDP{State, Int, Int}
    heaven::Int
end

function MCPOMDPWorld()
    MCPOMDPWorld(rand(Categorical([0.5, 0.5])))
end




POMDPs.states(m::MCPOMDPWorld) = State.(Iterators.product(0:20, 0:20, 0:2)) # mozna bude potreba transponovat
POMDPs.actions(m::MCPOMDPWorld) = 1:8
POMDPs.observations(m::MCPOMDPWorld) = 1:4


POMDPs.stateindex(m::MCPOMDPWorld ,ss::State)::Int = ss.heaven * 441 + min(20, ss.y) * 21 + min(20, ss.x) + 1
POMDPs.actionindex(m::MCPOMDPWorld, a::Int)::Int = a
POMDPs.obsindex(m::MCPOMDPWorld, o::Int)::Int = o


POMDPs.initialstate(m::MCPOMDPWorld) = Deterministic(State(0, 0, 0))
# POMDPs.initialstate(m::MCPOMDPWorld) = Deterministic(1)
POMDPs.discount(m::MCPOMDPWorld)::Float64 = 0.95
POMDPs.isterminal(m::MCPOMDPWorld, ss::State)::Bool = ss == (20, 0) || ss == (0, 20)
function POMDPs.reward(m::MCPOMDPWorld, ss::State, a::Int, sp::State)::Float64
    !isterminal(m, sp) && return -1.
    sp.heaven == m.heaven && return 100
    if sp.heaven == 0
        (sp == (0, 20) && m.heaven == 1 || sp == (20, 0) && m.heaven == 2) && return 50
        return -500
    end
    return -5000
end


function POMDPs.transition(m::MCPOMDPWorld, ss::State, a::Int)::SparseCat
    values = (0.025, 0.025, 0.025, 0.025, 0.8, 0.025, 0.025, 0.025, 0.025)
    if a % 2 != 0
        if a < 4
            a == 1 && return SparseCat(ss .+ Iterators.product((1:3), -(1:3)), values) # NE
            return SparseCat(ss .+ Iterators.product((1:3), (1:3)), values) # SE
        else
            a == 7 && return SparseCat(ss .+ Iterators.product(-(1:3), -(1:3)), values) # NW
            return SparseCat(ss .+ Iterators.product(-(1:3), (1:3)), values) # SW
        end
    else
        if a < 5
            a == 2 && return SparseCat(ss .+
                ((1, -1), (2, -1), (3, -1), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)),
                values) # E
            return SparseCat(ss .+
                ((-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2), (-1, 3), (0, 3), (1, 3)),
                values) # S
        else
            a == 6 && return SparseCat(ss .+
                ((-3, -1), (-2, -1), (-1, -1),  (-3, 0), (-2, 0), (-1, 0), (-3, 1), (-2, 1), (-1, 1)),
                values) # W
            return SparseCat(ss .+
                ((-1, -3), (0, -3), (1, -3), (-1, -2), (0, -2), (1, -2), (-1, 1), (0, 1), (1, 1)),
                values) # N
        end
    end
end



# chybi observace doteku se zdi
function POMDPs.observation(m::MCPOMDPWorld, a::Int, sp::State)::Union{Deterministic, DiscreteUniform}
    sp == (20, 20) && return Deterministic(stateindex(m, State(20, 20, m.heaven)))
    sp.heaven == 0 && return DiscreteUniform(1, stateindex(m, State(20, 20, 0)))
    sp.heaven == 1 && return DiscreteUniform(stateindex(m, State(0, 0, 1)), stateindex(m, State(20, 20, 1)))
    sp.heaven == 2 && return DiscreteUniform(stateindex(m, State(0, 0, 2)), stateindex(m, State(20, 20, 2)))
end





using SARSOP


pomdp = MCPOMDPWorld()

solver = SARSOPSolver()
policy = solve(solver, pomdp)


using IncrementalPruning
solver = PruneSolver() # set the solver

policy = solve(solver, pomdp) # solve the POMDP


using QMDP

# initialize the solver
# key-word args are the maximum number of iterations the solver will run for, and the Bellman tolerance
solver = QMDPSolver(max_iterations=40,
                    belres=1e-5,
                    verbose=true
                   )

policy = solve(solver, pomdp)



using PointBasedValueIteration


solver = PBVISolver(verbose=true) # set the solver

policy = solve(solver, pomdp)



using POMDPSolve
solver = POMDPSolveSolver()
solve(solver, pomdp) # returns an AlphaVectorPolicy
