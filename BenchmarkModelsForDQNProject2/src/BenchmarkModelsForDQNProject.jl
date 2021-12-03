module BenchmarkModelsForDQNProject

using POMDPs
using Distributions
using StaticArrays
using POMDPModelTools

import Base: +, ==, hash

import POMDPs: discount, isterminal
import POMDPs: actions, actionindex
import POMDPs: transition
import POMDPs: initialstate
import POMDPs: reward
import POMDPs: convert_s


export
    Cont_Collection_HPH_MDP,
    StateCollection,
    Particle,
    Direction,
    get_coords,
    get_heaven,
    get_position,
    get_particles

include("continuous_collection_MDP.jl")

end
