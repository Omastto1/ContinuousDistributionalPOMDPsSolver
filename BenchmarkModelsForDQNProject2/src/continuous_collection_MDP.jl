const Direction = SVector{2, Float32}

struct Particle
    x::Float32
    y::Float32
    heaven::Int
    Particle(x, y, heaven) = new(max(min(5, x), 0), max(min(5, y), 0), heaven)
end

get_coords(p::Particle) = (p.x, p.y)
get_heaven(p::Particle) = p.heaven

+(a::Particle, b::Direction) = Particle(a.x + b[1], a.y + b[2], a.heaven)
+(a::Tuple, b::Direction) = (first(a) + b[1], last(a) + b[2])
# +(a::Tuple, b::SVector{2, Float32}) = (first(a) + b[1], last(a) + b[2])
function +(a::Particle, b::SMatrix{2, 3, Float32})
    SA[Particle(a.x + b[1, 1], a.y + b[2, 1], a.heaven),
             Particle(a.x + b[1, 2], a.y + b[2, 2], a.heaven),
             Particle(a.x + b[1, 3], a.y + b[2, 3], a.heaven)]
end

==(ss1::Particle, ss2::Tuple{Float32, Float32})::Bool = ss1.x == first(ss2) && ss1.y == last(ss2)
==(ss1::Particle, ss2::Particle)::Bool = ss1.x == ss2.x && ss1.y == ss2.y && ss1.heaven == ss2.heaven


struct StateCollection
    real_position::Particle
    particles::Array{Particle}
end

get_position(s::StateCollection) = s.real_position
get_particles(s::StateCollection) = s.particles


struct Cont_Collection_HPH_MDP <: MDP{StateCollection, Direction}
    heaven::Int
    step_size::Float32
end

function Cont_Collection_HPH_MDP()
    Cont_Collection_HPH_MDP(rand(Categorical([0.5f0, 0.5f0])), 1.0f0)
end

function Cont_Collection_HPH_MDP(;step_size::Float32)
    Cont_Collection_HPH_MDP(rand(Categorical([0.5f0, 0.5f0])), step_size)
end


function initialstate(m::Cont_Collection_HPH_MDP; random_start::Bool=false)
    !random_start && return Deterministic(StateCollection(Particle(0.0f0, 0.0f0, 0.0f0), Particle[Particle(0.0f0, 0.0f0, 0.0f0)]))
    random_state_coords = round.(rand(Distributions.Uniform(0.0f0, 5.0f0), 2), digits = 1)
    return Deterministic(StateCollection(Particle(random_state_coords..., 0),
                            Particle[Particle(random_state_coords..., 0)]))
end

near_priest(s::StateCollection) = near_priest(get_position(s))
near_priest(p::Particle) = p == Particle(5., 5., 0)

function transition(m::Cont_Collection_HPH_MDP, s::StateCollection, a::Direction)
    coords = get_coords(get_position(s)) + a

    if near_priest(get_position(s) + a)
        pos = Particle((get_coords(get_position(s)) + a)..., m.heaven)
        sp = StateCollection(pos, Particle[pos])
    else
        sp = StateCollection(get_position(s) + a, Particle[])
        ps = get_particles(s)
        for particle in rand(ps, min(3, length(ps)))
            append!(sp.particles, transition(m, particle, a))
        end
    end

    # UNCOMMENT `return sp` with DeepQLearning.jl
    # UNCOMMENT `return Deterministic(sp)` with POMDPTutorials.jl
    # return sp
    return Deterministic(sp)
end

### Draw random get_position (or more) from some distribution centered on s + a
function transition(m::Cont_Collection_HPH_MDP, s::Particle,
                            a::Direction)::AbstractArray{Particle}
    n_particles = rand(3:8)
    return s + a + SMatrix{2, n_particles, Float32}(rand(
        Distributions.Uniform(-m.step_size*0.25f0, m.step_size*0.25f0), (2, n_particles)))
end

function reward(m::Cont_Collection_HPH_MDP, s::StateCollection, a::Direction,
                            sp::StateCollection)
    if isterminal(m, sp)
        get_heaven(get_position(sp)) == m.heaven && return 1.0f0
        return -1.0f0
    else
        return -0.001f0
    end
end

function isterminal(m::Cont_Collection_HPH_MDP, s::StateCollection)
    get_position(s) == (5.0f0, 0.0f0) || get_position(s) == (0.0f0, 5.0f0)
end

function convert_s(::Type{A}, s::StateCollection,
                    mdp::Cont_Collection_HPH_MDP) where A<:AbstractArray
    hcat(convert_s.(Array, s.particles, mdp)...)
end
function convert_s(::Type{A}, s::Particle,
                    mdp::Cont_Collection_HPH_MDP) where A<:AbstractArray
    Float32[s.x, s.y, s.heaven]
end
function convert_s(::Type{StateCollection}, s::A,
                    mdp::Cont_Collection_HPH_MDP) where A<:AbstractArray
    StateCollection(Particle(mdp.step_size * round.(mean(s, dims=2)/mdp.step_size)...),
                    [Particle(s[:,c]...) for c in 1:size(s,2)])
end

function convert_s(::Type{StateCollection}, s::SVector{3, Float64},
                    mdp::Cont_Collection_HPH_MDP)
    StateCollection(Particle(mdp.step_size * round.(s/mdp.step_size)...),
                    [Particle(s...)])
end





function actions(m::Cont_Collection_HPH_MDP)
    SA[Direction(m.step_size, m.step_size), Direction(0., m.step_size),
        Direction(m.step_size, 0.), Direction(m.step_size, -m.step_size),
        Direction(-m.step_size, -m.step_size), Direction(0., -m.step_size),
        Direction(-m.step_size, 0.), Direction(-m.step_size, m.step_size)]
end

function actionindex(m::Cont_Collection_HPH_MDP, a::Direction)
    return Dict(Direction(m.step_size, m.step_size) => 1,
                Direction(0., m.step_size) => 2,
                Direction(m.step_size, 0.) => 3,
                Direction(m.step_size, -m.step_size) => 4,
                Direction(-m.step_size, -m.step_size) => 5,
                Direction(0., -m.step_size) => 6,
                Direction(-m.step_size, 0.) => 7,
                Direction(-m.step_size, m.step_size) => 8)[a]
end

discount(m::Cont_Collection_HPH_MDP) = 0.99

Base.broadcastable(m::Cont_Collection_HPH_MDP) = Ref(m)
Base.broadcastable(s::Particle) = Ref(s)
