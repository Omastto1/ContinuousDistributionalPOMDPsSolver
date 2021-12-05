# ContinuousDistributionalPOMDPsSolver
Unfinished project that combines DeepQLearning.jl with Mill.jl in order to solve POMDPs with nonparametric distributional states (variable number of input particles creating distribution of state).

As of summer 2021, I have not found any POMDP solver, that would solve non-parametric distributional states (variable number of input particle creating distribution - unfortunatelly I am not aware of correct name).

This project aimed to combine SoTA Mill.jl package using which we would encode set of variable number of particles into finite size state aggregation. With this aggregation state, DeepQLearning.jl would be able to solve problems, where the size of the set of possible particles varies. 

Main reasons this project did not end up finished.
  - Instead of dirty workspace with multiple projects where I made changes I should have proposed changes to original repositories in such a way that their methods are more abstract and the repositories can accept my inputs
  - Better benchmark architecture 
    - Due to my goal to use and compare multiple JuliaPOMDPs.jl solvers I ended up changing the benchmark example more often than I should have resulting in hours of repeated precompiling of huge Julia libraries.
  - Differences in POMDPs - The goal of comparing multiple JuliaPOMDPs.jl solvers was not achieved as I did not manage to run simple benchmark example on multiple solvers as the continuous POMDPs create trees, and thus need single states. However proposed task required set of particles. Both solvers use same API, but expected returns differ. I was not able to come up with a solution to this.

Results 
  - Unfortunattely, the project did not present any comparisson of solvers.
  - However, I managed to alter the packages in such a way, that this project is able to run POMDPs whose states are encoded with sets of variable size of particles.
