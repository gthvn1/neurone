include("neuron.jl")
using .SimpleNN

# Initial values
neuron = Neuron([0.5, 0.5], -0.5)
learn_rate = 0.01

# Expected results
data = [
    ([0.0, 0.0], 0.0), 
    ([0.0, 1.0], 1.0), 
    ([1.0, 0.0], 1.0), 
    ([1.0, 1.0], 0.0)]

neuron = train(neuron, data, learn_rate, 100_000)
println("[XOR]100.000 iterations : neuron $neuron")
