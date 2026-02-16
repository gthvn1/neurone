include("neuron.jl")
using .SimpleNN

# Initial values
neuron = Neuron([0.5, 0.5], -0.5)
learn_rate = 0.01

# Expected results
data = [([0.0, 0.0], 0.0), ([0.0, 1.0], 0.0), ([1.0, 0.0], 0.0), ([1.0, 1.0], 1.0)]

# Let's do a single iteration
println("Before first step: neuron $neuron")

neuron = one_step(neuron, data[1], learn_rate)
println("After first step : neuron $neuron")

neuron = one_step(neuron, data[2], learn_rate)
println("After second step: neuron $neuron")

neuron = one_step(neuron, data[3], learn_rate)
println("After third step : neuron $neuron")

neuron = one_step(neuron, data[4], learn_rate)
println("After fourth step: neuron $neuron")

println("One iteration done")

neuron = one_iteration(neuron, data, learn_rate)
println("Two iterations : neuron $neuron")

neuron = train(neuron, data, learn_rate, 100_000)
println("100.000 iterations : neuron $neuron")
