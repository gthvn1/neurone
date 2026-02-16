using LinearAlgebra

struct Neuron
    w::Vector{Float64}
    b::Float64
end

function sigmoid(z::Float64)
    return 1 / (1 + exp(-z))
end

function loss(y::Float64, t::Float64)
    1 / 2 * (y - t)^2
end

# z = w1 * x1 + w2 * x2 + b
#   On fait le produit scalaire de w et x et on ajoute b
function predict(neuron::Neuron, x::Vector{Float64})
    sigmoid(dot(neuron.w, x) + neuron.b)
end

function gradient(w::Vector{Float64}, b::Float64, data::Tuple{Vector{Float64},Float64})
    x, t = data
    y = predict(w, b, x)

    dE_dy = y - t
    dy_dz = y * (1 - y)

    dE_dz = dE_dy * dy_dz

    return dE_dz .* x, dE_dz
end

function one_step(
    neuron::Neuron,
    data::Tuple{Vector{Float64},Float64},
    learning_rate::Float64,
)

    dE_dw, dE_db = gradient(neuron.w, neuron.b, data)

    w = neuron.w - learning_rate .* dE_dw
    b = neuron.b - learning_rate * dE_db

    return Neuron(w, b)
end

function one_iteration(
    neuron::Neuron,
    data::Vector{Tuple{Vector{Float64},Float64}},
    learning_rate::Float64,
)
    for d in data
        neuron = one_step(neuron, d, learning_rate)
    end
    return neuron
end

function train(
    neuron::Neuron,
    data::Vector{Tuple{Vector{Float64},Float64}},
    learning_rate::Float64,
    times::Int,
)
    for _ = 1:times
        neuron = one_iteration(neuron, data, learning_rate)
    end

    return neuron
end

# Initial values
neuron = Neuron([0.5, 0.5], -0.5)
learn_rate = 0.0001

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
println(" after step : neuron $neuron")

println("Two iteration done")
