
module SimpleNN

using LinearAlgebra

export Neuron, one_step, one_iteration, train

struct Neuron
    w::Vector{Float64}
    b::Float64
end

sigmoid(z::Float64) = 1 / (1 + exp(-z))
loss(y::Float64, t::Float64) = 1 / 2 * (y - t)^2

# z = w1 * x1 + w2 * x2 + b == produit scalaire de w et x et on ajoute b
predict(neuron::Neuron, x::Vector{Float64}) = sigmoid(dot(neuron.w, x) + neuron.b)

function gradient(neuron::Neuron, data::Tuple{Vector{Float64},Float64})
    x, t = data
    y = predict(neuron, x)

    dE_dy = y - t
    dy_dz = y * (1 - y)

    dE_dz = dE_dy * dy_dz

    return Neuron(dE_dz .* x, dE_dz)
end

function one_step(
    neuron::Neuron,
    data::Tuple{Vector{Float64},Float64},
    learning_rate::Float64,
)

    grad = gradient(neuron, data)

    w = neuron.w - learning_rate .* grad.w
    b = neuron.b - learning_rate * grad.b

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

end
