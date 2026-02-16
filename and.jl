using LinearAlgebra

function sigmoid(z::Float64)
    return 1 / (1 + exp(-z))
end

function loss(y::Float64, t::Float64)
    1 / 2 * (y - t)^2
end

# z = w1 * x1 + w2 * x2 + b
#   On fait le produit scalaire de w et x et on ajoute b
function predict(w::Vector{Float64}, b::Float64, x::Vector{Float64})
    sigmoid(dot(w, x) + b)
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
    w::Vector{Float64},
    b::Float64,
    data::Tuple{Vector{Float64},Float64},
    learning_rate::Float64,
)

    dE_dw, dE_db = gradient(w, b, data)

    w -= learning_rate .* dE_dw
    b -= learning_rate * dE_db

    return w, b
end

function one_iteration(
    w::Vector{Float64},
    b::Float64,
    data::Vector{Tuple{Vector{Float64},Float64}},
    learning_rate::Float64,
)
    for d in data
        w, b = one_step(w, b, d, learning_rate)
    end
    return w, b
end

function train(
    w::Vector{Float64},
    b::Float64,
    data::Vector{Tuple{Vector{Float64},Float64}},
    learning_rate::Float64,
    times::Int64,
)
    while times > 0
        w, b = one_iteration(w, b, data, learning_rate)
        times = times - 1
    end

    return w, b
end

# Initial values
w = [0.5, 0.5]
b = -0.5
n = 0.0001

# Expected results
data = [([0.0, 0.0], 0.0), ([0.0, 1.0], 0.0), ([1.0, 0.0], 0.0), ([1.0, 1.0], 1.0)]

# Let's do a single iteration
println("Before first step: w = $w, b = $b")

w, b = one_step(w, b, data[1], n)
println("After first step : w = $w b = $b")

w, b = one_step(w, b, data[2], n)
println("After second step: w = $w, b = $b")

w, b = one_step(w, b, data[3], n)
println("After third step : w = $w, b = $b")

w, b = one_step(w, b, data[4], n)
println("After fourth step: w = $w, b = $b")

println("One iteration done")

w, b = one_iteration(w, b, data, n)
println(" after step : w = $w, b = $b")

println("Two iteration done")
