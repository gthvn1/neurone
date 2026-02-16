function sigmoid(z)
    return 1 / (1 + exp(-z))
end

function loss(y, t)
    1 / 2 * (y - t)^2
end

function gradient(w1, w2, b, x1, x2, t)
    z = w1 * x1 + w2 * x2 + b
    y = sigmoid(z)

    dE_dy = y - t
    dy_dz = y * (1 - y)

    dz_dw1 = x1
    dz_dw2 = x2
    dz_db = 1

    dE_dw1 = dE_dy * dy_dz * dz_dw1
    dE_dw2 = dE_dy * dy_dz * dz_dw2
    dE_db = dE_dy * dy_dz * dz_db

    dE_dw1, dE_dw2, dE_db
end

function one_step(w1, w2, b, data, learning_rate)
    x1, x2, t = data

    dE_dw1, dE_dw2, dE_db = gradient(w1, w2, b, x1, x2, t)

    w1 -= learning_rate * dE_dw1
    w2 -= learning_rate * dE_dw2
    b -= learning_rate * dE_db

    w1, w2, b
end

function one_iteration(w1, w2, b, data, learning_rate)
    for d in data
        w1, w2, b = one_step(w1, w2, b, d, learning_rate)
    end
    return w1, w2, b
end

function train(w1, w2, b, data, learning_rate, times)
    while times > 0
        w1, w2, b = one_iteration(w1, w2, b, data, learning_rate)
        times -= 1
    end

    return w1, w2, b
end

# Initial values
w1 = 0.5
w2 = 0.5
b = -0.5
n = 0.0001

# Expected results
data = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]

# Let's do a single iteration
println("Before first step: w1 = $w1, w2 = $w2, b = $b")

w1, w2, b = one_step(w1, w2, b, data[1], n)
println("After first step : w1 = $w1, w2 = $w2, b = $b")

w1, w2, b = one_step(w1, w2, b, data[2], n)
println("After second step: w1 = $w1, w2 = $w2, b = $b")

w1, w2, b = one_step(w1, w2, b, data[3], n)
println("After third step : w1 = $w1, w2 = $w2, b = $b")

w1, w2, b = one_step(w1, w2, b, data[4], n)
println("After fourth step: w1 = $w1, w2 = $w2, b = $b")

println("One iteration done")

w1, w2, b = one_iteration(w1, w2, b, data, n)
println(" after step : w1 = $w1, w2 = $w2, b = $b")

println("Two iteration done")
