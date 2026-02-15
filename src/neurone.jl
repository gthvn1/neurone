function sigmoid(z)
    return 1 / (1 + exp(-z))
end

function loss(y, t)
    1/2 * (y - t)^2
end

function gradient(w1, w2, b, x1, x2, t)
    z = w1*x1 + w2 * x2 + b
    y = sigmoid(z)

    dE_dy = y - t
    dy_dz = y * (1 - y)
    dz_dw1 = x1
    dz_dw2 = x2
    dz_db = b

    dE_dw1 = dE_dy * dy_dz * dz_dw1
    dE_dw2 = dE_dy * dy_dz * dz_dw2
    dE_db = dE_dy * dy_dz * dz_db

    dE_dw1, dE_dw2, dE_db
end
