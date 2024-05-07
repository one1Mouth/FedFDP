def weighted_variance(data, weights):
    if len(data) != len(weights):
        raise ValueError("The lengths of data and weights must be the same.")

    weighted_mean = sum(data[i] * weights[i] for i in range(len(data))) / sum(weights)
    weighted_squared_diff = sum(weights[i] * (data[i] - weighted_mean) ** 2 for i in range(len(data)))
    weighted_variance = weighted_squared_diff / sum(weights)

    return weighted_variance
