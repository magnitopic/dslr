from math import sqrt
from .count import count
from .mean import mean

def std(values):
    values_mean = mean(values)
    squares_sum = 0
    len = count(values)

    for val in values:
        squares_sum += (val - values_mean) * (val - values_mean)

    variance = squares_sum / (len - 1)
    std = sqrt(variance)

    return std


if __name__ == "__main__":
    std()
