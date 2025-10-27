from math import sqrt
from .count import count
from .mean import mean
from .std import std


""" 
Skew ≈ 0 (-0.5 to 0.5): Symmetric Distribution. The data is evenly balanced around the average (mean).

Negative Skew (< -0.5): Left Skewed 📉. 
The majority of data points are high, but a few extremely low values pull the mean down.

Positive Skew (> 0.5): Right Skewed 📈. 
The majority of data points are low, but a few extremely high values pull the mean up.
"""
def skew(values):
    n = count(values)
    if n < 3:
        return 0.0
    
    values_mean = mean(values)

    values_std = std(values)
    if values_std == 0:
        return 0.0
    
    sum_of_cubed_diffs = 0

    for val in values:
        sum_of_cubed_diffs += ((val - values_mean) / values_std) ** 3

    correction_factor = n / ((n - 1) * (n - 2))
    skewness = correction_factor * sum_of_cubed_diffs
    return skewness

if __name__ == "__main__":
    skew()