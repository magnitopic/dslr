from .count import count
from math import ceil

def percentile(values, percentile):
    values.sort()

    range = percentile * count(values) / 100 - 1
    
    return range

if __name__ == "__main__":
    percentile()