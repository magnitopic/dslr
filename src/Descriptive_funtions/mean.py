from .count import count

def mean(values):
    if not values:
        raise ValueError("mean() expect at least 1 argument")
    
    len = count(values)
    sum = 0
    for val in values:
        sum += val

    return sum / len

if __name__ == "__main__":
    mean()