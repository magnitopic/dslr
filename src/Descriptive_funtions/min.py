def min(values):
    if not values:
        raise ValueError("min() expect at least 1 argument")

    min_value = values[0]
    for val in values:
        if val < min_value:
            min_value = val

    return float(min_value)

if __name__ == "__main__":
    min()