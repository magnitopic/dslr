def max(values):
    if not values:
        raise ValueError("max() expect at least 1 argument")
    
    max_value = values[0]
    for val in values:
        if val > max_value:
            max_value = val

    return float(max_value)

if __name__ == "__main__":
    max()