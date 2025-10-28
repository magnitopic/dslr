import pandas as pd
from Descriptive_funtions import *


def create_describe_file():
    data = pd.read_csv("./data/dataset_train.csv")

    # Select only numeric columns
    data_numeric = data.select_dtypes(include=['float64'])

    # Construct the final aggregation dictionary for ALL columns
    agg_list = [
        ('count', lambda x: count(x)),
        ('mean', lambda x: mean(x)),
        ('std', lambda x: std(x)),
        ('min', lambda x: min(x)),
        ('25%', lambda x: percentile(x, 25)),
        ('50%', lambda x: percentile(x, 50)),
        ('75%', lambda x: percentile(x, 75)),
        ('max', lambda x: max(x)),
        ('skew', lambda x: skew(x)),
        ('iqr', lambda x: iqr(x))
    ]

    results = {}
    for col in data_numeric.columns:
        # Drop NaN before converting to list for clean calc
        col_series = data_numeric[col].dropna()

        col_results = {name: func(col_series.tolist())
                       for name, func in agg_list}

        results[col] = col_results

    # Convert the dictionary of results to a DataFrame
    result_df = pd.DataFrame(results)

    print(result_df)


if __name__ == "__main__":
    create_describe_file()
