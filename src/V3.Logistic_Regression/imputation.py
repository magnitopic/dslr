import pandas as pd
import numpy as np

def euclidean_distance(row1, row2, columns):
    return np.sqrt(sum((row1[columns] - row2[columns])**2))

def get_k_neighbors(target_row, complete_data, columns_to_use, k=5):
    distances = []
    for idx, row in complete_data.iterrows():
        dist = euclidean_distance(target_row, row, columns_to_use)
        distances.append((idx, dist))

    # Sort by distance and get closest neighbors
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances[:k]]

def impute_missing_values(df, k=5):
    df_imputed = df.copy()
    complete_data = df.dropna()

    # For each row of missing values
    for idx, row in df[df.isnull().any(axis=1)].iterrows():
        # Find missing values in row
        missing_cols = row[row.isnull()].index

        # Find available columns to calculate distance
        available_cols = row[row.notnull()].index

        # Find closest k neighbours
        neighbors_idx = get_k_neighbors(row, complete_data, available_cols, k)

        # Impute each missing row with avg of neighbours
        for col in missing_cols:
            neighbor_values = complete_data.loc[neighbors_idx, col]
            df_imputed.loc[idx, col] = neighbor_values.mean()

    return df_imputed