import pandas as pd
import numpy as np

def fill_data_correlation(df):
    print(f"Values in Defense Against the Dark Arts Before:\n{df['Defense Against the Dark Arts'].count()}/{len(df)}")

    mask = (df['Defense Against the Dark Arts'].isna() & 
            df['Astronomy'].notna())

    df.loc[mask, 'Defense Against the Dark Arts'] = df.loc[mask, 'Astronomy'] * -0.01

    print(f"Values in Defense Against the Dark Arts after:\n{df['Defense Against the Dark Arts'].count()}/{len(df)}")


def normalize(column):
    mean = column.mean()
    std = column.std()
    return (column - mean) / std


