import pandas as pd
import numpy as np
import sys

def calculate_accuracy(y_pred, y_real):
    correct_predictions = (y_pred == y_real)

    accuracy = np.mean(correct_predictions) * 100
    
    return accuracy

def main():
    try:
        # Load dataset
        data_pred = pd.read_csv(sys.argv[1], index_col=0)
        data_real = pd.read_csv(sys.argv[2], index_col=0)

        # Select column to compare
        y_pred = data_pred['Hogwarts House']
        y_real = data_real['Hogwarts House']

        if len(y_pred) != len(y_real):
            min_len = min(len(y_pred), len(y_real))

            # Cut both DataFrames to the shortest one
            y_pred = y_pred.iloc[:min_len]
            y_real = y_real.iloc[:min_len]


        accuracy = calculate_accuracy(y_pred, y_real)
        print(f"Accuracy: {accuracy:.4}%")

    except FileNotFoundError:
        print(f"Error: CSV file not found: {sys.argv[1]}")
        return False
    except IndexError:
        print("Error: Please, add 2 files as arguments")
        return False
    
    return True

if __name__ == "__main__":
    main()