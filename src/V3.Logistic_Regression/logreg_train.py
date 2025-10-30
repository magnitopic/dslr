import sys
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import json

from normalize import *
from imputation import *

# Train Variables
iterations = 2000
learning_rate = 0.01


# Logistic Regression Base Functions

r""" 
Sigmoid formula

$$
    f(z) = \frac{1}{1 + e^{-z}}
$$
"""


def sigmoid(z):
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


r""" 
Cost Function

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1-y^{(i)}) \log(1-h_{\theta}(x^{(i)})) \right]
$$
"""


def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    # Clip h to prevent log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


r""" 
Gradient descent

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$
"""


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []

    for _ in range(iterations):
        # 1. Compute error
        h = sigmoid(X @ theta)
        error = h - y

        # 2. Compute gradient
        gradient = (1 / m) * (X.T @ error)

        # 3. Update theta
        theta = theta - alpha * gradient

        # 4. Save cost
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

# 2. One vs All


def one_vs_all_train(X, y, num_labels, alpha, iterations):
    m, n = X.shape

    # Matrix to store the coefficients of all classifiers
    all_theta = np.zeros((num_labels, n + 1))

    X_b = np.c_[np.ones((m, 1)), X]

    for c in range(1, num_labels + 1):
        initial_theta = np.zeros(n + 1)

        # Create binary labels
        y_c = (y == c).astype(int)

        # Train with gradient descent
        theta_c, _ = gradient_descent(
            X_b, y_c, initial_theta, alpha, iterations)

        # Save trained parameters
        all_theta[c-1, :] = theta_c

    return all_theta


def main():
    try:
        # Load dataset
        data = pd.read_csv(sys.argv[1], index_col=0)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {sys.argv[1]}")
        return

    # Extract labels (Hogwarts House)
    y_labels = data['Hogwarts House'].copy()

    # Normalize Data
    fill_data_correlation(data)
    fields = data.select_dtypes(include=['float64']).columns
    fields = fields.drop('Astronomy')
    fields = fields.drop('History of Magic')
    fields = fields.drop('Arithmancy')
    fields = fields.drop('Potions')
    fields = fields.drop('Care of Magical Creatures')
    print(fields)

    new_data = data[fields]
    print(new_data)

    df_norm = new_data.apply(normalize, axis=0)
    print(df_norm)

    # Impute Data
    complete_rows = df_norm.dropna()
    incomplete_rows = df_norm[df_norm.isnull().any(axis=1)]
    print("Number of complete rows:", len(complete_rows))
    print("Number of incomplete rows:", len(incomplete_rows))

    df_norm_imputed = impute_missing_values(df_norm, k=5)

    print("Verify no missing values:")
    print(df_norm_imputed.isnull().sum())
    print("Dataset dimensions:", df_norm_imputed.shape)

    # Prepare data for training
    X = df_norm_imputed.values

    # Encode labels to numbers (1-4 for the 4 houses)
    house_mapping = {
        'Gryffindor': 1,
        'Hufflepuff': 2,
        'Ravenclaw': 3,
        'Slytherin': 4
    }
    y = y_labels.map(house_mapping).values
    num_labels = len(house_mapping)

    print(f"\nTraining with {num_labels} classes")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Train model One-Vs-All
    trained_theta = one_vs_all_train(
        X,
        y,
        num_labels,
        alpha=learning_rate,
        iterations=iterations
    )

    # Save parameters to Json
    params_to_save = {
        'num_labels': num_labels,
        'theta': trained_theta.tolist(),
        'feature_names': fields.tolist(),
        'house_mapping': house_mapping,
    }

    with open('trained_params.json', 'w') as f:
        json.dump(params_to_save, f, indent=4)

    print("Success: Parameters saved in 'trained_params.json'")
    print(
        f"Dimensions of saved theta (Classes x coefficients): {trained_theta.shape}")


if __name__ == "__main__":
    main()
