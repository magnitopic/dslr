# DSLR - Data Science Logistic Regression

## 📖 Description

**DSLR** is a Machine Learning project focused on multiclass classification using **Logistic Regression** implemented from scratch without using ML libraries (except for validation). The goal is to predict which Hogwarts house a student belongs to based on their grades in various magical subjects.

This project is inspired by Kaggle's famous Titanic challenge, but set in the Harry Potter universe, making it an educational and entertaining exercise to learn the fundamentals of data analysis and supervised learning.

## 🎯 Project Objectives

1. **Exploratory Data Analysis (EDA)**: Implement descriptive statistical functions from scratch
2. **Data Visualization**: Create meaningful plots to understand patterns and correlations
3. **Preprocessing**: Normalization and imputation of missing values using KNN
4. **Classification Model**: Implement multiclass logistic regression (One-vs-All)
5. **Optimization**: Implement gradient descent variants (Batch, Mini-batch, SGD)

## ✨ Features

### 1. Descriptive Statistical Analysis (`describe.py`)
From-scratch implementation of statistical functions without specialized libraries:
- Count, Mean, Standard Deviation
- Min, Max, Percentiles (25%, 50%, 75%)
- Skewness
- IQR (Interquartile Range)

### 2. Data Visualization
- **Histograms**: Grade distribution analysis by house
- **Scatter Plots**: Identification of correlations between subjects
- **Pair Plots**: Multidimensional visualization of variable relationships
- **Heatmap**: Correlation matrix between all subjects

### 3. Data Preprocessing
- **Normalization**: Z-score standardization to scale features
- **KNN Imputation**: Filling missing values using K-nearest neighbors
- **Feature Selection**: Selection of most relevant features for classification

### 4. Logistic Regression Model

#### Standard Implementation (Batch Gradient Descent)
- One-vs-All for multiclass classification (4 houses)
- 2000 iterations with learning rate of 0.1
- Validation with train/validation split
- Learning curve visualization

#### Bonus Implementations
- **Mini-batch Gradient Descent**: Training with batches of 32 samples for 100 epochs
- **Stochastic Gradient Descent (SGD)**: Weight update sample by sample (50 iterations)

### 5. Prediction and Evaluation
- House prediction on test dataset
- Accuracy calculation by comparing with ground truth
- CSV file generation with predictions

## 📁 Project Structure

```
dslr/
├── data/                          # Datasets
│   ├── dataset_train.csv          # Training data
│   └── dataset_test.csv           # Test data
├── output/                        # Generated visualizations
│   ├── heatmap.png
│   ├── histogram.png
│   ├── scatter_plot.png
│   ├── scatter_plot_2.png
│   └── pair_plot.png
├── src/
│   ├── V1.Data_Analysis/          # Descriptive analysis
│   │   ├── describe.py            # Descriptive statistics
│   │   └── Descriptive_funtions/  # Statistical functions implementation
│   ├── V2.Data_Visualization/     # Visualization notebooks
│   │   ├── histogram.ipynb
│   │   ├── pair_plot.ipynb
│   │   └── scatter_plot.ipynb
│   ├── V3.Logistic_Regression/    # Main model
│   │   ├── logreg_train.py        # Model training
│   │   ├── logreg_predict.py      # Prediction
│   │   ├── normalize.py           # Data normalization
│   │   └── imputation.py          # KNN imputation
│   └── Bonus/                     # Additional implementations
│       ├── logreg_train_minibatch.py  # Mini-batch GD
│       ├── logreg_train_SGD.py        # Stochastic GD
│       └── accuracy.py                # Accuracy calculation
├── Makefile                       # Automation commands
├── requirements.txt               # Dependencies
└── README.md
```

## 🚀 Installation

### 1. Create virtual environment and install dependencies

```bash
# Option 1: Using Make
make all

# Option 2: Manual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Activate virtual environment

```bash
source venv/bin/activate
```

## 📊 Usage

### Descriptive Statistical Analysis

```bash
make describe
```

Generates descriptive statistics for all numeric subjects in the dataset.

### Data Visualization

```bash
make jupyter
```

Opens JupyterLab to explore interactive visualization notebooks.

## 📈 Data Exploration

### Heatmap

Correlation matrix between all magical subjects. Helps identify linear relationships between variables.

![Heatmap](output/heatmap.png)

### Histogram

Distribution of grades by Hogwarts house. Allows identification of which subjects are most distinctive for each house.

![Histogram](output/histogram.png)

### Scatter Plot

**Strongest correlation between subjects:**

![Scatter plot](output/scatter_plot.png)

**Second strongest correlation:**

![Scatter plot](output/scatter_plot_2.png)

### Pair Plot

Multidimensional visualization showing all combinations of variable pairs colored by house.

![Pair plot](output/pair_plot.png)

### Model Training

#### Standard training (Batch Gradient Descent)
```bash
make train
```

Trains the model and saves weights to `trained_params.json`. Displays cost evolution graphs during training.

<img width="800" height="600" alt="dslr-cost-plot" src="https://github.com/user-attachments/assets/210e2ed9-7e05-48df-8987-0105dd4beb3d" />


#### Training with Mini-batch
```bash
make train-minibatch
```

#### Training with SGD
```bash
make train-sgd
```

### Prediction

```bash
make predict
```

Generates predictions for test dataset and saves them to `houses.csv`.

<img width="672" height="493" alt="dslr-predict" src="https://github.com/user-attachments/assets/fe97765b-900f-489f-a196-5958a69e2181" />

### Accuracy Evaluation

```bash
make accuracy
```

Calculates model accuracy by comparing predictions with training dataset.

## 🧮 Mathematical Foundations

### Sigmoid Function

$$
f(z) = \frac{1}{1 + e^{-z}}
$$

### Cost Function (Cross-Entropy)

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1-y^{(i)}) \log(1-h_{\theta}(x^{(i)})) \right]
$$

### Gradient Descent

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

## 🛠️ Technologies Used

- **Python 3.x**: Main language
- **NumPy**: Matrix and numerical operations
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **JupyterLab**: Interactive analysis
- **scikit-learn**: Only for results validation

## 📝 Technical Notes

- **One-vs-All Strategy**: To convert the multiclass problem (4 houses) into 4 binary problems
- **Feature Selection**: Some subjects with low correlation are excluded to improve the model
- **Train/Validation Split**: Dataset is split for cross-validation during training
- **Regularization**: Value clipping to prevent overflow in sigmoid and log

## 🎓 Key Learnings

1. Implementation of ML algorithms from scratch without specialized libraries
2. Importance of exploratory data analysis before modeling
3. Preprocessing techniques: normalization and imputation
4. Different optimization strategies (Batch, Mini-batch, SGD)
5. Handling multiclass classification with One-vs-All strategy
6. Effective visualization for results interpretation

# Team work 💪

This project was a team effort. You can checkout the team members here:

-   **Alejandro Díaz Ufano Pérez**
    -   [Github](https://github.com/adiaz-uf)
    -   [LinkedIn](https://www.linkedin.com/in/alejandro-d%C3%ADaz-35a996303/)
    -   [42 intra](https://profile.intra.42.fr/users/adiaz-uf)
-   **Alejandro Aparicio**
    -   [Github](https://github.com/magnitopic)
    -   [LinkedIn](https://www.linkedin.com/in/magnitopic/)
    -   [42 intra](https://profile.intra.42.fr/users/alaparic)

## 📄 License

This project is open source and available for educational purposes.

