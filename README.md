# DSLR - Data Science and Logistic Regression

A Harry Potter-themed data science project implementing logistic regression from scratch for multi-class classification. This project is part of the 42 school curriculum and focuses on understanding the fundamentals of machine learning, data visualization, and statistical analysis.

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Analysis](#data-analysis)
  - [Data Visualization](#data-visualization)
  - [Model Training](#model-training)
  - [Making Predictions](#making-predictions)
  - [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [Theory](#theory)
- [Requirements](#requirements)
- [Author](#author)

##  Overview

This project implements a complete machine learning pipeline for classifying Hogwarts students into their respective houses (Gryffindor, Hufflepuff, Ravenclaw, or Slytherin) based on their academic performance across various magical subjects.

The implementation includes:
- Statistical analysis tools built from scratch (no pandas describe!)
- Data visualization for exploratory data analysis
- Logistic regression with One-vs-All (OvA) strategy for multi-class classification
- Model training, prediction, and evaluation

##  Features

### Data Analysis
- **Custom Statistical Functions**: Implementation of mean, standard deviation, percentiles, variance, median, and more without using built-in statistical libraries
- **Describe Tool**: Replicates pandas `describe()` functionality from scratch

### Data Visualization
- **Histogram**: Visualize feature distributions across all Hogwarts houses with house-themed colors
- **Scatter Plot**: Identify correlations between features using manual Pearson correlation calculation
- **Pair Plot**: Generate comprehensive pair plots to analyze relationships between all numerical features

### Machine Learning
- **Logistic Regression from Scratch**: Implementation of binary logistic regression with sigmoid activation
- **One-vs-All Classification**: Multi-class classification strategy for 4 Hogwarts houses
- **Gradient Descent Optimization**: Custom implementation with configurable learning rate and iterations
- **Cross-Validation**: K-fold cross-validation for robust model evaluation
- **Model Persistence**: Save and load trained model weights in JSON format

##  Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dslr_42
```

2. Install required dependencies:
```bash
pip install pandas numpy matplotlib scipy
```

3. Download the dataset:
```bash
bash init.sh
```

This will download and extract the required datasets into the `datasets/` directory.

##  Usage

### Data Analysis

Get statistical summary of the dataset (like pandas describe):

```bash
python describe.py datasets/dataset_train.csv
```

This displays count, mean, standard deviation, min, quartiles (25%, 50%, 75%), max, variance, and median for all numerical features.

### Data Visualization

#### Histogram
Visualize which Hogwarts course has the most homogeneous score distribution:

```bash
python histogram.py datasets/dataset_train.csv
```

#### Scatter Plot
Find the two features that are most similar (highest correlation):

```bash
python scatter_plot.py datasets/dataset_train.csv
```

#### Pair Plot
Generate a comprehensive pair plot showing relationships between all features:

```bash
python pair_plot.py datasets/dataset_train.csv
```

### Model Training

Train the logistic regression model using One-vs-All strategy:

```bash
python logreg_train.py datasets/dataset_train.csv
```

This will:
- Train 4 binary classifiers (one for each house)
- Save the model weights to `weights.json`
- Display training progress and final cost for each classifier

**Optional parameters:**
- Custom learning rate and iterations can be configured in the script

For bonus training with additional features:

```bash
python bonus_train.py datasets/dataset_train.csv
```

### Making Predictions

Use the trained model to predict house assignments:

```bash
python logreg_predict.py datasets/dataset_test.csv weights.json
```

This generates `houses.csv` containing predictions for all students in the test dataset.

### Model Evaluation

#### Accuracy Evaluation
Compare predictions against ground truth:

```bash
python evaluate_accuracy.py datasets/dataset_truth.csv houses.csv
```

#### Cross-Validation
Perform k-fold cross-validation on the training set:

```bash
python cross_validate.py datasets/dataset_train.csv
```

This provides a more robust evaluation by training and testing on multiple data splits.

##  Project Structure

```
dslr_42/
├── README.md                    # This file
├── init.sh                      # Script to download datasets
│
├── Data Analysis
│   ├── describe.py              # Statistical analysis tool
│   └── statistic.py             # Statistical utility functions
│
├── Data Visualization
│   ├── histogram.py             # Histogram visualization
│   ├── scatter_plot.py          # Correlation scatter plot
│   └── pair_plot.py             # Pair plot visualization
│
├── Machine Learning
│   ├── logreg_train.py          # Model training
│   ├── logreg_predict.py        # Model prediction
│   ├── bonus_train.py           # Bonus training implementation
│   ├── evaluate_accuracy.py     # Accuracy evaluation
│   └── cross_validate.py        # Cross-validation
│
├── datasets/
│   ├── dataset_train.csv        # Training data
│   ├── dataset_test.csv         # Test data (without labels)
│   └── dataset_truth.csv        # Test data ground truth
│
└── docs/
    └── theory.md                # Theoretical background on logistic regression
```

##  Theory

The project implements logistic regression, a supervised learning algorithm for classification tasks. Key concepts include:

### Logistic Regression
- Uses the **sigmoid function** to map linear combinations to probabilities: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Outputs probabilities between 0 and 1 for binary classification

### Multi-Class Classification
- **One-vs-All (OvA)** strategy: Train one binary classifier per class
- Each classifier predicts probability of belonging to its respective house
- Final prediction: class with highest probability

### Cost Function
- Binary cross-entropy loss: $J(\theta) = -\frac{1}{m}\sum[y\log(h_\theta(x)) + (1-y)\log(1-h_\theta(x))]$
- Convex function suitable for gradient descent optimization

### Gradient Descent
- Iteratively updates weights to minimize cost function
- Weight update rule: $\theta := \theta - \alpha\nabla J(\theta)$
- Learning rate ($\alpha$) controls step size

For detailed theoretical background, see [docs/theory.md](docs/theory.md).

##  Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scipy (for visualization only)
- sklearn (for evaluation only)

**Note**: The core logistic regression implementation does not use scikit-learn; it's built from scratch using only numpy for numerical operations.

##  Learning Objectives

This project demonstrates:
- Understanding of statistical measures and their implementation
- Data visualization techniques for exploratory analysis
- Mathematical foundations of logistic regression
- Implementation of gradient descent optimization
- Multi-class classification strategies
- Model evaluation and validation techniques
- Software engineering practices (modularity, code organization)

##  Bonus Features

- Advanced training implementation (`bonus_train.py`)
- Cross-validation for robust model evaluation
- Feature correlation analysis
- Comprehensive data visualization suite
- Statistical hypothesis testing for feature selection

##  Author

This project is part of the 42 school curriculum, focusing on practical implementation of data science and machine learning fundamentals.

---

**Note**: This implementation is for educational purposes, demonstrating understanding of machine learning algorithms by building them from scratch rather than relying on high-level libraries.
