# Machine Learning Concepts Explained (Bootcamp Notebooks)

## What is Machine Learning?
Machine Learning (ML) is a field of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed. ML algorithms build mathematical models based on sample data ("training data") to make predictions or decisions without being programmed to do so. ML is widely used in applications such as image recognition, natural language processing, recommendation systems, and more.

### Types of Machine Learning
- **Supervised Learning:** The algorithm learns from labeled data and makes predictions based on that data. Tasks include regression and classification.
- **Unsupervised Learning:** The algorithm finds patterns and relationships in unlabeled data. Tasks include clustering and dimensionality reduction.
- **Reinforcement Learning:** The algorithm learns by interacting with its environment and receiving feedback (rewards or penalties).

---

## Regression (Linear and Non-Linear)
Regression is a supervised learning technique used to predict continuous outcomes.

### Linear Regression
- **Concept:** Models the relationship between a dependent variable and one or more independent variables using a straight line.
- **Equation:** $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$
- **Notebook Example:**
  - Loads data (e.g., `tamil_cinema_dataset.csv`)
  - Uses `LinearRegression` from scikit-learn
  - Splits data into training and test sets
  - Evaluates with metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ score
  - Visualizes actual vs predicted values

### Non-Linear Regression
- **Concept:** Models the relationship between variables using curves or more complex functions (e.g., polynomial regression, decision trees, etc.).
- **Notebook Example:**
  - May use regression trees or other non-linear models for complex relationships
  - Evaluates with similar metrics as linear regression

---

## Classification
Classification is a supervised learning technique used to predict categorical outcomes (classes).

### Logistic Regression
- **Concept:** Used for binary classification; models the probability that an instance belongs to a particular class using the logistic (sigmoid) function.
- **Notebook Example:**
  - Loads data (e.g., `logistic_regression_dataset.csv`)
  - Splits data, scales features
  - Trains a `LogisticRegression` model
  - Evaluates with accuracy, confusion matrix, ROC curve, and classification report

### Decision Tree
- **Concept:** Splits data into branches based on feature values, creating a tree structure for decision making.
- **Notebook Example:**
  - Loads data (e.g., `dt_data.csv`)
  - Trains a `DecisionTreeClassifier`
  - Evaluates with accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix

### Random Forest
- **Concept:** An ensemble of decision trees; improves accuracy and reduces overfitting by averaging multiple trees.
- **Notebook Example:**
  - Loads data (e.g., `rf_data.csv`)
  - Trains a `RandomForestClassifier`
  - Evaluates with standard classification metrics

### K-Nearest Neighbors (KNN)
- **Concept:** Classifies data points based on the majority class among their k-nearest neighbors in feature space.
- **Notebook Example:**
  - Loads data (e.g., `knn_data.csv`)
  - Scales features
  - Trains a `KNeighborsClassifier`
  - Evaluates with accuracy, precision, recall, F1-score, ROC-AUC

### Naive Bayes
- **Concept:** Probabilistic classifier based on Bayes' theorem; assumes independence between features.
- **Notebook Example:**
  - Loads data (e.g., `nb_data.csv`)
  - Trains a `GaussianNB` model
  - Evaluates with standard metrics

### Support Vector Machine (SVM)
- **Concept:** Finds the optimal hyperplane that separates classes with the maximum margin; effective for high-dimensional data.
- **Notebook Example:**
  - Loads data (e.g., `svm_data.csv`)
  - Scales features
  - Trains an `SVC` model
  - Evaluates with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix

---

## General Workflow in the Notebooks
1. **Data Loading:** Using pandas to read CSV files from the datasets folder.
2. **Exploratory Data Analysis (EDA):** Checking data shape, summary statistics, and class distribution.
3. **Data Preprocessing:** Cleaning, scaling, and splitting data into train/test sets.
4. **Model Training:** Using scikit-learn models and pipelines for training.
5. **Model Evaluation:** Using metrics like accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and regression metrics.
6. **Visualization:** Plotting results (e.g., ROC curve, actual vs predicted values).
7. **Model Saving (Optional):** Saving trained models for future use.

---

## Summary Table
| Notebook                | Task           | Model(s) Used           | Key Concepts                |
|-------------------------|----------------|-------------------------|-----------------------------|
| bootcamp.ipynb          | Regression     | LinearRegression        | Linear regression, metrics  |
| cinema.ipynb            | Regression     | LinearRegression        | Feature engineering, metrics|
| logistic_regression.ipynb| Classification| LogisticRegression      | Sigmoid, ROC, confusion matrix|
| Decision_Tree_tutorial.ipynb| Classification| DecisionTreeClassifier | Tree structure, metrics     |
| Random_Forest_tutorial.ipynb| Classification| RandomForestClassifier | Ensemble, metrics           |
| KNN_tutorial.ipynb      | Classification | KNeighborsClassifier    | Distance, scaling, metrics  |
| Naive_Bayes_tutorial.ipynb| Classification| GaussianNB             | Probabilistic, Bayes theorem|
| SVM_tutorial.ipynb      | Classification | SVC                     | Hyperplane, margin, scaling |
| insurance.ipynb         | Data Analysis  | Pandas                  | Cleaning, EDA               |

---

This markdown file provides a comprehensive overview of the machine learning concepts and models used in your bootcamp notebooks, with detailed explanations and workflow steps for both regression and classification tasks.