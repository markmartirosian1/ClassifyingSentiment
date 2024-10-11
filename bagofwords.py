import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
x_train_df = pd.read_csv('data_reviews/x_train.csv')
y_train = pd.read_csv('data_reviews/y_train.csv')

# Extract the review text and the labels
X = x_train_df['text'].values  # text data
y = y_train['is_positive_sentiment'].values  # labels

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: TF-IDF Vectorizer
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))

# Logistic Regression Classifier
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Create a pipeline for preprocessing and classification
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', log_reg)
])

# Set up hyperparameter grid for Logistic Regression
param_grid = {
    'classifier__C': [1.00000000e-06, 5.17947468e-06, 2.68269580e-05, 1.38949549e-04,
                      7.19685673e-04, 3.72759372e-03, 1.93069773e-02, 1.00000000e-01,
                      5.17947468e-01, 2.68269580e+00, 1.38949549e+01, 7.19685673e+01,
                      3.72759372e+02, 1.93069773e+03, 1.00000000e+04, 5.17947468e+04,
                      2.68269580e+05, 1.38949549e+06, 7.19685673e+06, 1.00000000e+06],
    'classifier__penalty': ['l2'],
}

# Cross-validation with Grid Search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='roc_auc')

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters from grid search
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the model on validation set using AUROC
y_val_pred_proba = grid_search.predict_proba(X_val)[:, 1]  # Get the predicted probabilities
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUROC: {roc_auc}")

# Cross-validation scores for the best model (10 folds)
cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=10, scoring='roc_auc')
print(f"Cross-validation AUROC scores: {cv_scores}")
print(f"Mean AUROC: {cv_scores.mean()}")

# ------------------------ Figure 1: Hyperparameter Search Results ------------------------

# Get the mean test scores for different C values from the grid search results
mean_test_scores = grid_search.cv_results_['mean_test_score']
std_test_scores = grid_search.cv_results_['std_test_score']
C_values = param_grid['classifier__C']

# Plot the results of the hyperparameter tuning
plt.figure(figsize=(10, 6))
plt.errorbar(C_values, mean_test_scores, yerr=std_test_scores, fmt='-o', capsize=5)
plt.xscale('log')  # Use logarithmic scale for C values
plt.xlabel('Regularization strength (C)')
plt.ylabel('Mean AUROC')
plt.title('Hyperparameter Search: AUROC for Logistic Regression with TF-IDF')
plt.grid(True)
plt.show()

# ------------------------ Figure 2: False Positives and False Negatives ------------------------
# Get the predictions for the validation set
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

# False positives (predicted positive, but actually negative)
false_positives = X_val[(y_val == 0) & (y_val_pred == 1)]

# False negatives (predicted negative, but actually positive)
false_negatives = X_val[(y_val == 1) & (y_val_pred == 0)]

# Set up figure for False Positives and False Negatives
plt.figure(figsize=(10, 8))

# Plot 10 false positives
for i, example in enumerate(false_positives[:7]):
    plt.text(0.1, 0.9 - (i * 0.07), f"FP{i+1}: {example[:80]}...", fontsize=10, ha='left', wrap=True)

# Plot 10 false negatives
for i, example in enumerate(false_negatives[:7]):
    plt.text(0.1, 0.4 - (i * 0.07), f"FN{i+1}: {example[:80]}...", fontsize=10, ha='left', wrap=True)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.axis('off')
plt.title('Examples of False Positives and False Negatives')
plt.show()

# Print the full examples of false positives and false negatives in the console
print("\nFalse Positives (FP):")
for i, example in enumerate(false_positives[:3]):
    print(f"{i+1}. {example}\n")

print("\nFalse Negatives (FN):")
for i, example in enumerate(false_negatives[:3]):
    print(f"{i+1}. {example}\n")