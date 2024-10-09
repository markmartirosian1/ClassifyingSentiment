import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Load the data
x_train_df = pd.read_csv('data_reviews/x_train.csv')
y_train = pd.read_csv('data_reviews/y_train.csv')

# Extract the review text and the labels
X = x_train_df['text'].values  # text data
y = y_train['is_positive_sentiment'].values  # labels

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Bag-of-Words
vectorizer = CountVectorizer(lowercase=True, stop_words='english', max_df=0.6, min_df=5)

# Logistic Regression Classifier
log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Create a pipeline for preprocessing and classification
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', log_reg)
])

# Set up hyperparameter grid for Logistic Regression
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Regularization strength
    'classifier__penalty': ['l2'],  # Only L2 is supported for this solver
}

# Cross-validation with Grid Search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1, return_train_score=True)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Retrieve grid search results
results = grid_search.cv_results_

# # Check the available keys in the results
# print("Available keys in results:", results.keys())

# # Extract mean and std for training and validation AUROC
# mean_train_scores = results['mean_train_score']
# mean_val_scores = results['mean_test_score']
# std_train_scores = results['std_train_score']
# std_val_scores = results['std_test_score']
# param_values = results['params']  # This should be a list of dictionaries

# # Ensure param_values is of the expected format
# print("Example parameter set:", param_values[0])  # Check the first item

# # Convert param_values to a list for plotting
# C_values = [param['classifier__C'] for param in param_values if isinstance(param, dict)]

# # Plot the results
# plt.figure(figsize=(10, 6))

# # Plot training scores
# plt.errorbar(C_values, mean_train_scores, yerr=std_train_scores, label='Training AUROC', marker='o')
# # Plot validation scores
# plt.errorbar(C_values, mean_val_scores, yerr=std_val_scores, label='Validation AUROC', marker='o')

# plt.xscale('log')
# plt.xlabel('Regularization Strength (C)')
# plt.ylabel('AUROC Score')
# plt.title('AUROC Scores for Logistic Regression with Varying C Values')
# plt.legend()
# plt.grid()
# plt.show()

# Evaluate the model on validation set using AUROC
# Get the predictions on the validation set
y_val_pred_proba = grid_search.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)  # Set a threshold for classification

# Create a DataFrame to hold the predictions, actual values, and the text
predictions_df = pd.DataFrame({
    'text': X_val,
    'predicted': y_val_pred,
    'actual': y_val,
    'predicted_proba': y_val_pred_proba
})

# Identify false positives and false negatives
false_positives = predictions_df[(predictions_df['predicted'] == 1) & (predictions_df['actual'] == 0)]
false_negatives = predictions_df[(predictions_df['predicted'] == 0) & (predictions_df['actual'] == 1)]

# Select a subset for visualization
fp_samples = false_positives.sample(n=5, random_state=42)  # 5 random false positives
fn_samples = false_negatives.sample(n=5, random_state=42)  # 5 random false negatives

# Create a figure to show the results
plt.figure(figsize=(12, 8))

# Plot false positives
plt.subplot(2, 1, 1)
plt.title('False Positives')
plt.axis('off')
for i, (index, row) in enumerate(fp_samples.iterrows()):
    plt.text(0.5, 1 - i * 0.2, f'Predicted: {row["predicted"]}, Actual: {row["actual"]}\nReview: {row["text"]}',
             wrap=True, horizontalalignment='center', fontsize=12)

# Plot false negatives
plt.subplot(2, 1, 2)
plt.title('False Negatives')
plt.axis('off')
for i, (index, row) in enumerate(fn_samples.iterrows()):
    plt.text(0.5, 1 - i * 0.2, f'Predicted: {row["predicted"]}, Actual: {row["actual"]}\nReview: {row["text"]}',
             wrap=True, horizontalalignment='center', fontsize=12)

plt.tight_layout()
plt.show()

roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUROC: {roc_auc}")

# Access the fitted vectorizer from the pipeline
best_pipeline = grid_search.best_estimator_
fitted_vectorizer = best_pipeline.named_steps['vectorizer']

# Load and predict on the test set
x_test_df = pd.read_csv('data_reviews/x_test.csv')
X_test = x_test_df['text'].values  # text data for the test set

y_test_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_proba >= 0.3).astype(int)
pd.DataFrame(y_test_pred, columns=['is_positive_sentiment']).to_csv('yproba1_test.txt', index=False)