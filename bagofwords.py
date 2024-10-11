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

# # Plot the results of the hyperparameter tuning
# plt.figure(figsize=(10, 6))
# plt.errorbar(C_values, mean_test_scores, yerr=std_test_scores, fmt='-o', capsize=5)
# plt.xscale('log')  # Use logarithmic scale for C values
# plt.xlabel('Regularization strength (C)')
# plt.ylabel('Mean AUROC')
# plt.title('Hyperparameter Search: AUROC for Logistic Regression with TF-IDF')
# plt.grid(True)
# plt.show()

# ------------------------ Figure 2: False Positives and False Negatives ------------------------
# Get the predictions for the validation set
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

# False positives (predicted positive, but actually negative)
false_positives = X_val[(y_val == 0) & (y_val_pred == 1)]

# False negatives (predicted negative, but actually positive)
false_negatives = X_val[(y_val == 1) & (y_val_pred == 0)]

# # Set up figure for False Positives and False Negatives
# plt.figure(figsize=(10, 8))

# # Plot 10 false positives
# for i, example in enumerate(false_positives[:7]):
#     plt.text(0.1, 0.9 - (i * 0.07), f"FP{i+1}: {example[:80]}...", fontsize=10, ha='left', wrap=True)

# # Plot 10 false negatives
# for i, example in enumerate(false_negatives[:7]):
#     plt.text(0.1, 0.4 - (i * 0.07), f"FN{i+1}: {example[:80]}...", fontsize=10, ha='left', wrap=True)

# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.axis('off')
# plt.title('Examples of False Positives and False Negatives')
# plt.show()

# # Print the full examples of false positives and false negatives in the console
# print("\nFalse Positives (FP):")
# for i, example in enumerate(false_positives[:3]):
#     print(f"{i+1}. {example}\n")

# print("\nFalse Negatives (FN):")
# for i, example in enumerate(false_negatives[:3]):
#     print(f"{i+1}. {example}\n")

# Example list of negation words
negation_words = ["not", "n't", "didn't", "shouldn't", "can't", "won't", "no", "never"]

# Function to check if a sentence contains any negation words
def contains_negation(sentence):
    return any(neg_word in sentence.lower() for neg_word in negation_words)

# Get the predictions for the validation set
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

# False positives (predicted positive, but actually negative)
false_positives = X_val[(y_val == 0) & (y_val_pred == 1)]

# False negatives (predicted negative, but actually positive)
false_negatives = X_val[(y_val == 1) & (y_val_pred == 0)]

# --- Analysis 1: Longer vs. Shorter Sentences ---

# Calculate sentence lengths (number of words)
sentence_lengths = np.array([len(text.split()) for text in X_val])

# Median length as the threshold
median_length = np.median(sentence_lengths)

# Long vs. short
short_sentences = sentence_lengths <= median_length
long_sentences = sentence_lengths > median_length

# FP and FN rates for short vs long sentences
short_sentence_fp_rate = np.mean((y_val[short_sentences] == 0) & (y_val_pred[short_sentences] == 1))
short_sentence_fn_rate = np.mean((y_val[short_sentences] == 1) & (y_val_pred[short_sentences] == 0))

long_sentence_fp_rate = np.mean((y_val[long_sentences] == 0) & (y_val_pred[long_sentences] == 1))
long_sentence_fn_rate = np.mean((y_val[long_sentences] == 1) & (y_val_pred[long_sentences] == 0))

# --- Analysis 2: Reviews from Different Sources (Assumes 'source' column in the dataset) ---

# Ensure 'website_name' is the column being used for review source analysis (full dataset)
review_sources = x_train_df['website_name'].values

# We need to ensure that the 'review_sources' matches the validation set size
X_train, X_val_full, y_train, y_val_full, review_sources_train, review_sources_val = train_test_split(
    X, y, review_sources, test_size=0.2, random_state=42
)

# Extract the boolean masks for the review sources in the validation set
amazon_reviews = review_sources_val == 'amazon'
imdb_reviews = review_sources_val == 'imdb'
yelp_reviews = review_sources_val == 'yelp'

# Now calculate False Positive and False Negative rates for Amazon reviews
amazon_fp_rate = np.mean((y_val_full[amazon_reviews] == 0) & (y_val_pred[amazon_reviews] == 1))
amazon_fn_rate = np.mean((y_val_full[amazon_reviews] == 1) & (y_val_pred[amazon_reviews] == 0))

# Same for IMDB reviews
imdb_fp_rate = np.mean((y_val_full[imdb_reviews] == 0) & (y_val_pred[imdb_reviews] == 1))
imdb_fn_rate = np.mean((y_val_full[imdb_reviews] == 1) & (y_val_pred[imdb_reviews] == 0))

# Same for Yelp reviews
yelp_fp_rate = np.mean((y_val_full[yelp_reviews] == 0) & (y_val_pred[yelp_reviews] == 1))
yelp_fn_rate = np.mean((y_val_full[yelp_reviews] == 1) & (y_val_pred[yelp_reviews] == 0))

# Output the results
sources_with_rates = [
    ("Amazon", amazon_fp_rate, amazon_fn_rate),
    ("IMDB", imdb_fp_rate, imdb_fn_rate),
    ("Yelp", yelp_fp_rate, yelp_fn_rate)
]

for source_name, fp_rate, fn_rate in sources_with_rates:
    print(f"{source_name} - FP: {fp_rate}, FN: {fn_rate}")

# --- Analysis 3: Negation Words ---

# Check presence of negation words in the validation set
negation_present = np.array([contains_negation(text) for text in X_val])

# FP and FN rates for sentences with/without negation
negation_fp_rate = np.mean((y_val[negation_present] == 0) & (y_val_pred[negation_present] == 1))
negation_fn_rate = np.mean((y_val[negation_present] == 1) & (y_val_pred[negation_present] == 0))

non_negation_fp_rate = np.mean((y_val[~negation_present] == 0) & (y_val_pred[~negation_present] == 1))
non_negation_fn_rate = np.mean((y_val[~negation_present] == 1) & (y_val_pred[~negation_present] == 0))

# --- Plotting Results ---

categories = ['Short Sentences', 'Long Sentences']
fp_rates = [short_sentence_fp_rate, long_sentence_fp_rate]
fn_rates = [short_sentence_fn_rate, long_sentence_fn_rate]

# Plot for Sentence Length Analysis (FP and FN rates)
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
width = 0.35  # Bar width
x = np.arange(len(categories))
plt.bar(x - width/2, fp_rates, width, label='False Positive Rate')
plt.bar(x + width/2, fn_rates, width, label='False Negative Rate')
plt.xticks(x, categories)
plt.ylabel('Rate')
plt.title('False Positive and False Negative Rates by Sentence Length')
plt.legend()

# Plot for Negation Words Analysis (FP and FN rates)
categories = ['With Negation', 'Without Negation']
fp_rates = [negation_fp_rate, non_negation_fp_rate]
fn_rates = [negation_fn_rate, non_negation_fn_rate]

plt.subplot(3, 1, 2)
x = np.arange(len(categories))
plt.bar(x - width/2, fp_rates, width, label='False Positive Rate')
plt.bar(x + width/2, fn_rates, width, label='False Negative Rate')
plt.xticks(x, categories)
plt.ylabel('Rate')
plt.title('False Positive and False Negative Rates by Negation Words')
plt.legend()

# --- Plotting Results ---

categories = ['Short Sentences', 'Long Sentences']
fp_rates = [short_sentence_fp_rate, long_sentence_fp_rate]
fn_rates = [short_sentence_fn_rate, long_sentence_fn_rate]

# Plot for Sentence Length Analysis (FP and FN rates)
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
width = 0.35  # Bar width
x = np.arange(len(categories))
plt.bar(x - width/2, fp_rates, width, label='False Positive Rate')
plt.bar(x + width/2, fn_rates, width, label='False Negative Rate')
plt.xticks(x, categories)
plt.ylabel('Rate')
plt.title('False Positive and False Negative Rates by Sentence Length')
plt.legend()

# Plot for Negation Words Analysis (FP and FN rates)
categories = ['With Negation', 'Without Negation']
fp_rates = [negation_fp_rate, non_negation_fp_rate]
fn_rates = [negation_fn_rate, non_negation_fn_rate]

plt.subplot(3, 1, 2)
x = np.arange(len(categories))
plt.bar(x - width/2, fp_rates, width, label='False Positive Rate')
plt.bar(x + width/2, fn_rates, width, label='False Negative Rate')
plt.xticks(x, categories)
plt.ylabel('Rate')
plt.title('False Positive and False Negative Rates by Negation Words')
plt.legend()

# Plot for Review Source (if available)
# if 'source' in x_train_df.columns:
categories = ['Amazon', 'IMDB', 'Yelp']
fp_rates = [amazon_fp_rate, imdb_fp_rate, yelp_fp_rate]
fn_rates = [amazon_fn_rate, imdb_fn_rate, yelp_fn_rate]

plt.subplot(3, 1, 3)
x = np.arange(len(categories))
plt.bar(x - width/2, fp_rates, width, label='False Positive Rate')
plt.bar(x + width/2, fn_rates, width, label='False Negative Rate')
plt.xticks(x, categories)
plt.ylabel('Rate')
plt.title('False Positive and False Negative Rates by Review Source')
plt.legend()

plt.tight_layout()
plt.show()

# --- Print results in the console ---

print("False Positive and False Negative Rates for Short and Long Sentences:")
print(f"Short Sentences - FP: {short_sentence_fp_rate}, FN: {short_sentence_fn_rate}")
print(f"Long Sentences - FP: {long_sentence_fp_rate}, FN: {long_sentence_fn_rate}")

print("\nFalse Positive and False Negative Rates for Negation Words:")
print(f"With Negation - FP: {negation_fp_rate}, FN: {negation_fn_rate}")
print(f"Without Negation - FP: {non_negation_fp_rate}, FN: {non_negation_fn_rate}")

if 'source' in x_train_df.columns:
    print("\nFalse Positive and False Negative Rates by Review Source:")
    print(f"Amazon - FP: {amazon_fp_rate}, FN: {amazon_fn_rate}")
    print(f"IMDB - FP: {imdb_fp_rate}, FN: {imdb_fn_rate}")
    print(f"Yelp - FP: {yelp_fp_rate}, FN: {yelp_fn_rate}")