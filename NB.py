import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import os
import matplotlib.pyplot as plt

# Load training and test data
x_train_df = pd.read_csv('data_reviews/x_train.csv')
x_test_df = pd.read_csv('data_reviews/x_test.csv')
y_train = pd.read_csv('data_reviews/y_train.csv')['is_positive_sentiment'].values

tr_text_list = x_train_df['text'].values.tolist()
te_text_list = x_test_df['text'].values.tolist()

# --- Step 1: TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for speed
tr_tfidf = vectorizer.fit_transform(tr_text_list)
te_tfidf = vectorizer.transform(te_text_list)

# --- Step 2: Train Naive Bayes Classifier ---
nb_clf = MultinomialNB()
nb_clf.fit(tr_tfidf, y_train)

# --- Step 3: Cross-validation ---
cv_scores = cross_val_score(nb_clf, tr_tfidf, y_train, cv=5, scoring='accuracy')  # 5-fold cross-validation
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores)}")

# --- Step 4: Predict probabilities for test set ---
y_proba_test = nb_clf.predict_proba(te_tfidf)[:, 1]  # Get probability of positive class

# --- Step 5: Save the test probabilities to 'yproba2_test.txt' ---
np.savetxt('yproba2_test.txt', y_proba_test)
print(f"Test probabilities saved to 'yproba2_test.txt'")