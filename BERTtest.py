# import torch
# import numpy as np
# import pandas as pd
# from transformers import BertTokenizer, BertModel
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
# from sklearn.model_selection import cross_val_score, train_test_split
# import os
# import matplotlib.pyplot as plt

# # Load BERT tokenizer and model
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)

# # Function to get BERT embeddings
# def get_bert_embedding(sentence_list, pooling_strategy='cls'):
#     embedding_list = []
#     for sentence in sentence_list:
#         # Tokenize the sentence and get the output from BERT
#         inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         last_hidden_states = outputs.last_hidden_state[0]
        
#         # Pooling strategy
#         if pooling_strategy == "cls":
#             sentence_embedding = last_hidden_states[0]  # Use the [CLS] token embedding
#         elif pooling_strategy == "mean":
#             sentence_embedding = torch.mean(last_hidden_states, dim=0)  # Mean pooling
#         embedding_list.append(sentence_embedding)
#     return torch.stack(embedding_list)

# # Load training and test data
# x_train_df = pd.read_csv('data_reviews/x_train.csv')
# x_test_df = pd.read_csv('data_reviews/x_test.csv')
# y_train = pd.read_csv('data_reviews/y_train.csv')['is_positive_sentiment'].values

# tr_text_list = x_train_df['text'].values.tolist()
# te_text_list = x_test_df['text'].values.tolist()

# # Generate BERT embeddings for train and test sets
# print('Generating embeddings for train sequences...')
# tr_embedding = get_bert_embedding(tr_text_list)
# print('Generating embeddings for test sequences...')
# te_embedding = get_bert_embedding(te_text_list)

# # Convert embeddings to numpy arrays
# tr_embeddings_ND = tr_embedding.numpy()
# te_embeddings_ND = te_embedding.numpy()

# # --- Step 3: Train Logistic Regression classifier ---
# clf = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues
# clf.fit(tr_embeddings_ND, y_train)

# # --- Step 4: Cross-validation ---
# cv_scores = cross_val_score(clf, tr_embeddings_ND, y_train, cv=5, scoring='accuracy')  # 5-fold cross-validation
# print(f"Cross-validation accuracy scores: {cv_scores}")
# print(f"Mean CV accuracy: {np.mean(cv_scores)}")

# # --- Step 5: Predict probabilities for test set ---
# y_proba_test = clf.predict_proba(te_embeddings_ND)[:, 1]  # Get probability of positive class

# # --- Step 6: Save the test probabilities to 'yproba2_test.txt' ---
# np.savetxt('yproba2_test.txt', y_proba_test)
# print(f"Test probabilities saved to 'yproba2_test.txt'")

# import torch
# import numpy as np
# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# # Load BERT tokenizer and model for sequence classification
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)

# # Load training data
# x_train_df = pd.read_csv('data_reviews/x_train.csv')
# y_train = pd.read_csv('data_reviews/y_train.csv')['is_positive_sentiment'].values
# tr_text_list = x_train_df['text'].values.tolist()

# # --- Step 1: Split the data into training and validation sets ---
# train_texts, val_texts, y_train, y_val = train_test_split(tr_text_list, y_train, test_size=0.2, random_state=42)

# # --- Step 2: Tokenize inputs for both training and validation sets ---
# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# # Convert to torch dataset
# class SentimentDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = SentimentDataset(train_encodings, y_train)
# val_dataset = SentimentDataset(val_encodings, y_val)

# # --- Step 3: Function to fine-tune and evaluate the model at different learning rates ---
# def evaluate_learning_rates(learning_rates):
#     performance_scores = []

#     for lr in learning_rates:
#         print(f"Training with learning rate: {lr}")
#         # Load model for each training session
#         model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

#         # Define training arguments with variable learning rate
#         training_args = TrainingArguments(
#             output_dir='./results',          # Output directory
#             num_train_epochs=3,              # Number of training epochs
#             per_device_train_batch_size=16,  # Batch size
#             learning_rate=lr,                # Learning rate
#             evaluation_strategy="epoch",     # Evaluate at the end of each epoch
#             logging_dir='./logs',            # Directory for logging
#             logging_steps=10,                # Log every 10 steps
#         )

#         # Define Trainer
#         trainer = Trainer(
#             model=model,                         
#             args=training_args,                  
#             train_dataset=train_dataset,         
#             eval_dataset=val_dataset,            # Evaluation set
#             compute_metrics=lambda p: {
#                 'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
#                 'roc_auc': roc_auc_score(p.label_ids, p.predictions[:, 1])
#             }
#         )

#         # Train the model
#         trainer.train()

#         # Evaluate the model
#         eval_metrics = trainer.evaluate()
#         print(f"Validation Accuracy: {eval_metrics['eval_accuracy']}, ROC AUC: {eval_metrics['eval_roc_auc']}")
#         performance_scores.append((eval_metrics['eval_accuracy'], eval_metrics['eval_roc_auc']))

#     return performance_scores

# # --- Step 4: Define learning rates to experiment with ---
# learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]

# # --- Step 5: Evaluate model performance at different learning rates ---
# performance_scores = evaluate_learning_rates(learning_rates)

# # Extract accuracy and ROC AUC scores for plotting
# accuracies = [score[0] for score in performance_scores]
# roc_aucs = [score[1] for score in performance_scores]

# # --- Step 6: Plot the results ---
# plt.figure(figsize=(10, 5))

# # Accuracy plot
# plt.subplot(1, 2, 1)
# plt.plot(learning_rates, accuracies, marker='o', color='b', label='Accuracy')
# plt.xscale('log')
# plt.xlabel('Learning Rate (log scale)')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs Learning Rate')
# plt.grid(True)

# # ROC AUC plot
# plt.subplot(1, 2, 2)
# plt.plot(learning_rates, roc_aucs, marker='o', color='g', label='ROC AUC')
# plt.xscale('log')
# plt.xlabel('Learning Rate (log scale)')
# plt.ylabel('ROC AUC')
# plt.title('ROC AUC vs Learning Rate')
# plt.grid(True)

# plt.tight_layout()
# plt.show()



import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# Load the data
x_train_df = pd.read_csv('data_reviews/x_train.csv')
y_train = pd.read_csv('data_reviews/y_train.csv')


# Extract the review text and the labels
X_text = x_train_df['text'].values  # Text data
y = y_train['is_positive_sentiment'].values  # Labels


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Function to generate BERT embeddings
def get_bert_embeddings(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        # Get the CLS token embedding (this represents the entire sentence)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding)
    return np.vstack(embeddings)


# Split data into training and validation sets
X_text_train, X_text_val, y_train, y_val = train_test_split(X_text, y, test_size=0.2, random_state=42)


# Generate BERT embeddings for training and validation data
X_bert_train = get_bert_embeddings(X_text_train)
X_bert_val = get_bert_embeddings(X_text_val)


# Stack the text and BERT embeddings to pass them as one input to the pipeline
X_train_combined = np.hstack((X_text_train.reshape(-1, 1), X_bert_train))
X_val_combined = np.hstack((X_text_val.reshape(-1, 1), X_bert_val))


# Preprocessing: TF-IDF Vectorizer for text, and StandardScaler for BERT embeddings
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2)), 0),  # Apply TF-IDF on the text data (first column)
        ('scaler', StandardScaler(), slice(1, None))  # Apply scaling on the BERT embeddings (remaining columns)
    ],
    remainder='passthrough'
)


# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)


# Create a pipeline for preprocessing and classification
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_clf)
])


# Set up hyperparameter grid for Random Forest
param_grid = {
    'classifier__n_estimators': [700],  # Number of estimators for Random Forest
    # Below is the hyperparemter we changed
    'classifier__max_depth': [0, 10, 20, 30, 40]       # Max depth for Random Forest
}


# Cross-validation with Grid Search for hyperparameter tuning (using 10 folds)
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='roc_auc', verbose=1)


# Fit the model using the combined text and BERT embeddings
grid_search.fit(X_train_combined, y_train)


# Best parameters from grid search
print("Best parameters found: ", grid_search.best_params_)


# Evaluate the model on validation set using AUROC
y_val_pred_proba = grid_search.predict_proba(X_val_combined)[:, 1]  # Get the predicted probabilities
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUROC: {roc_auc}")


# Cross-validation scores for the best model (10 folds)
cv_scores = cross_val_score(grid_search.best_estimator_, X_train_combined, y_train, cv=10, scoring='roc_auc')
print(f"Cross-validation AUROC scores: {cv_scores}")
print(f"Mean AUROC: {cv_scores.mean()}")


# Access the best pipeline from the grid search
best_pipeline = grid_search.best_estimator_


# Access the fitted vectorizer from the pipeline
fitted_vectorizer = best_pipeline.named_steps['preprocessor'].transformers_[0][1]
vocabulary_size = len(fitted_vectorizer.vocabulary_)


print(f"Vocabulary size: {vocabulary_size}")


# Load test data
x_test_df = pd.read_csv('data_reviews/x_test.csv')


# Extract the review text (from the test data)
X_test_text = x_test_df['text'].values  # Text data for the test set


# Generate BERT embeddings for the test set
X_bert_test = get_bert_embeddings(X_test_text)


# Combine text and BERT embeddings for the test set
X_test_combined = np.hstack((X_test_text.reshape(-1, 1), X_bert_test))


# Predict probabilities for the test set
y_test_pred_proba = best_pipeline.predict_proba(X_test_combined)[:, 1]


# Save predictions
pd.DataFrame(y_test_pred_proba, columns=['is_positive_sentiment']).to_csv('yproba1_test.txt', index=False)

# # Extract the results from the grid search
# results = grid_search.cv_results_

# # Get the max_depth values and corresponding mean test scores (AUROC)
# max_depths = param_grid['classifier__max_depth']
# mean_roc_auc_scores = results['mean_test_score']

# # Plotting the results
# plt.figure(figsize=(8, 6))
# plt.plot(max_depths, mean_roc_auc_scores, marker='o', linestyle='-', color='b', label='Mean ROC AUC')

# # Adding labels and title
# plt.xlabel('Max Depth of Random Forest Classifier')
# plt.ylabel('Mean ROC AUC')
# plt.title('Classifier Performance vs Max Depth')
# plt.xticks(max_depths)  # Ensure the x-axis ticks correspond to max_depth values
# plt.grid(True)
# plt.legend()

# # Display the plot
# plt.show()

# Get the predicted labels and probabilities on the validation set
y_val_pred = best_pipeline.predict(X_val_combined)
y_val_pred_proba = best_pipeline.predict_proba(X_val_combined)[:, 1]

# Find the indices of false positives and false negatives
false_positives = np.where((y_val_pred == 1) & (y_val == 0))[0]
false_negatives = np.where((y_val_pred == 0) & (y_val == 1))[0]

# Select a few representative examples of false positives and false negatives
num_examples = 3  # Change this to select more/less examples

# Randomly select a few false positives and false negatives
selected_fp = np.random.choice(false_positives, size=num_examples, replace=False)
selected_fn = np.random.choice(false_negatives, size=num_examples, replace=False)

# Plotting the false positives and false negatives with their predicted probabilities
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot false positives
axs[0].set_title('False Positives (Predicted Positive, Actually Negative)')
for i, idx in enumerate(selected_fp):
    axs[0].text(0.1, 0.8 - i * 0.2, f"Example {i+1}: '{X_text_val[idx]}'\n"
               f"Predicted Probability: {y_val_pred_proba[idx]:.2f} | True Label: {y_val[idx]}\n",
               fontsize=10, color='red', wrap=True)
axs[0].axis('off')

# Plot false negatives
axs[1].set_title('False Negatives (Predicted Negative, Actually Positive)')
for i, idx in enumerate(selected_fn):
    axs[1].text(0.1, 0.8 - i * 0.2, f"Example {i+1}: '{X_text_val[idx]}'\n"
               f"Predicted Probability: {y_val_pred_proba[idx]:.2f} | True Label: {y_val[idx]}\n",
               fontsize=10, color='blue', wrap=True)
axs[1].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()