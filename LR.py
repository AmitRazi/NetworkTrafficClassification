import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


df = pd.read_parquet('./final_processed_pcaps.parquet')

num_of_packets = 30

# Filter sessions with at least num_of_packets packets
df_filtered = df.groupby('flownum').filter(lambda x: len(x) >= num_of_packets)

# Generate sequences and labels
sequences = []
labels = []

for name, group in df_filtered.groupby('flownum'):
    session_data = group[['direction', 'length', 'relative_time']].values[:30]
    sequences.append(session_data)
    labels.append(group['application'].iloc[0])

# Convert lists to arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Now, balance the classes by selecting 5,000 samples from each class
seq_df = pd.DataFrame({'sequences': list(sequences), 'labels': labels})

# Balance the classes
balanced_df = seq_df.groupby('labels').apply(lambda x: x.sample(n=5000, replace=True, random_state=42)).reset_index(drop=True)

# Separate the balanced sequences and labels
balanced_sequences = np.array(list(balanced_df['sequences']))
balanced_labels = balanced_df['labels'].values

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(balanced_labels)

# Split the dataset into 70% training and 30% for testing + validation
X_train, X_test_val, y_train, y_test_val = train_test_split(balanced_sequences, encoded_labels, test_size=0.3, random_state=42)

# Split the 30% test_val set equally into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

num_samples, num_time_steps, num_features = X_train.shape

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

log_reg = LogisticRegression(max_iter=1000)

grid_search_lr = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, cv=2, scoring='accuracy', verbose=2, n_jobs=-1)

# Train the Logistic Regression model
grid_search_lr.fit(X_train_flat, y_train)

# Print the best parameters and best score found
print("Best parameters for Logistic Regression: ", grid_search_lr.best_params_)
print("Best cross-validation score for Logistic Regression: {:.2f}".format(grid_search_lr.best_score_))

# Evaluate on test data
best_model_lr = grid_search_lr.best_estimator_
test_accuracy_lr = best_model_lr.score(X_test_flat, y_test)
print(f'Test Accuracy for Logistic Regression: {test_accuracy_lr:.4f}')


