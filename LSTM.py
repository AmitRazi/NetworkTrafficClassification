import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report

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
# First, convert sequences and labels into a DataFrame for easier manipulation
seq_df = pd.DataFrame({'sequences': list(sequences), 'labels': labels})

# Balance the classes
balanced_df = seq_df.groupby('labels').apply(lambda x: x.sample(n=5000, replace=True, random_state=42)).reset_index(drop=True)

# Separate the balanced sequences and labels
balanced_sequences = np.array(list(balanced_df['sequences']))
balanced_labels = balanced_df['labels'].values

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(balanced_labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(balanced_sequences, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

# Normalize features based only on the training set
scaler = MinMaxScaler(feature_range=(0, 1))
num_samples, num_time_steps, num_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

# Fit on training set, transform both training and test sets
scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(num_samples, num_time_steps, num_features)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape[0], num_time_steps, num_features)

num_samples, num_time_steps, num_features = X_train_scaled.shape

print("Building and training LSTM model...")

# Define the LSTM model
model_lstm = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(num_time_steps, num_features)),  # First Bidirectional LSTM layer
    Dropout(0.25),  # Dropout for regularization
    LSTM(64),  # Second LSTM layer
    Dense(np.unique(y_train).size, activation='softmax')  # Output layer
])


model_lstm.compile(optimizer=Adam(learning_rate=0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Note: Make sure X_train and X_test are scaled and reshaped appropriately before this step
model_lstm.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.5, callbacks=[early_stopping])

y_pred_lstm = model_lstm.predict(X_test_scaled)
y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
report_lstm = classification_report(y_test, y_pred_lstm)

print(report_lstm)