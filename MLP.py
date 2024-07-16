import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
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

# Define the model
model = Sequential([
    Flatten(input_shape=(num_time_steps, num_features)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(np.unique(y_train).size, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])


# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}')