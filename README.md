# Network Traffic Classification Project

This project focuses on evaluating different machine learning models for a network traffic classification problem. The models used in this project include Long Short-Term Memory (LSTM), Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), and Logistic Regression (LR).

## Project Structure

- `LSTM.py`: Script to train and evaluate an LSTM model.
- `MLP.py`: Script to train and evaluate an MLP model.
- `CNN.py`: Script to train and evaluate a CNN model.
- `LR.py`: Script to train and evaluate a Logistic Regression model.

## Data

The data used in this project is a Parquet file containing processed packet capture (pcap) data. Each session includes features such as direction, length, and relative time of packets.

## Steps

### Data Preprocessing

1. **Filtering**: Sessions with at least 30 packets are selected.
2. **Sequencing**: Sequences of 30 packets are generated for each session.
3. **Balancing**: Classes are balanced by selecting 5,000 samples from each class.
4. **Encoding**: Labels are encoded using `LabelEncoder`.
5. **Splitting**: Data is split into training, validation, and test sets.
6. **Normalization**: Features are normalized based on the training set.

### Model Training and Evaluation

1. **LSTM**:
   - Defined using Keras Sequential API.
   - Includes layers such as Bidirectional LSTM, Dense, and Dropout.
   - Optimized using Adam optimizer.
   - Evaluated using classification report.
   
2. **MLP**:
   - Defined using Keras Sequential API.
   - Includes Dense layers with ReLU activations and Dropout.
   - Optimized using Adam optimizer.
   - Evaluated using classification report.

3. **CNN**:
   - Defined using Keras Sequential API.
   - Includes Conv1D, MaxPooling1D, Dense, and Dropout layers.
   - Optimized using Adam optimizer.
   - Evaluated using classification report.

4. **Logistic Regression**:
   - Implemented using `LogisticRegression` from `sklearn`.
   - Hyperparameters tuned using grid search.
   - Evaluated based on cross-validation score and test accuracy.

## How to Run

1. Ensure you have the necessary dependencies installed:
   \`\`\`bash
   pip install pandas numpy scikit-learn tensorflow
   \`\`\`

2. Run each script to train and evaluate the respective models:
   \`\`\`bash
   python LSTM.py
   python MLP.py
   python CNN.py
   python LR.py
   \`\`\`

## Results

The performance of each model is evaluated based on classification accuracy and other metrics provided by the classification report. The best hyperparameters for Logistic Regression are determined using grid search.
