#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:36:24 2025

@author: zhouyiyao
"""


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download SVB stock data
data = yf.download("SIVBQ", start="1998-01-01", end="2023-12-31")
data = data[['Close']]

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into train, validation, and test sets
train_size = int(len(scaled_data) * 0.7)
val_size = int(len(scaled_data) * 0.15)

train_data = scaled_data[:train_size]
val_data = scaled_data[train_size:train_size + val_size]
test_data = scaled_data[train_size + val_size:]

# Create sequences for LSTM
def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(train_data, seq_length)
x_val, y_val = create_sequences(val_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# Reshape for LSTM input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(x_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=32)

# Evaluate on test data
test_loss = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")

# Generate predictions on the test set
test_predictions = model.predict(x_test)

# Convert predictions and actual values back to the original scale
test_predictions = scaler.inverse_transform(test_predictions)  # Rescale predictions
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))  # Rescale actual values
# Assuming your dataset has a 'Date' column
test_dates = data.index[-len(test_actual):]

import matplotlib.pyplot as plt

# Plot the actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(test_dates, test_actual, label="Actual Prices", color="blue")
plt.plot(test_dates, test_predictions, label="Predicted Prices", color="orange")
plt.title("Actual vs Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()

import json

metrics = {
    "test_loss": 0.0006055646226741374,
    "final_train_loss": 6.6221e-06,
    "final_val_loss": 7.7668e-05,
}
# Specify the output file path
output_path = "baseline_metrics.json"  # This defines the variable

with open("baseline_metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Metrics saved to {output_path}")
    
import matplotlib.pyplot as plt

# Example loss data
train_loss = [
    9.0912e-04, 1.5984e-05, 1.3891e-05, 1.1790e-05, 1.1512e-05, 
    1.0796e-05, 1.1238e-05, 9.5181e-06, 9.7670e-06, 8.0615e-06,
    9.9661e-06, 8.0112e-06, 8.3685e-06, 8.1281e-06, 7.2181e-06,
    7.9138e-06, 7.6266e-06, 7.3746e-06, 6.1516e-06, 6.6221e-06] # Add your training loss per epoch
val_loss = [
    5.5633e-04, 3.4259e-04, 2.2109e-04, 1.7560e-04, 1.5624e-04,
    1.4619e-04, 1.3233e-04, 1.2710e-04, 1.2059e-04, 1.0743e-04,
    1.2899e-04, 9.9558e-05, 1.2373e-04, 1.0790e-04, 8.6707e-05,
    8.8498e-05, 8.6866e-05, 7.6992e-05, 7.4812e-05, 7.7668e-05]   # Add your validation loss per epoch
epochs = range(1, 21)

plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("loss_curves.png")
plt.show()

model.save("baseline_lstm_model.keras")

