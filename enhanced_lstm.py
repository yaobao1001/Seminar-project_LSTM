#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:30:46 2025

@author: zhouyiyao
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

# Load the merged datasets
merged_train = pd.read_csv('/Users/zhouyiyao/Downloads/merged_train_with_sentiment.csv')
merged_val = pd.read_csv('/Users/zhouyiyao/Downloads/merged_val_with_sentiment.csv')
merged_test = pd.read_csv('/Users/zhouyiyao/Downloads/merged_test_with_sentiment.csv')

# Initialize scalers
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_sentiment = MinMaxScaler(feature_range=(0, 1))

# Normalize the 'Close' prices
merged_train['Close'] = scaler_close.fit_transform(merged_train[['Close']])
merged_val['Close'] = scaler_close.transform(merged_val[['Close']])
merged_test['Close'] = scaler_close.transform(merged_test[['Close']])

# Normalize the sentiment scores
merged_train['sentiment_score'] = scaler_sentiment.fit_transform(merged_train[['sentiment_score']])
merged_val['sentiment_score'] = scaler_sentiment.transform(merged_val[['sentiment_score']])
merged_test['sentiment_score'] = scaler_sentiment.transform(merged_test[['sentiment_score']])

# Prepare features and target
X_train = merged_train[['Close', 'sentiment_score']].values
X_val = merged_val[['Close', 'sentiment_score']].values
X_test = merged_test[['Close', 'sentiment_score']].values

y_train = merged_train['Close'].values
y_val = merged_val['Close'].values
y_test = merged_test['Close'].values

# Reshape features for LSTM input: [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

from keras.models import Sequential
from keras.layers import LSTM, Dense
import json


# Define a new LSTM model with two features
new_model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(1, 2)),  # Single LSTM with more units
    Dense(1)  # Output layer for regression
])

# Compile the model
new_model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping with stricter criteria
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Stop after 3 epochs without improvement
    restore_best_weights=True) # Revert to the best weights


# Train the model
history = new_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the new model for future use
new_model.save('three-layer-lstm_with_sentiment_model.keras')

# Evaluate the model
test_loss = new_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predict stock prices using the trained model
predicted_prices = new_model.predict(X_test)

# Reverse scaling for actual and predicted prices
y_test_rescaled = scaler_close.inverse_transform(y_test.reshape(-1, 1))
predicted_prices_rescaled = scaler_close.inverse_transform(predicted_prices)

# Ensure the length of the test data matches
filtered_test_data = pd.DataFrame({
    'Date': merged_test['Date'],
    'Close': y_test_rescaled.flatten()})

# Assign predicted and actual prices to the filtered_test_data DataFrame
filtered_test_data['Predicted_Close'] = predicted_prices_rescaled[:len(filtered_test_data)].flatten()
filtered_test_data['Actual_Close'] = y_test_rescaled[:len(filtered_test_data)].flatten()


# Sort by date to ensure proper alignment for plotting
filtered_test_data['Date'] = pd.to_datetime(filtered_test_data['Date'])
filtered_test_data = filtered_test_data.sort_values(by='Date').reset_index(drop=True)

# Aggregate data to remove duplicates
filtered_test_data_agg = (
    filtered_test_data.groupby("Date", as_index=False)
    .agg({"Actual_Close": "mean", "Predicted_Close": "mean"})
)

print(filtered_test_data_agg.head())
print(filtered_test_data_agg.describe())

# Compute metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predicted_prices_rescaled))
mae = mean_absolute_error(y_test_rescaled, predicted_prices_rescaled)
print(f"RMSE: {rmse}, MAE: {mae}")


# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(
    filtered_test_data_agg["Date"],
    filtered_test_data_agg["Actual_Close"],
    label="Actual Prices",
    color="blue",
)
plt.plot(
    filtered_test_data_agg["Date"],
    filtered_test_data_agg["Predicted_Close"],
    label="Predicted Prices",
    color="orange",
)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Train Loss", color="blue")
plt.plot(epochs, val_loss, label="Validation Loss", color="orange")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Optional: Save metrics
metrics = {
    "test_loss": test_loss,
    "rmse": rmse,
    "mae": mae,
    "train_loss": train_loss[-1],
    "val_loss": val_loss[-1]
}
with open("/Users/zhouyiyao/Downloads/enhanced_lstm_with_sentiment_metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved to 'enhanced_lstm_with_sentiment_metrics.json'.")
