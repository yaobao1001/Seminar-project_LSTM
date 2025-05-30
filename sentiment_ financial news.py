#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:37:34 2025

@author: zhouyiyao
"""

import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# Specify column types explicitly to avoid the mixed types warning
dtype_dict = {
    'Column1': 'str',
    'Column2': 'float64'}

chunk_size = 100000
chunks = []

for chunk in pd.read_csv('/Users/zhouyiyao/Downloads/nasdaq_exteral_data.csv', chunksize=chunk_size, delimiter=',', dtype=dtype_dict):
    # Filter columns as needed
    chunk = chunk[['Date', 'Article_title']]  # Adjust column names if needed
    chunks.append(chunk)

# Combine all chunks into a final DataFrame
filtered_news_data = pd.concat(chunks, ignore_index=True)

# Save the filtered news data to a CSV file
filtered_news_data.to_csv('/Users/zhouyiyao/Desktop/filtered_news_data.csv', index=False)

# Verify that the data has been saved correctly
print("Filtered news data saved successfully!")



# Load the filtered news data
filtered_news_data = pd.read_csv('/Users/zhouyiyao/Desktop/filtered_news_data.csv')

# Convert 'Date' column to datetime if not already
filtered_news_data['Date'] = pd.to_datetime(filtered_news_data['Date'], errors='coerce')

# Define the time range
start_date = '1998-01-01'
end_date = '2023-03-31'

# Filter the news data to match the time period
filtered_news_data = filtered_news_data[(filtered_news_data['Date'] >= start_date) & 
                                         (filtered_news_data['Date'] <= end_date)]

# Sort the news data by Date
filtered_news_data = filtered_news_data.sort_values(by='Date')

# Sample 2% of the data
filtered_news_data = filtered_news_data.sample(frac=0.02, random_state=42)

# Sort the sampled data by Date
filtered_news_data = filtered_news_data.sort_values(by='Date').reset_index(drop=True)

# Split the data into train, validation, and test sets (70%, 15%, 15%)
train_size = int(len(filtered_news_data) * 0.7)
val_size = int(len(filtered_news_data) * 0.15)

news_train = filtered_news_data[:train_size]
news_val = filtered_news_data[train_size:train_size + val_size]
news_test = filtered_news_data[train_size + val_size:]

# Save the split datasets for reference
news_train.to_csv('/Users/zhouyiyao/Downloads/news_train.csv', index=False)
news_val.to_csv('/Users/zhouyiyao/Downloads/news_val.csv', index=False)
news_test.to_csv('/Users/zhouyiyao/Downloads/news_test.csv', index=False)

# Print the size of each split to verify
print(f"Train size: {len(news_train)}")
print(f"Validation size: {len(news_val)}")
print(f"Test size: {len(news_test)}")


# Disable Huggingface tokenizer parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Function to predict continuous sentiment scores
def predict_continuous_sentiment(texts, batch_size=32):
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits[:, 2].tolist()  # Extract positive sentiment scores
        all_scores.extend(scores)
    return all_scores

# Function to add continuous sentiment scores to data
def add_sentiment_scores(df, batch_size=32):
    texts = df["Article_title"].tolist()
    sentiment_scores = predict_continuous_sentiment(texts, batch_size)
    df["sentiment_score"] = sentiment_scores
    return df

# Load datasets
news_train = pd.read_csv('/Users/zhouyiyao/Downloads/news_train.csv')
news_val = pd.read_csv('/Users/zhouyiyao/Downloads/news_val.csv')
news_test = pd.read_csv('/Users/zhouyiyao/Downloads/news_test.csv')

# Apply sentiment prediction to datasets
news_train = add_sentiment_scores(news_train, batch_size=32)
news_val = add_sentiment_scores(news_val, batch_size=32)
news_test = add_sentiment_scores(news_test, batch_size=32)

# Save the sentiment-scored datasets
news_train.to_csv('/Users/zhouyiyao/Downloads/news_train_with_sentiment.csv', index=False)
news_val.to_csv('/Users/zhouyiyao/Downloads/news_val_with_sentiment.csv', index=False)
news_test.to_csv('/Users/zhouyiyao/Downloads/news_test_with_sentiment.csv', index=False)

# Print out some of the processed examples to verify
print(news_train.head())
print(news_train.describe())


import matplotlib.pyplot as plt

# Plot sentiment score distribution
plt.figure(figsize=(10, 6))
news_train['sentiment_score'].hist(bins=50, color='blue', alpha=0.7, label='Train')
news_val['sentiment_score'].hist(bins=50, color='green', alpha=0.5, label='Validation')
news_test['sentiment_score'].hist(bins=50, color='orange', alpha=0.5, label='Test')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()

