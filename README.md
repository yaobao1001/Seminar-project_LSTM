# Seminar-project_LSTM
This repository contains the code, data, and documentation for my seminar project titled "Forecasting Bank Crises-Integrating LSTM Models with Sentiment Analysis", developed as part of my Master's studies in Money and Finance at Goethe University Frankfurt.

## 📌 Project Description 
The goal of this project is to predict potential banking crises using sentiment analysis and stock prices. The model integrates:

- Sentiment scores extracted from financial news headlines
- Stock prices 
- A deep learning model based on LSTM architecture

## 🧠 Technologies Used
- Python 3.12
- Pandas & NumPy
- Scikit-learn
- TensorFlow / PyTorch
- Huggingface Transformers
- Matplotlib

## 📁 Folder Structure
project/
│
├── data/
│   ├── merged_train_with_sentiment.csv
│   ├── merged_val_with_sentiment.csv
│   └── merged_test_with_sentiment.csv
│
├── src/
│   ├── sentiment_financial news.py
│   ├── merged_dataset.py
│   ├── lstm_baseline.py
│   ├── enhanced_lstm.py
│   └── enhanced_lstm.py
│
├── notebooks/
│   └── baseline_metrics_with_rmse_mae.json
│   └── enhanced_lstm_with_sentiment_metrics.json
│
├── README.md
└── requirements.txt

## 📊 Results

The model achieved the final test loss of 0.0195 from the enhanced model. Detailed evaluation results and graphs can be found in the `notebooks` folder.

## 🔧 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yaobao1001/Seminar-project_LSTM.git
cd seminar-project

2.	Install the required packages:

pip install -r requirements.txt

3.	Run the main script:

python src/lstm_baseline.py


