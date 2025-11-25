# 📈 Stock Price Forecasting System — ARIMA, LSTM and Transformer

This repository contains my end-to-end Stock Price Forecasting System built using Python, PyTorch and time-series modeling techniques.  
The system predicts the next-day closing price of a stock using historical OHLCV data and compares how classical statistical models perform against modern deep learning architectures.

---

## 🎯 Objective
The aim of this project is to understand how different modeling approaches handle noisy, volatile and nonlinear financial data.  
The system analyzes historical market behavior and forecasts the next closing price using three fundamentally different models: ARIMA, LSTM and Transformer.

---

## 📊 Dataset
Source: Yahoo Finance  
Records: Daily OHLCV data  
Target: Next-day **Close** price  
Features include:  
Date, Open, High, Low, Close, Adj Close and Volume.

Each row represents one trading day and contains both price movement and liquidity information.

---

## 🧠 Models Used

### **1. ARIMA (Classical Baseline)**
A statistical model used to capture linear relationships and short-term dependencies.  
Used here as a baseline to show the performance gap between classical and deep learning methods.

### **2. LSTM (Recurrent Neural Network)**
Learns nonlinear patterns and long-term dependencies in stock data.  
More capable of handling volatility, drift and irregular temporal structures.

### **3. Transformer (Attention-based Model)**
Processes sequences in parallel using self-attention.  
Captures long-range interactions and focuses on the most relevant historical movements.  
Represents the modern state of the art in time-series forecasting.

---

## ⚙️ Data Preprocessing
All data is sorted chronologically and split into Train, Validation and Test sets to avoid leakage.  
Deep learning models use **MinMax scaling** on the Close price.  
Sliding windows of past days are converted into supervised learning samples for LSTM and Transformer.

Example:  
Window of 60 past days → predict next day’s Close.

A consistent scaler is saved and reused for evaluation.

---

## 🔧 Model Training

### **ARIMA**
Fitted using `auto_arima` to automatically detect optimal p, d and q values.  
Trained directly on the raw Close price.

### **LSTM**
Framework: PyTorch  
Loss: MSE  
Optimizer: Adam  
Early Stopping: Enabled  
Lookback Window: 60 days  
Saved as: `lstm_model.pth`

### **Transformer**
Built using PyTorch multi-head attention layers.  
Learns relationships between all timesteps simultaneously.  
Saved as: `tft_model.pth`

---

## 📈 Evaluation Metrics
Used on all models:  
RMSE, MAE, MAPE and R².

### **Performance Summary**
ARIMA performed poorly due to its linear assumptions and inability to capture volatility.  
LSTM and Transformer performed significantly better, achieving more accurate predictions and explaining more variance in price movements.

Sample results:  
ARIMA RMSE ≈ 444  
LSTM RMSE ≈ 109  
Transformer RMSE ≈ 110

Both deep learning models outperform ARIMA by nearly **4×**.

---

## 📌 Example Prediction Flow
1. Load historical data  
2. Scale inputs  
3. Construct sliding window sequences  
4. Feed the model  
5. Predict next-day Close  
6. Plot predicted vs true values  

Visual plots clearly show that deep learning models follow price movements more closely.

---

## 🌐 App (Optional)
A lightweight Streamlit interface can be added to input a sequence of past prices and generate real-time model predictions.

---

## 📂 Key Files
- `data/yahoo_stock.csv`  
- `models/arima_model.pkl`  
- `models/lstm_model.pth`  
- `models/tft_model.pth`  
- `notebooks/eda.ipynb`  
- `notebooks/preprocessing.ipynb`  
- `notebooks/train_lstm.ipynb`  
- `notebooks/train_tft.ipynb`  
- `notebooks/arima.ipynb`  

All scripts inside the `src/` folder manage preprocessing, metrics and model definitions.

---

## 🌟 Key Learnings
Built and compared three generations of time-series forecasting models.  
Understood limitations of classical forecasting in financial markets.  
Learned how LSTM and Transformers interpret temporal patterns.  
Designed a reproducible forecasting pipeline with clean modular code.  
Practiced visualization and error analysis on real stock data.

---

## 🔮 Future Enhancements
Add technical indicators such as RSI, MACD and moving averages.  
Use multivariate modeling with OHLCV inputs.  
Implement walk-forward validation for realistic forecasting.  
Deploy the model via Streamlit or FastAPI.  
Add trading strategy backtesting for practical evaluation.

---

## 👨‍💻 Author
Sai Tanoj Salehundam  
M.S. in Data Analytics Engineering — Northeastern University  
Email: salehundam.s@northeastern.edu  
