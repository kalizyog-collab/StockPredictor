# Stock Predictor Project

## Overview
This project compares two stock prediction methods:

1. **Baseline LSTM**
   - Uses only stock market data

2. **Proposed LSTM + Sentiment**
   - Uses stock market data
   - Also uses sentiment scores from headlines

The goal is to see whether adding sentiment information improves stock prediction.

--------------------------------------------------------------------------------------

## Files in the Project

- `main.py`
  - Main file that runs the whole project

- `data_loader.py`
  - Downloads stock data from Yahoo Finance
  - Prepares the data for the LSTM model

- `sentiment.py`
  - Creates demo headlines
  - Converts them into sentiment scores

- `model.py`
  - Builds the LSTM neural network

- `evaluate.py`
  - Calculates MAE, RMSE, and Direction Accuracy

- `requirements.txt`
  - List of required Python libraries

-----------------------------------------------------------------------

## Stock Used
The current stock ticker used is:

AAPL

This can be changed inside `main.py`.

----------------------------------------------------------------------

## Date Range
The project currently uses stock data from:

- Start: 2020-01-01
- End: 2024-01-01

------------------------------------------------------------------------------------------

## How the Program Works

1. Download stock data using `yfinance`
2. Create simple demo headlines from stock movement
3. Convert headlines into sentiment scores using VADER
4. Train the baseline LSTM model
5. Train the proposed LSTM + Sentiment model
6. Compare both models using evaluation metrics
7. Save the results and graph into the `outputs` folder

---------------------------------------------------------------------

## Metrics Used

- **MAE**
  - Mean Absolute Error
  - Lower is better

- **RMSE**
  - Root Mean Squared Error
  - Lower is better

- **Direction Accuracy**
  - Checks whether the model predicted the correct price direction
  - Higher is better

---------------------------------------------------------------------------

## Install Required Libraries

Open terminal inside the project folder and run:

```bash
py -3.12 -m pip install -r requirements.txt
