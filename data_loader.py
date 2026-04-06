import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def download_stock_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    """
    Download stock data from Yahoo Finance.

    Parameters:
        ticker: stock ticker symbol, for example AAPL
        start: start date
        end: end date

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume
    """

    # Download the stock data
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    # Sometimes yfinance can return multi-level columns
    # This makes sure the columns are simple names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Move Date from index into a normal column
    df.reset_index(inplace=True)

    # Keep only the columns we need
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Remove any missing rows
    df.dropna(inplace=True)

    # Reset row numbers
    df.reset_index(drop=True, inplace=True)

    return df


def prepare_sequences(df, feature_cols, target_col="Close", sequence_length=10):
    """
    Turn the stock data into sequences for LSTM.

    Example:
    If sequence_length = 10, the model looks at 10 days
    to predict the next day's closing price.

    Returns:
        X: input sequences
        y: target values
        feature_scaler: scaler used on input features
        target_scaler: scaler used on target
    """

    # Get only the chosen features
    feature_data = df[feature_cols].values

    # Get the target column and make it 2D for scaling
    target_data = df[target_col].values.reshape(-1, 1)

    # Scale input features and target values to numbers between 0 and 1
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaled = feature_scaler.fit_transform(feature_data)
    target_scaled = target_scaler.fit_transform(target_data)

    X = []
    y = []

    # Build the time sequences
    for i in range(sequence_length, len(df)):
        # Previous sequence_length rows become one input sample
        X.append(feature_scaled[i - sequence_length:i])

        # The current day close becomes the target
        y.append(target_scaled[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, feature_scaler, target_scaler


def train_test_split_sequences(X, y, split_ratio=0.8):
    """
    Split the sequence data into training and testing parts.

    split_ratio = 0.8 means:
    - 80% training data
    - 20% testing data
    """

    split_index = int(len(X) * split_ratio)

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test