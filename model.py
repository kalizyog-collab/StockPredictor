from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def build_lstm_model(input_shape):
    """
    Build the LSTM model.

    input_shape is:
    (number of time steps, number of features)

    Example:
    (10, 5) means:
    - 10 days per sequence
    - 5 features per day
    """

    model = Sequential()

    # Define the input shape
    model.add(Input(shape=input_shape))

    # First LSTM layer learns time patterns
    model.add(LSTM(64, return_sequences=True))

    # Dropout helps reduce overfitting
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    # Dense layers for final prediction
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model