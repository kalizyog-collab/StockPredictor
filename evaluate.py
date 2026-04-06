import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    """
    Calculate prediction error metrics.

    MAE  = average absolute error
    RMSE = root mean squared error

    Lower values are better.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse
    }


def direction_accuracy(y_true, y_pred):
    """
    Check whether the model guessed the direction correctly.

    Example:
    If actual price went up and prediction also went up,
    that counts as correct.

    Output:
    Value between 0 and 1
    Higher is better.
    """

    correct = 0
    total = 0

    for i in range(1, len(y_true)):
        actual_move = y_true[i] - y_true[i - 1]
        predicted_move = y_pred[i] - y_pred[i - 1]

        # Compare direction only, not exact value
        if (actual_move >= 0 and predicted_move >= 0) or (actual_move < 0 and predicted_move < 0):
            correct += 1

        total += 1

    if total == 0:
        return 0.0

    return correct / total