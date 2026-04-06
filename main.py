import os
import matplotlib.pyplot as plt

from data_loader import download_stock_data, prepare_sequences, train_test_split_sequences
from sentiment import create_demo_headlines, add_sentiment_column
from model import build_lstm_model
from evaluate import calculate_metrics, direction_accuracy


def print_results(title, metrics, dir_acc):
    """
    Print the metrics in a clean format.
    """

    print(f"\n{title}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"Direction Accuracy: {dir_acc:.4f}")


def save_results_table(output_path, baseline_metrics, baseline_dir_acc, proposed_metrics, proposed_dir_acc):
    """
    Save the final results into a CSV file.
    This makes it easy to copy into the report later.
    """

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Model,MAE,RMSE,Direction Accuracy\n")
        file.write(
            f"Baseline LSTM,{baseline_metrics['MAE']:.4f},"
            f"{baseline_metrics['RMSE']:.4f},{baseline_dir_acc:.4f}\n"
        )
        file.write(
            f"LSTM + Sentiment,{proposed_metrics['MAE']:.4f},"
            f"{proposed_metrics['RMSE']:.4f},{proposed_dir_acc:.4f}\n"
        )


def plot_predictions(y_actual, y_base, y_prop, save_path, ticker):
    """
    Plot actual prices vs both model predictions.
    The ticker name is added to the chart title automatically.
    """

    plt.figure(figsize=(10, 6))

    # Blue line = actual stock prices
    plt.plot(y_actual, label=f"{ticker} Actual Price")

    # Orange line = baseline model
    plt.plot(y_base, label=f"{ticker} Baseline Prediction")

    # Green line = proposed model
    plt.plot(y_prop, label=f"{ticker} Proposed Prediction")

    plt.title(f"{ticker} Stock Price Prediction Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.tight_layout()

    # Save image for the report
    plt.savefig(save_path)

    # Show chart on screen
    plt.show()


def run_experiment():
    """
    Main function for the whole project.

    Steps:
    1. Download stock data
    2. Create demo headlines
    3. Convert headlines to sentiment scores
    4. Train baseline model
    5. Train proposed model
    6. Compare results
    7. Save chart and CSV results
    """

    # Create folder for output files if it does not exist
    os.makedirs("outputs", exist_ok=True)

    # Stock ticker used in this experiment
    ticker = "AAPL"

    # Download stock data
    df = download_stock_data(ticker=ticker, start="2020-01-01", end="2024-01-01")

    # Create simple demo headlines
    df = create_demo_headlines(df)

    # Add sentiment scores based on the headlines
    df = add_sentiment_column(df)

    
    # BASELINE MODEL
    # This model uses stock data only
    

    baseline_features = ["Open", "High", "Low", "Close", "Volume"]

    X_base, y_base, _, base_target_scaler = prepare_sequences(
        df,
        feature_cols=baseline_features,
        target_col="Close",
        sequence_length=10
    )

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split_sequences(X_base, y_base)

    baseline_model = build_lstm_model((X_train_b.shape[1], X_train_b.shape[2]))

    baseline_model.fit(
        X_train_b,
        y_train_b,
        epochs=20,
        batch_size=16,
        verbose=1
    )

    y_pred_b = baseline_model.predict(X_test_b, verbose=0)

    # Convert scaled values back to real stock prices
    y_test_b_inv = base_target_scaler.inverse_transform(y_test_b)
    y_pred_b_inv = base_target_scaler.inverse_transform(y_pred_b)

    # Measure baseline performance
    baseline_metrics = calculate_metrics(y_test_b_inv, y_pred_b_inv)
    baseline_dir_acc = direction_accuracy(y_test_b_inv.flatten(), y_pred_b_inv.flatten())

    print_results("Baseline Results", baseline_metrics, baseline_dir_acc)


    # ==========================================================
    # ==========================================================
    # PROPOSED MODEL
    # This model uses stock data + sentiment
   

    proposed_features = ["Open", "High", "Low", "Close", "Volume", "Sentiment"]

    X_prop, y_prop, _, prop_target_scaler = prepare_sequences(
        df,
        feature_cols=proposed_features,
        target_col="Close",
        sequence_length=10
    )

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split_sequences(X_prop, y_prop)

    proposed_model = build_lstm_model((X_train_p.shape[1], X_train_p.shape[2]))

    proposed_model.fit(
        X_train_p,
        y_train_p,
        epochs=20,
        batch_size=16,
        verbose=1
    )

    y_pred_p = proposed_model.predict(X_test_p, verbose=0)

    # Convert scaled values back to real stock prices
    y_test_p_inv = prop_target_scaler.inverse_transform(y_test_p)
    y_pred_p_inv = prop_target_scaler.inverse_transform(y_pred_p)

    # Measure proposed model performance
    proposed_metrics = calculate_metrics(y_test_p_inv, y_pred_p_inv)
    proposed_dir_acc = direction_accuracy(y_test_p_inv.flatten(), y_pred_p_inv.flatten())

    print_results("Proposed Model Results", proposed_metrics, proposed_dir_acc)

    # Save the results to a CSV file
    save_results_table(
        "outputs/results.csv",
        baseline_metrics,
        baseline_dir_acc,
        proposed_metrics,
        proposed_dir_acc
    )

    # Save and show the prediction graph
    plot_predictions(
        y_test_p_inv,
        y_pred_b_inv,
        y_pred_p_inv,
        "outputs/prediction_plot.png",
        ticker
    )


if __name__ == "__main__":
    run_experiment()