# predict_btc.py

import os
import sys
import numpy as np
import pandas as pd
import asyncio
import datetime
from tensorflow.keras.models import load_model # type: ignore
from tensorflow import keras
keras.config.enable_unsafe_deserialization()
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import ta

# Add your project path
sys.path.append("/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2")
from utilities.bitget_perp import PerpBitget
from strategies.nvp.embadding import generate_feature_vectors
from strategies.nvp.transformer_model import generate_additional_features, PositionalEncoding

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)
import numpy as np

def smooth_and_rescale_predictions(predictions, actual, sequence_length=150, window_size=5):
    """
    Smooths the predictions using a moving average and rescales them to align with the actual values.
    
    Parameters:
    - predictions (array-like): The predicted values.
    - actual (array-like): The actual values corresponding to the predictions.
    - sequence_length (int): The length of the initial sequence to use for rescaling.
    - window_size (int): The window size for the moving average smoothing.
    
    Returns:
    - smoothed_rescaled_predictions (array-like): The smoothed and rescaled predictions.
    """
    # Step 1: Smooth the predictions using a moving average
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    smoothed_predictions = moving_average(predictions, window_size)
    
    # Step 2: Rescale the predictions to align with the actual values over the first sequence_length
    if len(actual) < sequence_length or len(smoothed_predictions) < sequence_length:
        raise ValueError("The sequence length for rescaling is larger than the available data.")
    
    # Calculate the scaling factor based on the ratio of actual to predicted over the first sequence_length
    scaling_factor = np.mean(actual[:sequence_length]) / np.mean(smoothed_predictions[:sequence_length])
    
    # Apply the scaling factor to the smoothed predictions
    smoothed_rescaled_predictions = smoothed_predictions * scaling_factor
    
    # Pad the rescaled predictions to match the original length (since smoothing reduces the array size)
    padding_length = len(predictions) - len(smoothed_rescaled_predictions)
    smoothed_rescaled_predictions = np.pad(smoothed_rescaled_predictions, (padding_length, 0), mode='edge')
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(smoothed_rescaled_predictions, label='Smoothed and Rescaled Predicted', color='orange')
    plt.title('BTC Actual vs. Smoothed and Rescaled Predicted Close Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_predictions(y_true, y_pred, scaler, feature_index=3):
    """
    Plot the actual vs. predicted values for a specific feature over multiple time steps.

    Parameters:
    - y_true: Actual values, shape (num_samples, output_steps, num_features)
    - y_pred: Predicted values, shape (num_samples, output_steps, num_features)
    - scaler: Scaler used to normalize the data.
    - feature_index: Index of the feature to plot (e.g., 3 for 'close' price).
    """

    # Reshape to 2D arrays
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    # Inverse transform to original scale
    y_true_inv = scaler.inverse_transform(y_true_flat)
    y_pred_inv = scaler.inverse_transform(y_pred_flat)

    y_true_feature = y_true_inv[:, feature_index]
    y_pred_feature = y_pred_inv[:, feature_index]

    plt.figure(figsize=(12, 6))
    plt.plot(y_true_feature, label='Actual')
    plt.plot(y_pred_feature, label='Predicted')
    plt.title('BTC Actual vs. Predicted Close Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_raw_data(df, feature_column='close'):
    """
    Plot the raw data from the exchange for a specific feature.

    Parameters:
    - df: DataFrame containing the raw data.
    - feature_column: The column to plot (default is 'close').
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[feature_column], label='Raw Data')
    plt.title(f'BTC {feature_column.capitalize()} Price from Exchange')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

async def main():
    # Load the transformer model and scaler
    model = load_model('transformer_model_checkpoint.keras', custom_objects={'PositionalEncoding': PositionalEncoding})
    scaler = joblib.load('scaler.save')

    # Initialize the exchange
    exchange = PerpBitget()
    await exchange.load_markets()

    # Select the BTC/USDT pair
    pair = 'BTC/USDT'

    # Fetch the latest OHLCV data for BTC
    timeframe = '1h'
    limit = 500
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)
    await exchange.close()

    # Check if 'timestamp' column exists, create one if not
    if "timestamp" not in df.columns:
        # Create a timestamp starting from current time, decrementing by 1 hour
        now = datetime.datetime.now()
        timestamps = [
            now - datetime.timedelta(hours=(len(df) - i - 1)) for i in range(len(df))
        ]
        df["timestamp"] = timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
    df.index = df["timestamp"]
    # Generate feature vectors
    feature_vectors, feature_columns = generate_feature_vectors(df)

    # Generate additional features
    df = generate_additional_features(df)
    additional_features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "macd",
        "stochastic",
        "bollinger_mavg",
        "bollinger_hband",
        "bollinger_lband",
    ]
    additional_feature_vectors = df[additional_features].values

    # Combine original and additional features
    feature_vectors = np.concatenate(
        (feature_vectors, additional_feature_vectors), axis=1
    )
    # Scaling
    feature_vectors_scaled = scaler.transform(feature_vectors)
    # Define sequence length and output steps
    output_steps = 5  # Number of future steps predicted

    # Create sequences
    sequence_length = 150
    x = []
    for i in range(len(feature_vectors_scaled) - sequence_length - 5 + 1):
        x.append(feature_vectors_scaled[i : i + sequence_length])
    x = np.array(x)
    predictions = model.predict(x)
    # Get the actual values for comparison
    y_true = []
    for i in range(sequence_length, len(feature_vectors_scaled) - output_steps + 1):
        y_true.append(feature_vectors_scaled[i : i + output_steps])
    y_true = np.array(y_true)

    # Ensure y_true and predictions have the same length
    min_len = min(len(y_true), len(predictions))
    y_true = y_true[:min_len]
    predictions = predictions[:min_len]
    
    # Reshape predictions and y_true to 2D
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
    
        # Inverse transform to original scale
    y_true_inv = scaler.inverse_transform(y_true_flat)
    predictions_inv = scaler.inverse_transform(predictions_flat)

    # Extract the 'close' prices (assuming 'close' is at index 3)
    y_true_close = y_true_inv[:, 3]
    predictions_close = predictions_inv[:, 3]

    # Smooth and rescale predictions
    smooth_and_rescale_predictions(predictions_close, y_true_close, sequence_length=150, window_size=5)


if __name__ == '__main__':
    asyncio.run(main())
