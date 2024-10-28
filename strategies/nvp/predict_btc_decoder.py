# improved_btc_prediction.py

import os
import sys
import numpy as np
import pandas as pd
import asyncio
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import ta

# Add your project path for additional modules
sys.path.append("/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2")
from utilities.bitget_perp import PerpBitget
from strategies.nvp.embadding import generate_feature_vectors
from strategies.nvp.decoder_only import custom_objects

# Plot the smoothed data
def plot_smoothed_data(df_smoothed, feature_index=0, label='$BTC'):
    """
    Plot the smoothed features to inspect the data.

    Parameters:
    - df_smoothed: Numpy array containing the smoothed and scaled feature vectors.
    - feature_index: Index of the feature to plot (default is 0 for 'close' price).
    - label: Label for the plot legend.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(df_smoothed)), df_smoothed[:, feature_index], label=f'{label} Price')
    plt.title('Smoothed Features Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Feature Values')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

async def main():
    # ---------------------------- #
    #      Load Model and Scaler    #
    # ---------------------------- #
    print("Loading the trained model...")
    model = load_model('transformer_model_checkpoint_decoder_only.keras', custom_objects=custom_objects)
    print("Model loaded successfully.")

    print("Loading the scaler...")
    scaler = joblib.load('scaler.save')
    print("Scaler loaded successfully.")

    # ---------------------------- #
    #      Initialize Exchange     #
    # ---------------------------- #
    print("Initializing the exchange...")
    exchange = PerpBitget()
    await exchange.load_markets()
    print("Exchange initialized.")

    # ---------------------------- #
    #      Fetch BTC Data          #
    # ---------------------------- #
    pair = 'BTC/USDT'
    timeframe = '1m'
    limit = 1200 # Number of data points to fetch
    print(f"Fetching the last {limit} minutes of data for {pair}...")
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)
    await exchange.close()
    print("Data fetched and exchange session closed.")

    # ---------------------------- #
    #    Feature Engineering       #
    # ---------------------------- #
    print("Generating feature vectors...")
    feature_vectors, feature_columns = generate_feature_vectors(df)
    feature_vectors_df = pd.DataFrame(feature_vectors, columns=feature_columns)
    print("Feature vectors generated.")

    print("Applying rolling mean smoothing...")
    # df_smoothed = feature_vectors_df.rolling(window=2).mean().dropna()
    df_smoothed = feature_vectors
    print("Smoothing applied.")

    # ---------------------------- #
    #      Scaling the Data        #
    # ---------------------------- #
    print("Scaling the feature vectors...")
    # scaler is already loaded; no need to fit again
    feature_vectors_scaled = scaler.transform(df_smoothed)
    print("Feature vectors scaled.")

    # ---------------------------- #
    #      Prepare Input Data       #
    # ---------------------------- #
    SEQUENCE_LENGTH = 500
    OUTPUT_STEPS = 100

    # Ensure there are enough data points
    if feature_vectors_scaled.shape[0] < SEQUENCE_LENGTH + OUTPUT_STEPS:
        raise ValueError(f"Not enough data points. Required: {SEQUENCE_LENGTH + OUTPUT_STEPS}, Available: {feature_vectors_scaled.shape[0]}")

    # Extract the last SEQUENCE_LENGTH data points as input
    current_sequence = feature_vectors_scaled[-SEQUENCE_LENGTH:]
    print(f"Current sequence shape: {current_sequence.shape}")  # Should be (500, feature_dim)

    # Extract the actual next OUTPUT_STEPS data points for comparison
    actual_future = feature_vectors_scaled[-(SEQUENCE_LENGTH + OUTPUT_STEPS):-SEQUENCE_LENGTH]
    actual_future_inv = scaler.inverse_transform(actual_future)
    print(f"Actual future data shape: {actual_future_inv.shape}")  # Should be (100, feature_dim)

    # ---------------------------- #
    #        Make Predictions       #
    # ---------------------------- #
    print("Making predictions...")
    all_predictions = []

    # Since the decoder-only model can predict OUTPUT_STEPS in one go, we can predict once
    # Reshape current_sequence to match model input: (1, SEQUENCE_LENGTH, feature_dim)
    input_seq = current_sequence[np.newaxis, :, :]  # Shape: (1, 500, feature_dim)
    prediction = model.predict(input_seq)  # Shape: (1, 100, feature_dim)
    all_predictions.append(prediction[0])  # Append the (100, feature_dim) array

    all_predictions = np.vstack(all_predictions)  # Shape: (100, feature_dim)
    print(f"All predictions shape: {all_predictions.shape}")

    # ---------------------------- #
    #     Inverse Transform Data    #
    # ---------------------------- #
    print("Inverse transforming the predictions...")
    all_predictions_inv = scaler.inverse_transform(all_predictions)
    print("Inverse transformation complete.")

    # ---------------------------- #
    #      Compute Error Metrics    #
    # ---------------------------- #
    print("Computing error metrics...")
    # Assuming the 'close' price is at index 0
    mae = mean_absolute_error(actual_future_inv[:, 0], all_predictions_inv[:, 0])
    mse = mean_squared_error(actual_future_inv[:, 0], all_predictions_inv[:, 0])
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

    # ---------------------------- #
    #           Plotting            #
    # ---------------------------- #
    print("Plotting the results...")
    plt.figure(figsize=(14, 7))
    plt.plot(range(SEQUENCE_LENGTH), scaler.inverse_transform(current_sequence)[:, 0], label='Historical BTC Prices', color='blue')
    plt.plot(range(SEQUENCE_LENGTH, SEQUENCE_LENGTH + OUTPUT_STEPS), all_predictions_inv[:, 0], label='Predicted BTC Prices', color='orange')
    plt.plot(range(SEQUENCE_LENGTH, SEQUENCE_LENGTH + OUTPUT_STEPS), actual_future_inv[:, 0], label='Actual BTC Prices', color='green', linestyle='--')
    plt.title('BTC Price Prediction vs. Actual')
    plt.xlabel('Time Steps (1 Minute Each)')
    plt.ylabel('BTC Price (USDT)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # ---------------------------- #
    #      Print Sample Data        #
    # ---------------------------- #
    print("\nSample of Actual Future Values:")
    print(actual_future_inv[:5, 0])

    print("\nSample of Predicted Values:")
    print(all_predictions_inv[:5, 0])


if __name__ == '__main__':
    asyncio.run(main())
