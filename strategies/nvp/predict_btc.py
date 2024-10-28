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
from strategies.nvp.transformer_model import custom_objects

# Plot the smoothed data
def plot_smoothed_data(df_smoothed):
    """
    Plot the smoothed features to inspect the data.

    Parameters:
    - df_smoothed: DataFrame containing the smoothed feature vectors.
    """
    plt.plot(range(len(df_smoothed)), df_smoothed[:, 0], label='$')

    plt.title('Smoothed Features Over Time')
    plt.xlabel('Time')
    plt.ylabel('Feature Values')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

async def main():
    # Load the model and scaler
    model = load_model('transformer_model_checkpoint.keras', custom_objects=custom_objects)
    scaler = joblib.load('scaler.save')

    # Initialize the exchange
    exchange = PerpBitget()
    await exchange.load_markets()

    # Fetch the latest BTC/USDT data
    pair = 'BTC/USDT'
    timeframe = '1m'
    limit = 1000  # Number of data points to fetch
    # Retrieve data and preprocess
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)
    await exchange.close()

    # Plot original data for comparison
    # plot_smoothed_data(df[['open']].values)

    # Generate feature vectors
    feature_vectors, feature_columns = generate_feature_vectors(df)
    # plot_smoothed_data(feature_vectors)

    # Convert feature vectors to DataFrame for further processing
    feature_vectors_df = pd.DataFrame(feature_vectors, columns=feature_columns)

    # Apply rolling mean smoothing and drop NaN values
    df_smoothed = feature_vectors_df.rolling(window=1).mean().dropna()

    # Debug: Check the smoothed data
    print("Smoothed Data Sample:\n", df_smoothed.head())

    # Plot the smoothed data
    # plot_smoothed_data(df_smoothed[feature_columns].values)

    # Scale the feature vectors after fitting the scaler on smoothed data
    scaler.fit(df_smoothed)
    feature_vectors_scaled = scaler.transform(df_smoothed)

    # Now check if the smoothed and scaled data still looks appropriate
    # plot_smoothed_data(feature_vectors_scaled)

    # Define input sequence length and output steps
    SEQUENCE_LENGTH = 500
    OUTPUT_STEPS = 50

    # Predict the next 300 points recurrently
    all_predictions = []
    current_sequence = feature_vectors_scaled[-SEQUENCE_LENGTH:].copy()
    # plot_smoothed_data(current_sequence)
    print(current_sequence.shape)
    for _ in range(SEQUENCE_LENGTH // OUTPUT_STEPS):
        # Prepare decoder input (shifted version of last prediction)
        decoder_input = np.zeros((1, OUTPUT_STEPS, current_sequence.shape[1]))
        
        # Make a prediction
        prediction = model.predict([current_sequence[np.newaxis, :, :], decoder_input])
        
        # Append prediction to results
        all_predictions.append(prediction[0])
        
        # Update the current sequence with the latest prediction
        current_sequence = np.vstack((current_sequence[-(SEQUENCE_LENGTH - OUTPUT_STEPS):], prediction[0]))

    # Combine all predictions
    all_predictions = np.vstack(all_predictions)

    # Debug: Check the prediction shape
    print(f"All predictions shape: {all_predictions.shape}")

    # Inverse transform the predictions to original scale
    all_predictions_inv = scaler.inverse_transform(all_predictions)
    actual_data_inv = scaler.inverse_transform(feature_vectors_scaled[-SEQUENCE_LENGTH:].copy())

    # Calculate the difference between the last actual data point and the first prediction point
    first_prediction_value = all_predictions_inv[0, 0]  # Adjust index for "close" as needed
    last_actual_value = actual_data_inv[-1, 0]  # Adjust index for "close" as needed
    difference = last_actual_value - first_prediction_value

    # Apply the difference to adjust the predictions
    all_predictions_inv += difference

    # Concatenate the last 300 actual points with the predictions for a continuous plot
    combined_data = np.concatenate((actual_data_inv, all_predictions_inv), axis=0)

    # Calculate error metrics only for the prediction part
    mae = mean_absolute_error(actual_data_inv[:, 0], all_predictions_inv[:, 0])  # Use the appropriate index for "close"
    mse = mean_squared_error(actual_data_inv[:, 0], all_predictions_inv[:, 0])  # Use the appropriate index for "close"

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Plot the results with detailed checks
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(combined_data)), combined_data[:, 0], label='Actual and Predicted BTC Prices', color='blue')  # Adjust feature index as needed
    plt.axvline(x=500, color='red', linestyle='--', label='Prediction Start')
    plt.title('BTC Price Prediction vs. Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('BTC Price')
    plt.legend()

    # Annotate the difference on the plot
    plt.annotate(f'Adjustment Applied: {difference:.2f}', xy=(SEQUENCE_LENGTH, all_predictions_inv[0, 0]), 
                 xytext=(SEQUENCE_LENGTH+OUTPUT_STEPS, all_predictions_inv[0, 0] + 500),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12, color='red')

    plt.grid()
    plt.show()

    # Debug: Print the last few values of actual and predicted data
    print("Last few actual values:\n", actual_data_inv[-5:, 0])
    print("Last few adjusted predicted values:\n", all_predictions_inv[-5:, 0])

if __name__ == '__main__':
    asyncio.run(main())
