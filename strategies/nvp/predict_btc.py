# predict_btc.py

import datetime
import os
import sys
import numpy as np
import pandas as pd
import asyncio
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras
keras.config.enable_unsafe_deserialization()
# Add your project path
sys.path.append("/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2")
from utilities.bitget_perp import PerpBitget
from strategies.nvp.embadding import generate_feature_vectors

pair = 'BTC/USDT'

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def plot_predictions(y_true, y_pred, scaler, decoder_model, feature_index=3):
    # Decode embeddings to full feature vectors
    y_true_decoded = decoder_model.predict(y_true)
    y_pred_decoded = decoder_model.predict(y_pred)

    # Inverse transform to original scale
    y_true_inv = scaler.inverse_transform(y_true_decoded)
    y_pred_inv = scaler.inverse_transform(y_pred_decoded)

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
    # Load the transformer model, decoder model, and scaler
    model = load_model('transformer_model.keras')
    decoder_model = load_model('best_decoder_model.keras')
    scaler = joblib.load('scaler.save')

    # Initialize the exchange
    exchange = PerpBitget()
    await exchange.load_markets()

    # Select the BTC/USDT pair
 

    # Fetch the latest OHLCV data for BTC
    timeframe = '1h'
    limit = 5000
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)
    print(df.columns)
    await exchange.close()

    # Check if 'timestamp' column exists, create one if not
    if 'timestamp' not in df.columns:
        # Create a timestamp starting from current time, decrementing by 1 hour
        now = datetime.datetime.now()
        timestamps = [now - datetime.timedelta(hours=(len(df) - i - 1)) for i in range(len(df))]
        df['timestamp'] = timestamps
    else:
        # Convert 'timestamp' column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set 'timestamp' as index
    df.set_index('timestamp', inplace=True)

    # Plot raw data from the exchange
    plot_raw_data(df, feature_column='close')
    # Close the exchange session

    # Generate feature vectors
    feature_vectors, _ = generate_feature_vectors(df)

    # Scaling
    feature_vectors_scaled = scaler.transform(feature_vectors)

    # Generate embeddings using the encoder
    encoder_model = load_model('best_encoder_model.keras')
    embeddings = encoder_model.predict(feature_vectors_scaled)

    # Define sequence length
    sequence_length = 150  # Should match the sequence length used during training

    # Create sequences of embeddings
    x = create_sequences(embeddings, sequence_length)

    # Make predictions (embeddings)
    predictions = model.predict(x)

    # Align actual embeddings
    y_true = embeddings[sequence_length - 1:]

    # Plot the real vs. predicted data
    plot_predictions(y_true, predictions, scaler, decoder_model, feature_index=3)  # Assuming index 3 is 'close' price

if __name__ == '__main__':
    asyncio.run(main())
