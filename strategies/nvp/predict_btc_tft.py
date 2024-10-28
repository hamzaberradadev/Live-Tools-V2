# btc_predict_test.py

import os
import sys
import numpy as np
import pandas as pd
import asyncio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import ta

# ---------------------------- #
#          Configuration        #
# ---------------------------- #

# Hyperparameters
SEQUENCE_LENGTH = 500
OUTPUT_STEPS = 100

# Paths
MODEL_PATH = "transformer_model_checkpoint_tts.keras"
SCALER_PATH = "scaler.save"

# Add your project path for additional modules
sys.path.append("/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2")
from utilities.bitget_perp import PerpBitget
from strategies.nvp.embadding import generate_feature_vectors

# ---------------------------- #
#        Utility Functions      #
# ---------------------------- #

def plot_predictions(actual, predicted, title='BTC Price Prediction vs. Actual'):
    """
    Plots the actual and predicted BTC prices.
    
    Parameters:
    - actual: Array of actual BTC prices.
    - predicted: Array of predicted BTC prices.
    - title: Title of the plot.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(actual)), actual, label='Actual BTC Prices', color='blue')
    plt.plot(range(len(actual), len(actual) + len(predicted)), predicted, label='Predicted BTC Prices', color='orange')
    plt.title(title)
    plt.xlabel('Time Steps (1 Minute Each)')
    plt.ylabel('BTC Price (USDT)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

async def fetch_latest_data(exchange, pair='BTC/USDT', timeframe='1m', limit=SEQUENCE_LENGTH + OUTPUT_STEPS):
    """
    Fetches the latest OHLCV data for a given trading pair.
    
    Parameters:
    - exchange: An instance of PerpBitget or your exchange interface.
    - pair: Trading pair symbol.
    - timeframe: Timeframe for OHLCV data.
    - limit: Number of data points to fetch.
    
    Returns:
    - DataFrame containing OHLCV data.
    """
    print(f"Fetching the last {limit} minutes of data for {pair}...")
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)
    await exchange.close()
    print("Data fetched and exchange session closed.")
    return df

def generate_features(df):
    """
    Generates additional features from OHLCV data.
    
    Parameters:
    - df: DataFrame containing OHLCV data.
    
    Returns:
    - DataFrame with additional features.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Technical Indicators
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['stochastic'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    
    # Lag features
    for lag in range(1, 4):
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    # Rolling window statistics
    df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
    df['rolling_std_5'] = df['close'].rolling(window=5).std()
    
    # Fill missing values
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    return df

def preprocess_data(df, scaler=None):
    """
    Preprocesses the data by selecting features, scaling, and creating sequences.
    
    Parameters:
    - df: DataFrame with features.
    - scaler: Optional, a pre-fitted scaler. If None, fits a new scaler.
    
    Returns:
    - X: Input sequences.
    - y: Target sequences.
    - scaler: Fitted scaler.
    """
    df = df.copy()
    
    # Select features and target
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'hour', 'day_of_week', 'is_weekend',
        'macd', 'rsi', 'stochastic',
        'bollinger_mavg', 'bollinger_hband', 'bollinger_lband',
        'close_lag_1', 'close_lag_2', 'close_lag_3',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
        'rolling_mean_5', 'rolling_std_5',
    ]
    target = ['close']
    
    df_features = df[features]
    df_target = df[target]
    
    # Scaling
    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_features)
    else:
        features_scaled = scaler.transform(df_features)
    
    # Prepare sequences
    X, y = [], []
    for i in range(len(features_scaled) - SEQUENCE_LENGTH - OUTPUT_STEPS + 1):
        X.append(features_scaled[i:i + SEQUENCE_LENGTH])
        y.append(df_target.iloc[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + OUTPUT_STEPS].values)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# ---------------------------- #
#         Model Loading         #
# ---------------------------- #

def load_trained_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """
    Loads the trained model and scaler.
    
    Parameters:
    - model_path: Path to the saved Keras model.
    - scaler_path: Path to the saved scaler.
    
    Returns:
    - model: Loaded Keras model.
    - scaler: Loaded scaler.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    print("Loading the trained model...")
    model = load_model(model_path, custom_objects={"PositionalEncoding": tf.keras.layers.Layer})
    print("Model loaded successfully.")
    
    print("Loading the scaler...")
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
    
    return model, scaler

# ---------------------------- #
#      Prediction Function      #
# ---------------------------- #

def make_prediction(model, scaler, current_sequence):
    """
    Makes predictions using the trained model.
    
    Parameters:
    - model: Trained Keras model.
    - scaler: Fitted scaler.
    - current_sequence: Latest sequence data for prediction.
    
    Returns:
    - predictions_inv: Inverse scaled predictions.
    """
    # Reshape to match model input: (1, SEQUENCE_LENGTH, feature_dim)
    input_seq = current_sequence[np.newaxis, :, :]
    
    # Initialize decoder input as zeros
    decoder_input = np.zeros((1, OUTPUT_STEPS, current_sequence.shape[1]))
    
    # Make prediction
    prediction = model.predict([input_seq, decoder_input])  # Shape: (1, OUTPUT_STEPS, feature_dim)
    
    # Extract predictions
    predictions = prediction[0]  # Shape: (OUTPUT_STEPS, feature_dim)
    
    # Inverse transform
    predictions_inv = scaler.inverse_transform(predictions)
    
    return predictions_inv

# ---------------------------- #
#        Evaluation Function    #
# ---------------------------- #

def evaluate_predictions(actual, predicted):
    """
    Evaluates the predictions using MAE and MSE.
    
    Parameters:
    - actual: Actual BTC prices.
    - predicted: Predicted BTC prices.
    
    Returns:
    - mae: Mean Absolute Error.
    - mse: Mean Squared Error.
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    return mae, mse

# ---------------------------- #
#          Main Function        #
# ---------------------------- #

async def main():
    try:
        # ---------------------------- #
        #      Load Model and Scaler    #
        # ---------------------------- #
        model, scaler = load_trained_model()
    
        # ---------------------------- #
        #      Initialize Exchange     #
        # ---------------------------- #
        print("Initializing the exchange...")
        exchange = PerpBitget()
        await exchange.load_markets()
        print("Exchange initialized.")
    
        # ---------------------------- #
        #      Fetch Latest BTC Data    #
        # ---------------------------- #
        df_latest = await fetch_latest_data(exchange)
    
        # ---------------------------- #
        #    Feature Engineering       #
        # ---------------------------- #
        print("Generating feature vectors...")
        df_features = generate_features(df_latest)
        print("Feature vectors generated.")
    
        # ---------------------------- #
        #      Preprocess Data          #
        # ---------------------------- #
        print("Preprocessing data...")
        X, y, _ = preprocess_data(df_features, scaler=scaler)
        print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
        # ---------------------------- #
        #    Make Predictions           #
        # ---------------------------- #
        print("Making predictions...")
        # Use the last SEQUENCE_LENGTH data points
        current_sequence = X[-1]  # Shape: (SEQUENCE_LENGTH, feature_dim)
        predictions_inv = make_prediction(model, scaler, current_sequence)
        print(f"Predictions shape: {predictions_inv.shape}")
    
        # ---------------------------- #
        #      Fetch Actual Future Data #
        # ---------------------------- #
        # Note: To evaluate predictions, you need the actual future data.
        # This requires fetching the next OUTPUT_STEPS data points after the current_sequence.
        # If the exchange API allows fetching data beyond the current time, implement it here.
        # Otherwise, skip this step or use a holdout test set from training.
    
        # Example (may not work depending on exchange API):
        """
        print(f"Fetching the next {OUTPUT_STEPS} minutes of data for {pair}...")
        df_future = await exchange.get_last_ohlcv(pair, timeframe, OUTPUT_STEPS)
        actual_future = df_future.iloc[:OUTPUT_STEPS][feature_columns].values
        actual_future_inv = scaler.inverse_transform(actual_future)
        print("Actual future data fetched.")
        """
    
        # For demonstration purposes, assume actual_future_inv is available.
        # If not, you can comment out the evaluation section.
        # Replace the following lines with actual future data fetching if possible.
    
        # ---------------------------- #
        #      Compute Error Metrics    #
        # ---------------------------- #
        """
        print("Computing error metrics...")
        mae, mse = evaluate_predictions(actual_future_inv[:, 0], predictions_inv[:, 0])
        """
        print("Evaluation skipped due to lack of actual future data.")
    
        # ---------------------------- #
        #           Plotting            #
        # ---------------------------- #
        # If actual_future_inv is available, plot it alongside predictions.
        # Otherwise, plot predictions only or historical data.
    
        # Example plotting with actual future data:
        """
        print("Plotting the results...")
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(actual_future_inv)), actual_future_inv[:, 0], label='Actual BTC Prices', color='green')
        plt.plot(range(len(predictions_inv)), predictions_inv[:, 0], label='Predicted BTC Prices', color='orange')
        plt.title('BTC Price Prediction vs. Actual')
        plt.xlabel('Time Steps (1 Minute Each)')
        plt.ylabel('BTC Price (USDT)')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
        """
    
        # Example plotting without actual future data:
        print("Plotting the predictions...")
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(predictions_inv)), predictions_inv[:, 0], label='Predicted BTC Prices', color='orange')
        plt.title('BTC Price Predictions')
        plt.xlabel('Time Steps (1 Minute Each)')
        plt.ylabel('BTC Price (USDT)')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
    
        # ---------------------------- #
        #      Print Sample Data        #
        # ---------------------------- #
        print("\nSample of Predicted Values:")
        print(predictions_inv[:5, 0])
    
    except Exception as e:
        print(f"An error occurred: {e}")

# ---------------------------- #
#           Execution           #
# ---------------------------- #

if __name__ == "__main__":
    asyncio.run(main())
