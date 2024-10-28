# btc_price_prediction_tft_corrected.py

import os
import sys
import numpy as np
import pandas as pd
import asyncio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import ta

import tensorflow as tf
from tensorflow.keras.models import load_model, Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Dense, LayerNormalization, MultiHeadAttention, Dropout,
    TimeDistributed, LSTM, Add, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

# Add your project path for additional modules
sys.path.append("/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2")
from strategies.nvp.predict_btc_tft import make_prediction
from utilities.bitget_perp import PerpBitget  # Adjust the import according to your project structure
from strategies.nvp.embadding import generate_feature_vectors

# ---------------------------- #
#        Hyperparameters       #
# ---------------------------- #

# Model Configuration
MAX_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1

# Data Configuration
SEQUENCE_LENGTH = 500  # Number of past time steps to look at
FORECAST_STEPS = 100   # Number of future time steps to predict

# Paths
MODEL_CHECKPOINT = 'tft_model_checkpoint_tts.keras'
SCALER_PATH = 'scaler.save'

# ---------------------------- #
#      Custom Layers           #
# ---------------------------- #

class PositionalEncoding(Layer):
    def __init__(self, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(embed_dim, tf.float32))
        return pos * angle_rates

    def positional_encoding(self, position, embed_dim):
        pos = tf.range(position, dtype=tf.float32)[:, tf.newaxis]  # Shape: (position, 1)
        i = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :]  # Shape: (1, embed_dim)
        angle_rads = self.get_angles(pos, i, embed_dim)  # Shape: (position, embed_dim)

        # Apply sin to even indices (2i) and cos to odd indices (2i+1)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Concatenate sines and cosines
        pos_encoding = tf.concat([sines, cosines], axis=-1)  # Shape: (position, embed_dim)

        # Add a batch dimension
        pos_encoding = pos_encoding[tf.newaxis, ...]  # Shape: (1, position, embed_dim)

        return pos_encoding

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.positional_encoding(seq_len, self.embed_dim)
        return inputs + pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"embed_dim": self.embed_dim})
        return config

custom_objects = {"PositionalEncoding": PositionalEncoding}

# ---------------------------- #
#      Utility Functions       #
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

async def fetch_latest_data(exchange, pair='BTC/USDT', timeframe='1m', limit=SEQUENCE_LENGTH + FORECAST_STEPS):
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
    print(df.index)
    df['timestamp'] = pd.to_datetime(df.index)

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
    for i in range(len(df_features) - SEQUENCE_LENGTH - FORECAST_STEPS + 1):
        X.append(features_scaled[i:i + SEQUENCE_LENGTH])
        y.append(df_target.iloc[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + FORECAST_STEPS].values)

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# ---------------------------- #
#       Define TFT Model       #
# ---------------------------- #

def build_tft_model(input_shape):
    """
    Builds the Temporal Fusion Transformer (TFT) model.

    Parameters:
    - input_shape: Tuple representing the shape of the input data (SEQUENCE_LENGTH, feature_dim).

    Returns:
    - model: Compiled Keras model.
    """
    # Inputs
    encoder_inputs = Input(shape=input_shape, name='encoder_inputs')  # (batch, SEQUENCE_LENGTH, feature_dim)
    decoder_inputs = Input(shape=(FORECAST_STEPS, input_shape[1]), name='decoder_inputs')  # (batch, FORECAST_STEPS, feature_dim)

    # Encoder
    encoder_lstm = LSTM(128, return_sequences=True, name='encoder_lstm')(encoder_inputs)  # (batch, SEQUENCE_LENGTH, 128)

    # Decoder
    decoder_lstm = LSTM(128, return_sequences=True, name='decoder_lstm')(
        decoder_inputs, initial_state=[encoder_lstm[:, -1, :], encoder_lstm[:, -1, :]]
    )  # (batch, FORECAST_STEPS, 128)

    # Attention Mechanism
    attention = MultiHeadAttention(num_heads=4, key_dim=64, dropout=DROPOUT_RATE, name='multihead_attention')
    attn_output = attention(query=decoder_lstm, value=encoder_lstm, key=encoder_lstm)  # (batch, FORECAST_STEPS, 128)
    attn_output = Dropout(DROPOUT_RATE)(attn_output)
    attn_output = Add()([attn_output, decoder_lstm])  # (batch, FORECAST_STEPS, 128)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)

    # Fully Connected Layers
    fc = TimeDistributed(Dense(64, activation='relu'))(attn_output)  # (batch, FORECAST_STEPS, 64)
    fc = Dropout(DROPOUT_RATE)(fc)
    outputs = TimeDistributed(Dense(1))(fc)  # (batch, FORECAST_STEPS, 1)

    # Model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs, name="TFT_Model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
    return model


# ---------------------------- #
#           Training           #
# ---------------------------- #

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Trains the TFT model.

    Parameters:
    - model: Compiled Keras model.
    - X_train: Training input data.
    - y_train: Training target data.
    - X_val: Validation input data.
    - y_val: Validation target data.

    Returns:
    - history: Training history.
    """
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    checkpoint = ModelCheckpoint(MODEL_CHECKPOINT, monitor='val_loss', save_best_only=True)

    # Prepare decoder inputs (zeros with shape (batch_size, FORECAST_STEPS, feature_dim=22))
    decoder_inputs_train = np.zeros((y_train.shape[0], FORECAST_STEPS, X_train.shape[2]))
    decoder_inputs_val = np.zeros((y_val.shape[0], FORECAST_STEPS, X_val.shape[2]))

    # Train the model using a dictionary for inputs
    history = model.fit(
        {'encoder_inputs': X_train, 'decoder_inputs': decoder_inputs_train},
        y_train,
        validation_data=({'encoder_inputs': X_val, 'decoder_inputs': decoder_inputs_val}, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    return history

# ---------------------------- #
#          Evaluation          #
# ---------------------------- #

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set.

    Parameters:
    - model: Trained Keras model.
    - X_test: Test input data.
    - y_test: Test target data.

    Returns:
    - y_pred_inv: Inverse scaled predictions.
    - y_test_inv: Inverse scaled actual values.
    """
    # Prepare decoder inputs (zeros with shape (batch_size, FORECAST_STEPS, feature_dim=22))
    decoder_inputs_test = np.zeros((y_test.shape[0], FORECAST_STEPS, X_test.shape[2]))
    
    # Make predictions
    y_pred = model.predict({'encoder_inputs': X_test, 'decoder_inputs': decoder_inputs_test})
    
    # Inverse transform predictions and targets
    scaler = joblib.load(SCALER_PATH)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1]))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1]))
    
    # Reshape back to (samples, FORECAST_STEPS, 1)
    y_pred_inv = y_pred_inv.reshape(y_pred.shape[0], FORECAST_STEPS, -1)
    y_test_inv = y_test_inv.reshape(y_test.shape[0], FORECAST_STEPS, -1)
    
    # Compute metrics
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f"Test MAE: {mae}")
    print(f"Test MSE: {mse}")
    
    return y_pred_inv, y_test_inv

# ---------------------------- #
#        Visualization         #
# ---------------------------- #

def plot_predictions(y_test_inv, y_pred_inv):
    """
    Plots the actual and predicted BTC prices.

    Parameters:
    - y_test_inv: Inverse scaled actual BTC prices.
    - y_pred_inv: Inverse scaled predicted BTC prices.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inv.flatten()[:100], label='Actual BTC Prices', color='blue')
    plt.plot(range(len(y_test_inv.flatten()[:100]), len(y_test_inv.flatten()[:100]) + len(y_pred_inv.flatten()[:100])),
             y_pred_inv.flatten()[:100], label='Predicted BTC Prices', color='orange')
    plt.title('BTC Price Prediction vs. Actual')
    plt.xlabel('Time Steps (1 Minute Each)')
    plt.ylabel('BTC Price (USDT)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# ---------------------------- #
#           Main Flow          #
# ---------------------------- #

async def main():
    # ---------------------------- #
    #      Load Model and Scaler    #
    # ---------------------------- #
    print("Loading the trained model and scaler...")

    if not os.path.exists(MODEL_CHECKPOINT):
        # Build a new model if no checkpoint is found
        print("Model checkpoint not found. Building a new model...")
        input_shape = (SEQUENCE_LENGTH, 22)  # Example shape (SEQUENCE_LENGTH, feature_dim=22)
        model = build_tft_model(input_shape)
        print("New model built successfully.")
    else:
        # Load the existing model and scaler
        print("Model checkpoint found. Loading the existing model...")
        model = load_model(MODEL_CHECKPOINT, custom_objects=custom_objects)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        

    print("Model and scaler loaded successfully.")

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
    current_sequence = X[-1]  # Shape: (SEQUENCE_LENGTH, feature_dim=22)
    decoder_input = np.zeros((1, FORECAST_STEPS, SEQUENCE_LENGTH))  # This is incorrect
    predictions_inv = make_prediction(model, scaler, current_sequence)
    print(f"Predictions shape: {predictions_inv.shape}")

    # ---------------------------- #
    #      Fetch Actual Future Data #
    # ---------------------------- #
    # Note: To evaluate predictions, you need the actual future data.
    # This requires fetching the next FORECAST_STEPS data points after the current_sequence.
    # If the exchange API allows fetching data beyond the current time, implement it here.
    # Otherwise, skip this step or use a holdout test set from training.

    # Example (may not work depending on exchange API):
    """
    print(f"Fetching the next {FORECAST_STEPS} minutes of data for BTC/USDT...")
    df_future = await exchange.get_last_ohlcv('BTC/USDT', '1m', FORECAST_STEPS)
    actual_future = df_future.iloc[:FORECAST_STEPS][feature_columns].values
    actual_future_inv = scaler.inverse_transform(actual_future)
    print("Actual future data fetched.")
    """

    # For demonstration purposes, assume actual_future_inv is available.
    # If not, you can comment out the evaluation section.

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
    # Here, you can plot historical data along with predictions if available
    plot_predictions(predictions_inv.flatten(), predictions_inv.flatten())  # Adjust as needed

    # ---------------------------- #
    #      Print Sample Data        #
    # ---------------------------- #
    print("\nSample of Predicted Values:")
    print(predictions_inv[:5, 0])


# ---------------------------- #
#           Execution           #
# ---------------------------- #

if __name__ == "__main__":
    asyncio.run(main())
