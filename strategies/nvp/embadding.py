# embedding.py

import os
import sys
import asyncio
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from tensorflow.keras.optimizers import Adam
# Add your project path
sys.path.append("/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2")
from utilities.bitget_perp import PerpBitget
import ta
# Disable GPU and suppress TensorFlow messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generate_feature_vectors(df):
    """
    Generate feature vectors from OHLCV data including candlestick patterns,
    support and resistance levels, and technical indicators.
    """
    # Ensure the DataFrame has the necessary columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    # 1. Candlestick features
    df['price_range'] = df['high'] - df['low']
    df['price_change'] = df['close'] - df['open']
    df['direction'] = df['price_change'].apply(lambda x: 1 if x >= 0 else 0)  # Up or Down

    # Note: We are not scaling 'open', 'high', 'low', 'close' here
    # We will scale the entire feature vector later in the train_embedding function

    # 2. Support and Resistance levels
    prices = df['close'].values
    order = 5  # Adjust the order parameter as needed
    # Local minima (supports)
    local_min_indices = argrelextrema(prices, np.less_equal, order=order)[0]
    # Local maxima (resistances)
    local_max_indices = argrelextrema(prices, np.greater_equal, order=order)[0]

    df['min_price_resistance'] = np.nan
    df['max_price_resistance'] = np.nan

    last_min = np.nan
    last_max = np.nan
    for i in range(len(df)):
        if i in local_min_indices:
            last_min = prices[i]
        if i in local_max_indices:
            last_max = prices[i]
        df.at[i, 'min_price_resistance'] = last_min
        df.at[i, 'max_price_resistance'] = last_max

    # Fill NaN values
    df['min_price_resistance'] = df['min_price_resistance'].ffill()
    df['max_price_resistance'] = df['max_price_resistance'].ffill()

    # 3. Technical Indicators
    # Moving Average
    df['ma'] = ta.trend.sma_indicator(df['close'], window=14)
    # Exponential Moving Average
    df['ema'] = ta.trend.ema_indicator(df['close'], window=14)
    # Relative Strength Index
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    # Average True Range
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # Handle any remaining NaN values
    df = df.bfill()
    df = df.ffill()

    # Combine all features into a single DataFrame
    feature_columns = [
        'open', 'high', 'low', 'close',
        'price_range', 'price_change', 'direction',
        'min_price_resistance', 'max_price_resistance',
        'ma', 'ema', 'rsi', 'atr'
    ]

    # Create feature vectors
    feature_vectors = df[feature_columns].values
    
    return feature_vectors, feature_columns

async def main():
    # Initialize the exchange
    exchange = PerpBitget()
    await exchange.load_markets()

    # Select a single pair
    pair = 'BTC/USDT'

    # Fetch OHLCV data
    timeframe = '1h'
    limit = 10000
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)

    # Close the exchange session
    await exchange.close()

    # Generate feature vectors
    feature_vectors, feature_columns = generate_feature_vectors(df)

    # Define target embedding dimension
    embedding_dim = 16

    # Scale the feature vectors
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(feature_vectors)

    # Split data into training and validation sets
    x_train, x_val = train_test_split(feature_vectors_scaled, test_size=0.1, random_state=42)

    # Define autoencoder architecture
    input_dim = x_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(input_layer)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(embedding_dim, activation='relu')(encoded)
    # Decoder
    decoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(encoded)
    decoded = Dropout(0.3)(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    # Encoder model for embeddings
    encoder_model = Model(inputs=input_layer, outputs=encoded)

    # Decoder model to reconstruct from embeddings
    encoded_input = Input(shape=(embedding_dim,))
    decoder_layer1 = autoencoder.layers[-3](encoded_input)
    decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
    decoder_layer3 = autoencoder.layers[-1](decoder_layer2)
    decoder_model = Model(inputs=encoded_input, outputs=decoder_layer3)

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, min_lr=1e-10, verbose=1
    )
    # Train the autoencoder
    history = autoencoder.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=400,
        batch_size=32,
        shuffle=True,
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate reconstruction error
    reconstructed_vectors = autoencoder.predict(feature_vectors_scaled)
    reconstructed_vectors_inv = scaler.inverse_transform(reconstructed_vectors)
    mse = np.mean(np.square(feature_vectors - reconstructed_vectors_inv))
    print(f"Reconstruction Error (MSE): {mse}")

    # Save the encoder and decoder models, and the scaler
    encoder_model.save('best_encoder_model.keras')
    decoder_model.save('best_decoder_model.keras')
    joblib.dump(scaler, 'scaler.save')

    # Plot training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    asyncio.run(main())
