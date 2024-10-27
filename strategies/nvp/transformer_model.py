import datetime
import os
import sys
import numpy as np
import asyncio
import tensorflow as tf
# Disable GPU and suppress TensorFlow messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.regularizers import l1_l2  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Lambda,
    Layer,
    BatchNormalization,
    Flatten,
    Reshape,
    Concatenate,
)
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint  # type: ignore
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import ta
import pandas as pd

COMPLETE_TRAINING = False
STARTING_LR = 0.001
# Add your project path
sys.path.append(
    "/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2"
)
from utilities.bitget_perp import PerpBitget
from strategies.nvp.embadding import generate_feature_vectors
checkpoint_path = "transformer_model_checkpoint.keras"
scaler_path = "scaler.save"

from tensorflow import keras

# Register the custom layer with Keras serialization
class PositionalEncoding(Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_encoding = self.positional_encoding(maxlen, embed_dim)

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def positional_encoding(self, position, embed_dim):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(embed_dim)[np.newaxis, :],
            embed_dim,
        )

        # Apply sin to even indices (2i)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices (2i+1)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        # Return the configuration of the layer, including maxlen and embed_dim
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate the layer from the configuration
        return cls(**config)


# Data Augmentation Function
def augment_data(x, y, noise_factor=0.01):
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    y_noisy = y + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=y.shape)
    return x_noisy, y_noisy


def scheduler(epoch, lr):
    if epoch > 10:
        # Gradually increase the learning rate for the first 10 epochs
        return float(lr * 0.9)
    else:
        # Continue with the current learning rate
        return float(lr)


lr_scheduler = LearningRateScheduler(scheduler)


# Generate Additional Features
def generate_additional_features(df):
    # Add time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Add additional technical indicators
    df["macd"] = ta.trend.macd_diff(df["close"])
    df["stochastic"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
    bollinger = ta.volatility.BollingerBands(df["close"])
    df["bollinger_mavg"] = bollinger.bollinger_mavg()
    df["bollinger_hband"] = bollinger.bollinger_hband()
    df["bollinger_lband"] = bollinger.bollinger_lband()

    # Fill NaNs
    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df


async def main():
    # Initialize the exchange
    exchange = PerpBitget()
    await exchange.load_markets()

    # Select a single pair
    pair = "BTC/USDT"

    # Fetch OHLCV data
    timeframe = "1m"
    limit = 10000
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)

    # Close the exchange session
    await exchange.close()
    # Check if 'timestamp' column exists, create one if not
    # if "timestamp" not in df.columns:
    #     # Create a timestamp starting from current time, decrementing by 1 hour
    #     now = datetime.datetime.now()
    #     timestamps = [
    #         now - datetime.timedelta(minutes=(len(df) - i - 1)) for i in range(len(df))
    #     ]
    #     df["timestamp"] = timestamps
    # df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
    # df.index = df["timestamp"]
    # Generate feature vectors
    feature_vectors, feature_columns = generate_feature_vectors(df)

    # Generate additional features
    # df = generate_additional_features(df)
    # additional_features = [
    #     "hour",
    #     "day_of_week",
    #     "is_weekend",
    #     "macd",
    #     "stochastic",
    #     "bollinger_mavg",
    #     "bollinger_hband",
    #     "bollinger_lband",
    # ]
    # additional_feature_vectors = df[additional_features].values

    # # Combine original and additional features
    # feature_vectors = np.concatenate(
    #     (feature_vectors, additional_feature_vectors), axis=1
    # )
    feature_vectors = pd.DataFrame(feature_vectors)
    df_smoothed = feature_vectors.rolling(window=3).mean() 
    # Scaling
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(df_smoothed.values)
    joblib.dump(scaler, scaler_path)
    # Create sequences of feature vectors
    sequence_length = 500
    x = []
    y = []
    for i in range(len(feature_vectors_scaled) - sequence_length - 20 + 1):
        x.append(feature_vectors_scaled[i : i + sequence_length])
        y.append(
            feature_vectors_scaled[i + sequence_length : i + sequence_length + 20]
        )  # Next 5 data points

    x = np.array(x)
    y = np.array(y)

    # Data Augmentation
    x_aug, y_aug = x,y#augment_data(x, y)

    # Split data into training, validation, and testing sets
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_aug, y_aug, test_size=0.2, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    # Define transformer model parameters
    input_shape = (sequence_length, x.shape[2])
    num_heads = 8
    ff_dim = 128
    num_transformer_blocks = 6
    dropout_rate = 0.2
    embedding_dim = 128  # Dimension of the in-model embedding
    output_steps = 20  # Predict next 5 steps

    # Build the transformer model
    if COMPLETE_TRAINING and os.path.exists(checkpoint_path):
        print("Loading model from checkpoint...")
        model = load_model(checkpoint_path)
        scaler = joblib.load(scaler_path)
    else:
        print("Building a new model...")
        model = build_enhanced_transformer_model(
            input_shape=input_shape,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            dropout_rate=dropout_rate,
            embedding_dim=embedding_dim,
            output_steps=output_steps,
        )

    # Train the transformer model
    history = train_transformer_model(model, x_train, y_train, x_val, y_val)

    # Evaluate the model
    evaluate_model(model, x_test, y_test)

    # Save the transformer model and scaler
    model.save(checkpoint_path)


def build_enhanced_transformer_model(
    input_shape,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    dropout_rate,
    embedding_dim,
    output_steps=1,
):
    inputs = Input(shape=input_shape)

    # Embedding layer with increased dimensions
    embedding_layer = Dense(embedding_dim, activation="relu")(inputs)

    # Add positional encoding
    x = PositionalEncoding(input_shape[0], embedding_dim)(embedding_layer)

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate
        )(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward network
        ffn_output = Dense(
            ff_dim, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        )(x)
        ffn_output = BatchNormalization()(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    # Output layer for multi-step forecasting
    x = Flatten()(x)
    outputs = Dense(output_steps * input_shape[-1])(x)
    outputs = Reshape((output_steps, input_shape[-1]))(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=STARTING_LR), loss="mse", metrics=["mae"])
    return model


def train_transformer_model(
    model, x_train, y_train, x_val, y_val, epochs=100, batch_size=64
):
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )
    lr_scheduler = LearningRateScheduler(scheduler)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping, lr_scheduler, checkpoint],
        verbose=1,
    )
    return history


def evaluate_model(model, x_test, y_test):
    loss, mae = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")


if __name__ == "__main__":
    asyncio.run(main())
