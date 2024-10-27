import datetime
import os
import sys
import numpy as np
import asyncio
import tensorflow as tf

# Disable GPU and suppress TensorFlow messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import (
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
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    LearningRateScheduler,
    ModelCheckpoint,
)
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import ta
import pandas as pd

sys.path.append(
    "/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2"
)
from utilities.bitget_perp import PerpBitget
from strategies.nvp.embadding import generate_feature_vectors

# Constants
COMPLETE_TRAINING = False
STARTING_LR = 0.001
checkpoint_path = "transformer_model_checkpoint.keras"
scaler_path = "scaler.save"


class LookAheadMaskLayer(Layer):
    def call(self, inputs):
        size = tf.shape(inputs)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask


# Custom Positional Encoding Layer
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
        config = super(PositionalEncoding, self).get_config()
        config.update({"maxlen": self.maxlen, "embed_dim": self.embed_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Look-Ahead Mask for Decoder
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


# Data Augmentation Function (if needed)
def augment_data(x, y, noise_factor=0.01):
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    y_noisy = y + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=y.shape)
    return x_noisy, y_noisy


# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch > 10:
        return float(lr * 0.9)
    else:
        return float(lr)


lr_scheduler = LearningRateScheduler(scheduler)


# Generate Additional Features (if needed)
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


# Main Function
async def main():
    # Initialize the exchange (replace with your data source)
    # For demonstration, let's create a synthetic dataset
    # Replace this part with your data fetching and preparation
    # Here, we will simulate some data
    # Initialize the exchange
    exchange = PerpBitget()
    await exchange.load_markets()

    # Select a single pair
    pair = "BTC/USDT"

    # Fetch OHLCV data
    timeframe = "1m"
    limit = 5000
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)

    # Close the exchange session
    await exchange.close()

    feature_vectors, feature_columns = generate_feature_vectors(df)

    # Smoothing
    feature_vectors = pd.DataFrame(feature_vectors, columns=feature_columns)
    df_smoothed = feature_vectors.rolling(window=3).mean().dropna()

    # Scaling
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(df_smoothed.values)
    joblib.dump(scaler, scaler_path)

    # Create sequences of feature vectors
    sequence_length = 300
    output_steps = 20  # Predict next 20 steps
    x = []
    y = []
    for i in range(len(feature_vectors_scaled) - sequence_length - output_steps + 1):
        x.append(feature_vectors_scaled[i : i + sequence_length])
        y.append(
            feature_vectors_scaled[
                i + sequence_length : i + sequence_length + output_steps
            ]
        )

    x = np.array(x)
    y = np.array(y)

    # Data Augmentation (if needed)
    x_aug, y_aug = x, y  # augment_data(x, y)

    # Split data into training, validation, and testing sets
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_aug, y_aug, test_size=0.2, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    # Prepare decoder inputs by shifting y_train and y_val
    decoder_input_train = np.zeros_like(y_train)
    decoder_input_train[:, 1:, :] = y_train[:, :-1, :]
    decoder_input_val = np.zeros_like(y_val)
    decoder_input_val[:, 1:, :] = y_val[:, :-1, :]
    decoder_input_test = np.zeros_like(y_test)
    decoder_input_test[:, 1:, :] = y_test[:, :-1, :]

    # Define transformer model parameters
    input_shape = (sequence_length, x.shape[2])
    output_shape = (output_steps, x.shape[2])
    num_heads = 8
    ff_dim = 256  # Increased feed-forward network dimension
    num_transformer_blocks = 6
    dropout_rate = 0.2
    embedding_dim = 128  # Dimension of the embedding

    # Build the transformer model
    if COMPLETE_TRAINING and os.path.exists(checkpoint_path):
        print("Loading model from checkpoint...")
        custom_objects = {"PositionalEncoding": PositionalEncoding, "tf": tf}
        model = load_model(checkpoint_path, custom_objects=custom_objects)
        scaler = joblib.load(scaler_path)
    else:
        print("Building a new model...")
        model = build_transformer_encoder_decoder_model(
            input_shape=input_shape,
            output_shape=output_shape,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            dropout_rate=dropout_rate,
            embedding_dim=embedding_dim,
        )

    # Train the transformer model
    history = train_transformer_model(
        model,
        [
            x_train,
            decoder_input_train,
        ],  # Pass both encoder and decoder inputs for training
        y_train,
        [
            x_val,
            decoder_input_val,
        ],  # Pass both encoder and decoder inputs for validation
        y_val,
    )

    # Evaluate the model
    evaluate_model(model, [x_test, decoder_input_test], y_test)

    # Save the transformer model and scaler
    model.save(checkpoint_path)


# Build Encoder-Decoder Transformer Model
def build_transformer_encoder_decoder_model(
    input_shape,
    output_shape,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    dropout_rate,
    embedding_dim,
):
    # Define the encoder inputs
    encoder_inputs = Input(shape=input_shape, name="encoder_inputs")
    x = Dense(embedding_dim)(encoder_inputs)
    x = PositionalEncoding(input_shape[0], embedding_dim)(x)

    # Create the encoder with multiple transformer blocks
    for _ in range(num_transformer_blocks):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        ffn_output = Dense(ff_dim, activation='relu')(x)
        ffn_output = Dense(embedding_dim)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
    encoder_outputs = x

    # Define the decoder inputs
    decoder_inputs = Input(shape=output_shape, name="decoder_inputs")
    y = Dense(embedding_dim)(decoder_inputs)
    y = PositionalEncoding(output_shape[0], embedding_dim)(y)

    # Create the decoder with multiple transformer blocks
    for _ in range(num_transformer_blocks):
        # Masked Multi-Head Attention
        look_ahead_mask = LookAheadMaskLayer()(y)
        attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(
            y, y, attention_mask=look_ahead_mask
        )
        y = Add()([y, attn1])
        y = LayerNormalization(epsilon=1e-6)(y)
        # Encoder-Decoder Attention
        attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(y, encoder_outputs)
        y = Add()([y, attn2])
        y = LayerNormalization(epsilon=1e-6)(y)
        # Feed Forward Network
        ffn_output = Dense(ff_dim, activation='relu')(y)
        ffn_output = Dense(embedding_dim)(ffn_output)
        y = Add()([y, ffn_output])
        y = LayerNormalization(epsilon=1e-6)(y)
    decoder_outputs = Dense(output_shape[-1])(y)

    # Create the model with both encoder and decoder inputs
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=STARTING_LR), loss='mse', metrics=['mae'])
    return model

# Training Function
def train_transformer_model(
    model, x_train_inputs, y_train, x_val_inputs, y_val, epochs=100, batch_size=32
):
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    lr_scheduler = LearningRateScheduler(scheduler)

    # Pass the inputs as a list of [encoder_inputs, decoder_inputs]
    history = model.fit(
        [x_train_inputs[0], x_train_inputs[1]],  # [encoder_inputs, decoder_inputs]
        y_train,
        validation_data=(
            [x_val_inputs[0], x_val_inputs[1]],
            y_val,
        ),  # [encoder_inputs, decoder_inputs]
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping, lr_scheduler, checkpoint],
        verbose=1,
    )
    return history


# Evaluation Function
def evaluate_model(model, x_test_inputs, y_test):
    loss, mae = model.evaluate(x_test_inputs, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")


if __name__ == "__main__":
    asyncio.run(main())
