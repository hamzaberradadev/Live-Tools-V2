import datetime
import os
import sys
import numpy as np
import asyncio
import tensorflow as tf

# ---------------------------- #
#      Hyperparameters         #
# ---------------------------- #

# Model Configuration
NUM_HEADS = 4
FF_DIM = 124  # Adjusted for practicality
NUM_TRANSFORMER_BLOCKS = 4  # Reduced for faster training
DROPOUT_RATE = 0.2
EMBEDDING_DIM = 64
SEQUENCE_LENGTH = 500
OUTPUT_STEPS = 50  # Predict next 100 steps

# Training Configuration
COMPLETE_TRAINING = True  # Set to True to train a new model
STARTING_LR = 0.001
EPOCHS = 100
BATCH_SIZE = 32

# Paths
CHECKPOINT_PATH = "transformer_model_checkpoint_decoder_only.keras"
SCALER_PATH = "scaler.save"

# ---------------------------- #
#      Environment Setup       #
# ---------------------------- #

# Optionally, enable GPU usage by commenting out the following line
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Verify TensorFlow is using CPU or GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# ---------------------------- #
#        Imports & Paths       #
# ---------------------------- #

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Layer # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import ta
import pandas as pd

# Add your project path
sys.path.append("/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2")
from utilities.bitget_perp import PerpBitget
from strategies.nvp.embadding import generate_feature_vectors

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
custom_objects={"PositionalEncoding": PositionalEncoding}

# ---------------------------- #
#      Utility Functions       #
# ---------------------------- #

def augment_data(x, y, noise_factor=0.001):
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    y_noisy = y + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=y.shape)
    return x_noisy, y_noisy

def scheduler(epoch, lr):
    if epoch > 10:
        return float(lr * 0.9)
    else:
        return float(lr)

lr_scheduler = LearningRateScheduler(scheduler)

def generate_additional_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    df["macd"] = ta.trend.macd_diff(df["close"])
    df["stochastic"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
    bollinger = ta.volatility.BollingerBands(df["close"])
    df["bollinger_mavg"] = bollinger.bollinger_mavg()
    df["bollinger_hband"] = bollinger.bollinger_hband()
    df["bollinger_lband"] = bollinger.bollinger_lband()

    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df
import tensorflow as tf

def create_causal_mask(seq_len):
    """
    Creates a lower triangular matrix to be used as a causal mask, where each position 
    only attends to the previous positions and itself.
    """
    # Create a lower triangular matrix
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask  # Shape: (seq_len, seq_len)
# ---------------------------- #
#          Model Setup          #
# ---------------------------- #

def build_transformer_decoder_only_model(
    input_shape,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    dropout_rate,
    embedding_dim,
):
    # ------------------ Input ------------------ #
    inputs = Input(shape=input_shape, name="decoder_only_inputs")  # Shape: (sequence_length, feature_dim)
    
    # Project the input to the embedding dimension
    x = Dense(embedding_dim, name="decoder_dense")(inputs)  # Shape: (sequence_length, embedding_dim)
    
    # Apply Positional Encoding
    x = PositionalEncoding(embed_dim=embedding_dim, name="positional_encoding")(x)  # Shape: (sequence_length, embedding_dim)
    x = Dropout(dropout_rate)(x)
    
    # Create the causal mask
    seq_len = input_shape[0]
    causal_mask = create_causal_mask(seq_len)

    # Implement Transformer Blocks
    for block in range(num_transformer_blocks):
        # Multi-Head Self-Attention with Causal Mask
        mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
            name=f"mha_{block}"
        )
        
        # Apply self-attention with the causal mask
        attn_output = mha(x, x, attention_mask=causal_mask)  # Shape: (sequence_length, embedding_dim)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = Add(name=f"add_{block}")([x, attn_output])
        x = LayerNormalization(epsilon=1e-6, name=f"ln_{block}")(x)
        
        # Feed-Forward Network
        ffn_output = Dense(ff_dim, activation='relu', name=f"ffn_relu_{block}")(x)
        ffn_output = Dense(embedding_dim, name=f"ffn_dense_{block}")(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = Add(name=f"ffn_add_{block}")([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6, name=f"ffn_ln_{block}")(x)
    
    # Final Output Layer
    # Flatten and map to OUTPUT_STEPS * feature_dim
    x = tf.keras.layers.Flatten()(x)  # (batch_size, sequence_length * embedding_dim)
    x = Dense(OUTPUT_STEPS * input_shape[-1], activation='linear', name="output_dense")(x)  # (batch_size, OUTPUT_STEPS * feature_dim)
    outputs = tf.keras.layers.Reshape((OUTPUT_STEPS, input_shape[-1]), name="output_reshape")(x)  # (batch_size, OUTPUT_STEPS, feature_dim)

    # ------------------ Model Compilation ------------------ #
    model = Model(inputs=inputs, outputs=outputs, name="Decoder_Only_Transformer_Model")
    model.compile(optimizer=Adam(learning_rate=STARTING_LR), loss='mse', metrics=['mae'])
    return model

# ---------------------------- #
#         Training Setup        #
# ---------------------------- #

def train_transformer_model(
    model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32
):
    # Callbacks
    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH, monitor="val_loss", save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    tensorboard = TensorBoard(log_dir="./logs")

    # Fit the model
    history = model.fit(
        x_train,  # Single input
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping, lr_scheduler, checkpoint, tensorboard],
        verbose=1,
    )
    return history

def evaluate_model(model, x_test, y_test):
    loss, mae = model.evaluate(
        x_test,
        y_test
    )
    print(f"Test Loss: {loss}, Test MAE: {mae}")

# ---------------------------- #
#          Main Function        #
# ---------------------------- #

async def main():
    try:
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

        # Feature Engineering
        feature_vectors, feature_columns = generate_feature_vectors(df)

        # Smoothing
        feature_vectors = pd.DataFrame(feature_vectors, columns=feature_columns)
        df_smoothed = feature_vectors.rolling(window=3).mean().dropna()

        # Scaling
        scaler = StandardScaler()
        feature_vectors_scaled = scaler.fit_transform(df_smoothed.values)
        joblib.dump(scaler, SCALER_PATH)

        # Create sequences of feature vectors
        x = []
        y = []
        for i in range(len(feature_vectors_scaled) - SEQUENCE_LENGTH - OUTPUT_STEPS + 1):
            x.append(feature_vectors_scaled[i : i + SEQUENCE_LENGTH])
            y.append(
                feature_vectors_scaled[
                    i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + OUTPUT_STEPS
                ]
            )

        x = np.array(x)  # Shape: (samples, SEQUENCE_LENGTH, feature_dim)
        y = np.array(y)  # Shape: (samples, OUTPUT_STEPS, feature_dim)

        # Data Augmentation (if needed)
        x_aug, y_aug = augment_data(x, y)

        # Split data into training, validation, and testing sets
        x_train, x_temp, y_train, y_temp = train_test_split(
            x_aug, y_aug, test_size=0.2, random_state=42
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=42
        )

        # Define transformer model parameters
        input_shape = (SEQUENCE_LENGTH, x.shape[2])  # (500, 6)
        output_shape = (OUTPUT_STEPS, x.shape[2])   # (100, 6)

        # Debugging: Print shapes to verify
        print("x_train shape:", x_train.shape)  # Should be (batch_size, 500, 6)
        print("y_train shape:", y_train.shape)  # Should be (batch_size, 100, 6)

        print("x_val shape:", x_val.shape)
        print("y_val shape:", y_val.shape)

        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)

        # Delete existing checkpoint if COMPLETE_TRAINING is False
        if not COMPLETE_TRAINING and os.path.exists(CHECKPOINT_PATH):
            print("Deleting existing checkpoint to train a new model...")
            os.remove(CHECKPOINT_PATH)

        # Build or Load the transformer model
        if not os.path.exists(CHECKPOINT_PATH):
            print("Building a new model...")
            model = build_transformer_decoder_only_model(
                input_shape=input_shape,
                num_heads=NUM_HEADS,
                ff_dim=FF_DIM,
                num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
                dropout_rate=DROPOUT_RATE,
                embedding_dim=EMBEDDING_DIM,
            )
            print(model.summary())
        else:
            print("Loading model from checkpoint...")
            model = tf.keras.models.load_model(CHECKPOINT_PATH, custom_objects=custom_objects)
            scaler = joblib.load(SCALER_PATH)

        # Train the transformer model
        history = train_transformer_model(
            model,
            x_train,
            y_train,
            x_val,
            y_val
        )

        # Evaluate the model
        evaluate_model(model, x_test, y_test)

        # Save the transformer model and scaler
        model.save(CHECKPOINT_PATH)

    except Exception as e:
        print(f"An error occurred: {e}")

# ---------------------------- #
#           Execution          #
# ---------------------------- #

if __name__ == "__main__":
    asyncio.run(main())
