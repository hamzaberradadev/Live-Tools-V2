import datetime
import os
import sys
import numpy as np
import asyncio
import tensorflow as tf
import logging

# Set up logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------- #
#      Hyperparameters         #
# ---------------------------- #

# Model Configuration
NUM_HEADS = 18
FF_DIM = 1256  # Feed-forward network dimension
NUM_TRANSFORMER_BLOCKS = 16
DROPOUT_RATE = 0.2
EMBEDDING_DIM = 256
SEQUENCE_LENGTH = 1000
OUTPUT_STEPS = 50  # Predict next 20 steps
LOAD_NEW_DATA = True
pair = "BTC/USDT"
coin_to_dl = ["BTC/USDT:USDT"]
interval = "1m"
start_date = "2024-10-01 00:00:00"
# Training Configuration
COMPLETE_TRAINING = False  # Set to True to train a new model
STARTING_LR = 0.001
EPOCHS = 20
BATCH_SIZE = 32

# Paths
CHECKPOINT_PATH = "transformer_model_checkpoint.keras"
SCALER_PATH = "scaler.save"

# ---------------------------- #
#      Environment Setup       #
# ---------------------------- #

# Disable GPU and suppress TensorFlow messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Verify TensorFlow is using CPU
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

#from utilities.bitget_perp import PerpBitget
from utilities.data_manager import ExchangeDataManager
from strategies.nvp.embadding import generate_feature_vectors

# ---------------------------- #
#      Custom Layers           #
# ---------------------------- #

# Corrected Positional Encoding Layer using TensorFlow operations
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

# ---------------------------- #
#      Utility Functions       #
# ---------------------------- #

# Data Augmentation Function (if needed)
def augment_data(x, y, noise_factor=0.001):
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    y_noisy = y + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=y.shape)
    return x_noisy, y_noisy

# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr / 0.9)
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

# Custom Causal Mask Function using Lambda Layers
def create_causal_mask_layer(x):
    # x: KerasTensor with shape (batch_size, target_seq_len, embed_dim)
    target_seq_len = tf.shape(x)[1]
    # Create a lower triangular matrix
    mask = tf.linalg.band_part(tf.ones((target_seq_len, target_seq_len)), -1, 0)
    # Reshape to (1, 1, target_seq_len, target_seq_len)
    mask = tf.reshape(mask, (1, 1, target_seq_len, target_seq_len))
    # Tile the mask for the batch size
    mask = tf.tile(mask, [tf.shape(x)[0], 1, 1, 1])
    return mask  # Shape: (batch_size, 1, target_seq_len, target_seq_len)

# ---------------------------- #
#          Model Setup          #
# ---------------------------- #

# Implementing the Decoder Layer (Inspired by the Tutorial)
class DecoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.multihead_attention1 = MultiHeadAttention(num_heads=h, key_dim=d_k, dropout=rate)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = LayerNormalization(epsilon=1e-6)

        self.multihead_attention2 = MultiHeadAttention(num_heads=h, key_dim=d_k, dropout=rate)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = LayerNormalization(epsilon=1e-6)

        self.feed_forward = Dense(d_ff, activation='relu')
        self.feed_forward_dense = Dense(d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = LayerNormalization(epsilon=1e-6)

    def call(self, x, encoder_output=None, lookahead_mask=None, padding_mask=None, training=False):
        # Self-Attention block
        attn1 = self.multihead_attention1(query=x, value=x, key=x, attention_mask=lookahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.add_norm1(x + attn1)

        # Encoder-Decoder Attention block
        attn2 = self.multihead_attention2(query=out1, value=encoder_output, key=encoder_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.add_norm2(out1 + attn2)

        # Feed Forward Network
        ffn_output = self.feed_forward(out2)
        ffn_output = self.feed_forward_dense(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.add_norm3(out2 + ffn_output)

        return out3

# Implementing the Decoder (Inspired by the Tutorial)
class Decoder(Layer):
    def __init__(self, output_steps, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_steps = output_steps
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(embed_dim=d_model, name="decoder_pos_encoding")
        self.dropout = Dropout(rate)
        self.decoder_layers = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate, name=f"decoder_layer_{i}") for i in range(n)]

    def call(self, decoder_inputs, encoder_output, lookahead_mask=None, padding_mask=None, training=False):
        # Apply positional encoding
        x = self.pos_encoding(decoder_inputs)
        x = self.dropout(x, training=training)

        # Pass through each decoder layer
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                x, 
                encoder_output=encoder_output, 
                lookahead_mask=lookahead_mask, 
                padding_mask=padding_mask, 
                training=training
            )

        return x
custom_objects = {
    "PositionalEncoding": PositionalEncoding, 
    "Decoder": Decoder, 
    "DecoderLayer": DecoderLayer,
    "tf": tf
}
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
    # ------------------ Encoder ------------------ #
    encoder_inputs = Input(shape=input_shape, name="encoder_inputs")  # Shape: (300, 6)
    # Project the input to the embedding dimension
    x = Dense(embedding_dim, name="encoder_dense")(encoder_inputs)   # Shape: (300, 128)
    encoder_pos_encoding = PositionalEncoding(
        embed_dim=embedding_dim,
        name="encoder_pos_encoding"
    )(x)  # Shape: (300, 128)

    for block in range(num_transformer_blocks):
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
            name=f"encoder_mha_{block}"
        )(encoder_pos_encoding, encoder_pos_encoding, attention_mask=None)  # Shape: (300, 128)
        x = Add(name=f"encoder_add_{block}")([encoder_pos_encoding, attn_output])  # Residual connection
        x = LayerNormalization(epsilon=1e-6, name=f"encoder_ln_{block}")(x)  # Shape: (300, 128)

        ffn_output = Dense(ff_dim, activation='relu', name=f"encoder_ffn_relu_{block}")(x)  # Shape: (300, ff_dim)
        ffn_output = Dense(embedding_dim, name=f"encoder_ffn_dense_{block}")(ffn_output)    # Shape: (300, 128)
        x = Add(name=f"encoder_ffn_add_{block}")([x, ffn_output])  # Residual connection
        x = LayerNormalization(epsilon=1e-6, name=f"encoder_ffn_ln_{block}")(x)  # Shape: (300, 128)
    encoder_outputs = x  # Final encoder output: (300, 128)

    # ------------------ Decoder ------------------ #
    decoder_inputs = Input(shape=output_shape, name="decoder_inputs")  # Shape: (20, 6)
    # Project the decoder inputs to the embedding dimension
    y = Dense(embedding_dim, name="decoder_dense")(decoder_inputs)  # Shape: (20, 128)
    decoder_pos_encoding = PositionalEncoding(
        embed_dim=embedding_dim,
        name="decoder_pos_encoding"
    )(y)  # Shape: (20, 128)

    # Pass through the decoder layers using keyword arguments
    decoder = Decoder(
        output_steps=output_shape[0],  # 20
        h=num_heads,
        d_k=embedding_dim // num_heads,
        d_v=embedding_dim // num_heads,
        d_model=embedding_dim,
        d_ff=ff_dim,
        n=num_transformer_blocks,
        rate=dropout_rate,
        name="decoder"
    )(decoder_inputs=decoder_pos_encoding, 
      encoder_output=encoder_outputs, 
      lookahead_mask=None, 
      padding_mask=None, 
      training=True)  # Shape: (20, 128)

    # Final Output Layer
    decoder_outputs = Dense(output_shape[-1], name="decoder_output_dense")(decoder)  # Shape: (20, 6)

    # ------------------ Model Compilation ------------------ #
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs, name="Transformer_Model")
    model.compile(optimizer=Adam(learning_rate=STARTING_LR), loss='mse', metrics=['mae'])
    return model

# ---------------------------- #
#         Training Setup        #
# ---------------------------- #

# Training Function
def train_transformer_model(
    model, x_train_inputs, y_train, x_val_inputs, y_val, epochs=100, batch_size=32
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
        [x_train_inputs[0], x_train_inputs[1]],  # Pass as list
        y_train,
        validation_data=(
            [x_val_inputs[0], x_val_inputs[1]],  # Pass as list
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping, lr_scheduler, checkpoint, tensorboard],
        verbose=1,
    )
    return history

# Evaluation Function
def evaluate_model(model, x_test_inputs, y_test):
    loss, mae = model.evaluate(
        [x_test_inputs[0], x_test_inputs[1]],
        y_test
    )
    print(f"Test Loss: {loss}, Test MAE: {mae}")

# ---------------------------- #
#          Main Function        #
# ---------------------------- #

async def main():
    pair = "BTC/USDT"
    df_smoothed = pd.DataFrame()
    
    if LOAD_NEW_DATA:
        # Initialisation de l'ExchangeDataManager
        exchange = ExchangeDataManager(
            exchange_name="bitget", path_download="./database/exchanges"
        )
        
        # Définir les paramètres pour le téléchargement des données

        intervals = [interval]
        
        # Télécharger les données
        await exchange.download_data(
            coins=coin_to_dl,
            intervals=intervals,
            start_date=start_date,
        )
        df = exchange.load_data(pair, interval, start_date)
        logger.info("Starting generate feature vectors.")
        df_donnes = pd.DataFrame()
        for i in range(0, len(df), 5000):
            subset = df.iloc[i:i + 5000].copy()

            # Feature Engineering
            feature_vectors, _ = generate_feature_vectors(subset)
            df_donnes = pd.concat([df_donnes,feature_vectors])
        logger.info("generate feature vectors done.")
        df_smoothed = df_donnes.rolling(window=2).mean().dropna()
        df_smoothed.to_csv(f'./data.csv')
    else:
        df_smoothed = pd.read_csv(f'./data.csv')
    # Smoothing

    # Scaling
    logger.info("Scaling feature vectors.")
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(df_smoothed.drop(columns=['date']).values)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaling feature vectors done!.")

    # Create sequences of feature vectors
    logger.info("Creating vector traning")
    x = []
    y = []
    for i in range(len(feature_vectors_scaled) - SEQUENCE_LENGTH - OUTPUT_STEPS + 1):
        x.append(feature_vectors_scaled[i : i + SEQUENCE_LENGTH])
        y.append(
            feature_vectors_scaled[
                i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + OUTPUT_STEPS
            ]
        )

    x = np.array(x)
    y = np.array(y)

    # Data Augmentation (if needed)
    # x_aug, y_aug = x, y  # Uncomment if augmentation is desired
    x_aug, y_aug = augment_data(x, y)

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
    # For the first timestep, typically a start token is used, but here zero is used
    decoder_input_val = np.zeros_like(y_val)
    decoder_input_val[:, 1:, :] = y_val[:, :-1, :]
    decoder_input_test = np.zeros_like(y_test)
    decoder_input_test[:, 1:, :] = y_test[:, :-1, :]

    # Define transformer model parameters
    input_shape = (SEQUENCE_LENGTH, x.shape[2])  # (300,6)
    output_shape = (OUTPUT_STEPS, x.shape[2])   # (20,6)
    logger.info("Creating vector traning done!")
    # Debugging: Print shapes to verify
    print("x_train shape:", x_train.shape)  # Should be (batch_size, 300, 6)
    print("decoder_input_train shape:", decoder_input_train.shape)  # Should be (batch_size, 20, 6)
    print("y_train shape:", y_train.shape)  # Should be (batch_size, 20, 6)

    print("x_val shape:", x_val.shape)  # Should be (468, 300, 6)
    print("decoder_input_val shape:", decoder_input_val.shape)  # Should be (468, 20, 6)
    print("y_val shape:", y_val.shape)  # Should be (468, 20, 6)

    print("x_test shape:", x_test.shape)  # Should be (468, 300, 6)
    print("decoder_input_test shape:", decoder_input_test.shape)  # Should be (468, 20, 6)
    print("y_test shape:", y_test.shape)  # Should be (468, 20, 6)

    
    if not COMPLETE_TRAINING and os.path.exists(CHECKPOINT_PATH):
        logger.info("Deleting existing checkpoint to train a new model...")
        os.remove(CHECKPOINT_PATH)

    # Build or Load the transformer model
    if not os.path.exists(CHECKPOINT_PATH):
        logger.info("Building a new model...")
        model = build_transformer_encoder_decoder_model(
            input_shape=input_shape,
            output_shape=output_shape,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
            dropout_rate=DROPOUT_RATE,
            embedding_dim=EMBEDDING_DIM,
        )
        print(model.summary())
    else:
        logger.info("Loading model from checkpoint...")

        model = tf.keras.models.load_model(CHECKPOINT_PATH, custom_objects=custom_objects)
        scaler = joblib.load(SCALER_PATH)
    

    # Train the transformer model
    history = train_transformer_model(
        model,
        [x_train, decoder_input_train],  # Pass as list
        y_train,
        [x_val, decoder_input_val],      # Pass as list
        y_val
    )

    # Evaluate the model
    evaluate_model(model, [x_test, decoder_input_test], y_test)

    # Save the transformer model and scaler
    model.save(CHECKPOINT_PATH)

# ---------------------------- #
#           Execution           #
# ---------------------------- #

if __name__ == "__main__":
    asyncio.run(main())
