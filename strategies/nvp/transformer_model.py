import os
import sys
import numpy as np
import asyncio
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Embedding,
    Lambda,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import joblib
# Disable GPU and suppress TensorFlow messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Add your project path
sys.path.append(
    "/home/hamza-berrada/Desktop/cooding/airflow/airflow/pluggings/Live-Tools-V2"
)
from utilities.bitget_perp import PerpBitget
from strategies.nvp.embadding import generate_feature_vectors


async def main():
    # Initialize the exchange
    exchange = PerpBitget()
    await exchange.load_markets()

    # Select a single pair
    pair = "BTC/USDT"

    # Fetch OHLCV data
    timeframe = "1h"
    limit = 10000
    df = await exchange.get_last_ohlcv(pair, timeframe, limit)

    # Close the exchange session
    await exchange.close()

    # Generate feature vectors
    feature_vectors, _ = generate_feature_vectors(df)

    # Scaling
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(feature_vectors)

    # Create sequences of feature vectors
    sequence_length = 150
    x = []
    y = []
    for i in range(len(feature_vectors_scaled) - sequence_length):
        x.append(feature_vectors_scaled[i : i + sequence_length])
        y.append(feature_vectors_scaled[i + sequence_length])  # Next data point

    x = np.array(x)
    y = np.array(y)

    # Split data into training, validation, and testing sets
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    # Define transformer model parameters
    input_shape = (sequence_length, x.shape[2])
    num_heads = 36
    ff_dim = 128
    num_transformer_blocks = 20
    dropout_rate = 0.2
    embedding_dim = 64  # Dimension of the in-model embedding

    # Build the transformer model
    model = build_transformer_model(
        input_shape=input_shape,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        dropout_rate=dropout_rate,
        embedding_dim=embedding_dim,
    )

    # Train the transformer model
    history = train_transformer_model(model, x_train, y_train, x_val, y_val)

    # Evaluate the model
    evaluate_model(model, x_test, y_test)

    # Save the transformer model
    model.save("transformer_model.keras")
    joblib.dump(scaler, 'scaler.save')


def build_transformer_model(
    input_shape, num_heads, ff_dim, num_transformer_blocks, dropout_rate, embedding_dim
):
    inputs = Input(shape=(input_shape[0], input_shape[1]))  # Input shape: (sequence_length, num_features)
    
    # Embedding layer
    embedding_layer = Dense(embedding_dim, activation="relu")(inputs)  # Transform input to embedding_dim
    
    x = embedding_layer

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        ffn_output = Dense(ff_dim, activation="relu", kernel_regularizer=l2(0.01))(x)
        ffn_output = Dense(embedding_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    # Final output layer
    x = Lambda(lambda x: x[:, -1, :])(x)
    outputs = Dense(input_shape[-1])(x)  # Predict next feature vector

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    return model


def train_transformer_model(
    model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32
):
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping],
        verbose=1,
    )
    return history


def evaluate_model(model, x_test, y_test):
    loss, mae = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")


if __name__ == "__main__":
    asyncio.run(main())
