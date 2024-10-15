import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf



# Function to create sequences with multistep output for 30 days forecast
def create_sequences(data, target_index, sequence_length, forecast_steps=30):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - forecast_steps):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length:i + sequence_length + forecast_steps, target_index]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Build an LSTM model with dropout and batch normalization
def build_multistep_model(input_shape, output_steps, learning_rate=0.0001):
    model = Sequential()

    # Nur ein LSTM-Layer mit BatchNormalization und Dropout
    model.add(LSTM(256, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(output_steps))  # Multistep output

    # Optimizer
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model

# Model training and saving
def train_and_save_model(X_train, y_train, X_val, y_val, model_name, output_steps, epochs=200, batch_size=256,
                         learning_rate=0.0001):
    model_directory = "../../models/small_model"
    model_path = os.path.join(model_directory, f"{model_name}.h5")

    # Build the model
    model = build_multistep_model((X_train.shape[1], X_train.shape[2]), output_steps, learning_rate)

    # Training
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    os.makedirs(model_directory, exist_ok=True)
    model.save(model_path)
    print(f"Model {model_name} successfully saved.")

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training Loss vs Validation Loss - {model_name}')
    plt.legend()
    plt.savefig(os.path.join(model_directory, f"{model_name}_training_loss.png"))
    plt.close()

if __name__ == "__main__":
    model_directory = "../../models/small_model"
    model_name = "eth_small_model.h5"
    model_path = os.path.join(model_directory, model_name)

    # Prüfen, ob das Modell bereits existiert
    if os.path.exists(model_path):
        user_input = input(f"Model {model_name} already exists. Do you want to retrain it? (y/n): ").strip().lower()
        if user_input == 'n':
            print(f"Loading existing model {model_name}...")
            model = load_model(model_path)
        else:
            # Modell neu trainieren
            data_directories = [
                "../../data/eth_with_indicators"
            ]
            data_files = []
            for directory in data_directories:
                if os.path.exists(directory):
                    data_files.extend([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')])

            if not data_files:
                raise ValueError("No data found for training.")

            combined_df = pd.concat([pd.read_csv(file, index_col=0) for file in data_files], ignore_index=True)
            combined_df = combined_df.dropna()

            scaler = MinMaxScaler()
            feature_columns_with_indicators = ['Open', 'High', 'Low', 'Close', 'Volume', 'Fed_Rate', 'Inflation', 'rsi', 'macd',
                                               'macd_signal', 'bb_mavg', 'bb_high', 'bb_low', 'stoch', 'stoch_signal']
            close_index = feature_columns_with_indicators.index('Close')

            scaled_data = scaler.fit_transform(combined_df[feature_columns_with_indicators])

            # Create sequences for multistep forecasting (30 days ahead)
            X, y = create_sequences(scaled_data, close_index, sequence_length=50, forecast_steps=30)

            split_ratio = 0.8
            split_index = int(len(X) * split_ratio)
            X_train, X_val = X[:split_index], X[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]

            # Train and save the model for 30 days prediction
            train_and_save_model(X_train, y_train, X_val, y_val, model_name="eth_small_model", output_steps=30)

    else:
        # Modell existiert nicht, neues Training wird durchgeführt
        print(f"No existing model found. Training a new model {model_name}...")
        data_directories = [
            "../../data/eth_with_indicators"
        ]
        data_files = []
        for directory in data_directories:
            if os.path.exists(directory):
                data_files.extend([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')])

        if not data_files:
            raise ValueError("No data found for training.")

        combined_df = pd.concat([pd.read_csv(file, index_col=0) for file in data_files], ignore_index=True)
        combined_df = combined_df.dropna()

        scaler = MinMaxScaler()
        feature_columns_with_indicators = ['Open', 'High', 'Low', 'Close', 'Volume', 'Fed_Rate', 'Inflation', 'rsi', 'macd',
                                           'macd_signal', 'bb_mavg', 'bb_high', 'bb_low', 'stoch', 'stoch_signal']
        close_index = feature_columns_with_indicators.index('Close')

        scaled_data = scaler.fit_transform(combined_df[feature_columns_with_indicators])

        # Create sequences for multistep forecasting (30 days ahead)
        X, y = create_sequences(scaled_data, close_index, sequence_length=50, forecast_steps=30)

        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        # Train and save the model for 30 days prediction
        train_and_save_model(X_train, y_train, X_val, y_val, model_name="eth_small_model", output_steps=30)
