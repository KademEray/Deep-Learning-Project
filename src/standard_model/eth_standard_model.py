import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")


# Funktion zum Erstellen von Sequenzen
def create_sequences(data, target_index, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length, target_index]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# Modelltraining und -speicherung mit Überprüfung auf vorhandenes Modell
def train_and_save_model(X_train, y_train, X_val, y_val, model_name, optimizer='adam', epochs=100, batch_size=32,
                         learning_rate=0.0001):
    model_directory = "../../models/standard_model"
    model_path = os.path.join(model_directory, f"{model_name}.h5")

    # Überprüfung, ob das Modell bereits existiert
    if os.path.exists(model_path):
        user_input = input(
            f"Das Modell {model_name} existiert bereits. Möchtest du es löschen und neu trainieren? (y/n): ")
        if user_input.lower() == 'y':
            os.remove(model_path)
            print(f"Altes Modell {model_name} wurde gelöscht.")
        else:
            print(f"Training abgebrochen. Modell {model_name} wurde nicht überschrieben.")
            return

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stopping])

    os.makedirs(model_directory, exist_ok=True)
    model.save(model_path)
    print(f"Modell {model_name} wurde erfolgreich gespeichert.")

    # Plotten des Verlustes
    plt.plot(history.history['loss'], label='Trainingsverlust')
    plt.plot(history.history['val_loss'], label='Validierungsverlust')
    plt.title(f'Verlust während des Trainings - {model_name}')
    plt.legend()
    plt.savefig(os.path.join(model_directory, f"{model_name}_training_loss.png"))
    plt.close()


if __name__ == "__main__":

    data_directories = [
        "../../data/eth_with_indicators"
    ]
    data_files = []
    for directory in data_directories:
        if os.path.exists(directory):
            data_files.extend([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')])

    if not data_files:
        raise ValueError("Keine Daten zum Trainieren gefunden.")

    combined_df = pd.concat([pd.read_csv(file, index_col=0) for file in data_files], ignore_index=True)
    combined_df = combined_df.dropna()

    scaler = MinMaxScaler()
    feature_columns_with_indicators = ['Open', 'High', 'Low', 'Close', 'Volume', 'Fed_Rate', 'Inflation', 'rsi', 'macd',
                                       'macd_signal', 'bb_mavg', 'bb_high', 'bb_low', 'stoch', 'stoch_signal']
    close_index = feature_columns_with_indicators.index('Close')

    scaled_data = scaler.fit_transform(combined_df[feature_columns_with_indicators])
    X, y = create_sequences(scaled_data, close_index, sequence_length=50)

    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    train_and_save_model(X_train, y_train, X_val, y_val, model_name="eth_standard_model")
