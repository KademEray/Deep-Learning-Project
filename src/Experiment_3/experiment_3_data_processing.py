import os
import talib as ta
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands

# Prüfen, ob CUDA verfügbar ist, und das Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Funktionen zur Berechnung technischer Indikatoren
def calculate_indicators(df):
    # Technische Indikatoren berechnen
    df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()
    df['macd'] = MACD(df['Close']).macd()
    df['macd_signal'] = MACD(df['Close']).macd_signal()
    df['cci'] = CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    df['roc'] = ROCIndicator(df['Close'], window=12).roc()
    df['momentum'] = df['Close'].diff(1)
    df['stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    bb = BollingerBands(df['Close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    # Zusätzliche Indikatoren mit TA-Lib
    df['adx'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['willr'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['cci20'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
    df['mfi'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    df['sma20'] = ta.SMA(df['Close'], timeperiod=20)
    df['ema50'] = ta.EMA(df['Close'], timeperiod=50)
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['Close'], timeperiod=20)
    df['obv'] = ta.OBV(df['Close'], df['Volume'])

    return df.dropna()


def perform_correlation_reduction(df, threshold=0.9):
    # Berechne Korrelationsmatrix und entferne stark korrelierte Features
    cor_matrix = df.corr().abs()
    upper_triangle = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    reduced_df = df.drop(columns=to_drop)
    print(f"Feature-Menge nach Korrelationsanalyse reduziert auf {reduced_df.shape[1]} Features")
    return reduced_df


# Autoencoder-Klasse für nicht-lineare Reduktion
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(data_tensor, encoding_dim=10, epochs=50, lr=0.001):
    input_dim = data_tensor.shape[1]
    autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded, decoded = autoencoder(data_tensor)
        loss = criterion(decoded, data_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoche {epoch + 1}/{epochs}, Verlust: {loss.item():.6f}")

    with torch.no_grad():
        encoded_features, _ = autoencoder(data_tensor)
    return encoded_features


def generate_sequences(data, sequence_length=50, forecast_steps=30):
    X, Y = [], []
    for i in range(len(data) - sequence_length - forecast_steps):
        x_seq = data[i:i + sequence_length].values
        y_seq = data.iloc[i + sequence_length:i + sequence_length + forecast_steps].values
        X.append(x_seq)
        Y.append(y_seq)

    X, Y = np.array(X), np.array(Y)
    print(f"Debug: Final X shape {X.shape}, Y shape {Y.shape}")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


# Hauptskript zur Verarbeitung und Gruppierung der Daten
if __name__ == "__main__":
    output_dir = "./Data/"
    samples_dir = "./Data/Samples/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    crypto_symbol = "BTC-USD"
    start_date, end_date = "2010-01-01", "2023-01-31"

    # Daten laden und vorbereiten
    df = yf.download(crypto_symbol, start=start_date, end=end_date)
    df = calculate_indicators(df)
    print(f"Datensatzgröße nach Hinzufügen der Indikatoren: {df.shape}")

    # Skalierung der Daten
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    # Feature-Gruppen definieren
    standard_features = scaled_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    indicator_group_1 = scaled_df[['rsi', 'macd', 'macd_signal', 'momentum', 'stochastic', 'adx', 'willr', 'cci20', 'mfi']]
    indicator_group_2 = scaled_df[['cci', 'roc', 'bb_upper', 'bb_lower', 'sma20', 'ema50', 'upperband', 'lowerband', 'obv']]

    # Jede Gruppe separat durch den Reduktionsprozess schicken
    groups = {
        'Standard': standard_features,
        'Indicators_Group_1': indicator_group_1,
        'Indicators_Group_2': indicator_group_2,
    }

    for group_name, group_data in groups.items():
        print(f"Verarbeite Gruppe: {group_name}")

        # Korrelationsanalyse durchführen
        reduced_df = perform_correlation_reduction(group_data)

        # Tensor der reduzierten Daten vorbereiten
        data_tensor = torch.tensor(reduced_df.values, dtype=torch.float32).to(device)

        # Autoencoder zur weiteren Reduktion der Feature-Menge trainieren
        encoding_dim = 10  # Ziel-Dimension
        encoded_features = train_autoencoder(data_tensor, encoding_dim=encoding_dim)

        # Endgültige reduzierte Daten als DataFrame speichern
        reduced_features_df = pd.DataFrame(encoded_features.cpu().numpy(),
                                           columns=[f"feature_{i}" for i in range(encoding_dim)])
        print(f"Reduzierte Datenform nach Autoencoder für {group_name}: {reduced_features_df.shape}")

        # Sequenzen generieren
        sequence_length = 50  # Länge der Eingabesequenz in Tagen
        forecast_steps = 30  # Vorhersageperiode
        X, Y = generate_sequences(reduced_features_df, sequence_length=sequence_length, forecast_steps=forecast_steps)

        # Sequenzen speichern
        with open(os.path.join(samples_dir, f"{group_name}_X.pkl"), "wb") as f:
            pickle.dump(X, f)
        with open(os.path.join(samples_dir, f"{group_name}_Y.pkl"), "wb") as f:
            pickle.dump(Y, f)

    # Speichern des Scalers
    scaler_path = "./Data/Models/scaler.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler und Sequenzen wurden gespeichert.")
