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
import random

# Setzen der Zufallszahlenseeds für Python, NumPy und PyTorch
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# Falls CUDA verwendet wird, den Seed auch für CUDA setzen
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)  # Für Multi-GPU, falls verwendet

# Zusätzliche Einstellungen, um deterministisches Verhalten sicherzustellen
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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


def perform_correlation_with_close(df, target='Close', threshold=0.2):
    """
    Reduziere Features basierend auf ihrer Korrelation mit dem Zielwert (Close).

    Parameters:
    df (pd.DataFrame): Der DataFrame mit Features.
    target (str): Das Ziel-Feature, zu dem die Korrelation berechnet wird.
    threshold (float): Minimale Korrelation, um ein Feature beizubehalten.

    Returns:
    pd.DataFrame: Reduzierter DataFrame mit nur den relevanten Features.
    """
    cor_with_target = df.corr()[target].abs()  # Korrelation mit 'Close'
    print(f"Korrelation mit {target}:\n{cor_with_target}\n")

    # Behalte nur Features mit hoher Korrelation oder das Ziel-Feature selbst
    relevant_features = cor_with_target[cor_with_target > threshold].index
    reduced_df = df[relevant_features]

    print(f"Feature-Menge nach Korrelation mit {target} reduziert auf {len(relevant_features)} Features")
    return reduced_df


# Autoencoder-Klasse für nicht-lineare Reduktion
class Autoencoder(nn.Module):
    """
    A simple Autoencoder neural network for dimensionality reduction and feature learning.
    Args:
        input_dim (int): The number of input features.
        encoding_dim (int, optional): The dimension of the encoded representation. Default is 10.
    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder, which compresses the input.
        decoder (nn.Sequential): The decoder part of the autoencoder, which reconstructs the input from the encoded representation.
    Methods:
        forward(x):
            Performs a forward pass through the autoencoder.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                encoded (torch.Tensor): The encoded representation of the input.
                decoded (torch.Tensor): The reconstructed input from the encoded representation.
    """
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
    """
    Trains an autoencoder on the provided data tensor.
    Args:
        data_tensor (torch.Tensor): The input data tensor to be encoded and decoded.
        encoding_dim (int, optional): The dimension of the encoding layer. Default is 10.
        epochs (int, optional): The number of training epochs. Default is 50.
        lr (float, optional): The learning rate for the optimizer. Default is 0.001.
    Returns:
        torch.Tensor: The encoded features of the input data tensor after training.
    """
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
    """
    Generates sequences of data for time series forecasting.
    Args:
        data (pd.DataFrame or pd.Series): The input time series data.
        sequence_length (int, optional): The length of each input sequence. Defaults to 50.
        forecast_steps (int, optional): The number of steps to forecast. Defaults to 30.
    Returns:
        tuple: A tuple containing two torch.Tensors:
            - X (torch.Tensor): The input sequences of shape (num_sequences, sequence_length, num_features).
            - Y (torch.Tensor): The corresponding forecast sequences of shape (num_sequences, forecast_steps, num_features).
    """
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
    start_date, end_date = "2010-01-01", "2023-02-01"

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
    indicator_group_1 = scaled_df[
        ['rsi', 'macd', 'macd_signal', 'momentum', 'stochastic', 'adx', 'willr', 'cci20', 'mfi']]
    indicator_group_2 = scaled_df[
        ['cci', 'roc', 'bb_upper', 'bb_lower', 'sma20', 'ema50', 'upperband', 'lowerband', 'obv']]

    # Jede Gruppe separat durch den Reduktionsprozess schicken
    groups = {
        'Standard': standard_features,
        'Indicators_Group_1': indicator_group_1,
        'Indicators_Group_2': indicator_group_2,
    }

    for group_name, group_data in groups.items():
        print(f"\nVerarbeite Gruppe: {group_name}")
        print(f"Originalgröße der Gruppe: {group_data.shape}")
        print(f"Inhalte der Gruppe (erste 5 Zeilen):\n{group_data.head()}")

        # Erstelle eine Kopie von group_data, um Änderungen sicher vorzunehmen
        group_data = group_data.copy()

        # Füge 'Close' der Gruppe hinzu, falls es fehlt
        close_added = False
        if 'Close' not in group_data.columns:
            group_data['Close'] = scaled_df['Close']
            close_added = True
            print(f"'Close' wurde temporär zu Gruppe {group_name} hinzugefügt.")

        # Korrelation mit Close analysieren
        cor_with_target = group_data.corr()['Close'].abs()
        print(f"Korrelation mit Close:\n{cor_with_target}\n")

        # Behalte nur relevante Features
        relevant_features = cor_with_target[cor_with_target > 0.2].index.tolist()

        # Fallback: Wähle die zwei höchsten korrelierten Features, falls weniger als zwei übrig bleiben
        if len(relevant_features) < 2:
            top_features = cor_with_target.sort_values(ascending=False).index.tolist()
            relevant_features = top_features[:2]
            print(f"Fallback aktiviert: Die zwei höchsten korrelierten Features wurden ausgewählt: {relevant_features}")

        # Reduzierter DataFrame
        reduced_df = group_data[relevant_features]

        # Entferne 'Close', falls es temporär hinzugefügt wurde
        if close_added and 'Close' in reduced_df.columns:
            reduced_df = reduced_df.drop(columns=['Close'])
            print(f"'Close' wurde aus Gruppe {group_name} entfernt.")

        # Tensor der reduzierten Daten vorbereiten
        data_tensor = torch.tensor(reduced_df.values, dtype=torch.float32).to(device)

        # Autoencoder zur weiteren Reduktion der Feature-Menge trainieren
        encoding_dim = 10  # Ziel-Dimension
        encoded_features = train_autoencoder(data_tensor, encoding_dim=encoding_dim)

        # Endgültige reduzierte Daten als DataFrame speichern
        reduced_features_df = pd.DataFrame(encoded_features.cpu().numpy(),
                                           columns=[f"feature_{i}" for i in range(encoding_dim)])
        print(f"Endgültige reduzierte Features für Gruppe {group_name}: {reduced_features_df.shape}")
        print(f"Inhalte der reduzierten Gruppe (erste 5 Zeilen):\n{reduced_features_df.head()}")

        # Sequenzen generieren
        sequence_length = 50  # Länge der Eingabesequenz in Tagen
        forecast_steps = 30  # Vorhersageperiode
        X, Y = generate_sequences(reduced_features_df, sequence_length=sequence_length, forecast_steps=forecast_steps)

        # Sequenzen speichern
        with open(os.path.join(samples_dir, f"{group_name}_X.pkl"), "wb") as f:
            pickle.dump(X, f)
        with open(os.path.join(samples_dir, f"{group_name}_Y.pkl"), "wb") as f:
            pickle.dump(Y, f)

        print(f"Gruppe {group_name}: Sequenzen gespeichert (X: {X.shape}, Y: {Y.shape})")

    # Speichern des Scalers
    scaler_path = "./Data/Models/scaler.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler und Sequenzen wurden gespeichert.")