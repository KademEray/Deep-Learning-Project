import os
import sys
import numpy as np
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.Experiment_1.experiment_1_model import FusionModel


# Setze Zufallszahlenseeds, um reproduzierbare Ergebnisse sicherzustellen
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# Falls CUDA verfügbar ist, setze zusätzliche Seeds für GPU-Operationen
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)  # Für Multi-GPU-Umgebungen

# Zusätzliche Konfigurationen, um deterministisches Verhalten sicherzustellen
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Funktion zur Berechnung technischer Indikatoren basierend auf Preisdaten
def calculate_indicators(df):
    df = df.copy()  # Erstelle eine Kopie des DataFrames, um die Originaldaten nicht zu verändern

    # Berechnung verschiedener technischer Indikatoren
    df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()  # Relative Strength Index (Momentum-Indikator)
    df['macd'] = MACD(df['Close']).macd()  # Moving Average Convergence Divergence
    df['macd_signal'] = MACD(df['Close']).macd_signal()  # Signal-Linie von MACD
    df['cci'] = CCIIndicator(df['High'], df['Low'], df['Close']).cci()  # Commodity Channel Index
    df['roc'] = ROCIndicator(df['Close'], window=12).roc()  # Rate of Change (Momentum-Indikator)
    df['momentum'] = df['Close'].diff(1)  # Einfache Momentum-Berechnung
    df['stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()  # Stochastic Oscillator
    bb = BollingerBands(df['Close'])  # Bollinger Bands
    df['bb_upper'] = bb.bollinger_hband()  # Oberes Band
    df['bb_lower'] = bb.bollinger_lband()  # Unteres Band

    df = df.dropna()  # Entferne Zeilen mit fehlenden Werten
    return df


# Funktion zum Laden von Testdaten für eine bestimmte Kryptowährung
def load_test_data(symbol, seq_length, forecast_start="2023-02-01"):
    # Berechne Startdatum basierend auf der Sequenzlänge
    start_date = (pd.to_datetime(forecast_start) - pd.DateOffset(days=seq_length)).strftime('%Y-%m-%d')

    # Lade historische Preisdaten mit yfinance
    test_df = yf.download(symbol, start=start_date, end=forecast_start)
    test_df.reset_index(inplace=True)  # Setze den Index zurück
    test_df = test_df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Behalte nur relevante Spalten

    # Berechne technische Indikatoren und füge sie zum DataFrame hinzu
    indicators = calculate_indicators(test_df)
    test_df = test_df.rename(columns={'Close': 'Adj Close'})  # Passe den Namen der Close-Spalte an
    test_df['Close'] = test_df['Adj Close']  # Stelle sicher, dass 'Close' verwendet wird

    # Kombiniere Preisdaten mit Indikatoren
    test_data = pd.concat([test_df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']], indicators], axis=1)
    test_data = test_data.loc[:, ~test_data.columns.duplicated()].dropna()  # Entferne doppelte Spalten und NaNs
    return test_data


# Funktion zur Vorbereitung von Eingabesequenzen für das Modell
def prepare_test_sequences(test_data, scaler, group_features, seq_length):
    # Standardisiere die Daten basierend auf dem geladenen Scaler
    expected_features = scaler.feature_names_in_  # Erwarte bestimmte Features
    test_data = test_data.reindex(columns=expected_features)  # Sortiere Spalten in der richtigen Reihenfolge
    scaled_data = scaler.transform(test_data)  # Skaliere die Daten
    scaled_df = pd.DataFrame(scaled_data, columns=expected_features)

    # Erstelle Sequenzen für jedes Feature-Set
    sequences = {}
    for group_name, features in group_features.items():
        group_data = scaled_df[features].copy()  # Wähle die Features für die Gruppe aus
        sequence = group_data.values[-seq_length:]  # Schneide auf die Sequenzlänge zu
        sequences[group_name] = torch.tensor(sequence).unsqueeze(0).float()  # Konvertiere in Tensor
    return sequences


# Definiere die Feature-Gruppen, die als Eingaben verwendet werden
group_features = {
    'Standard': ['Open', 'High', 'Low', 'Close', 'Volume'],  # Grundlegende Preisdaten
    'Indicators_Group_1': ['rsi', 'macd', 'macd_signal', 'momentum', 'stochastic'],  # Momentum-Indikatoren
    'Indicators_Group_2': ['cci', 'roc', 'bb_upper', 'bb_lower']  # Volatilitäts-Indikatoren
}

# Hauptskript
if __name__ == "__main__":
    # Definiere Speicherorte für das Modell und den Scaler
    models_dir = "./Data/Models/"
    model_path = os.path.join(models_dir, "fusion_model_final.pth")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    # Setze Modellparameter und Symbol
    forecast_date = "2023-03-02"
    seq_length = 50
    hidden_dim = 64
    crypto_symbol = "BTC-USD"

    # Lade den Scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Lade und bereite die Testdaten vor
    try:
        test_data = load_test_data(crypto_symbol, seq_length, forecast_start="2023-02-01")
        close_scaler = StandardScaler()  # Skaler für 'Close' Preise
        close_scaler.fit(test_data[['Close']])  # Passe Skaler an 'Close' an
        test_sequences = prepare_test_sequences(test_data, scaler, group_features, seq_length)
    except Exception as e:
        print(f"Fehler beim Laden oder Vorbereiten der Testdaten: {e}")
        exit(1)

    # Lade das trainierte Modell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Wähle Gerät
    model = FusionModel(hidden_dim=hidden_dim, seq_length=seq_length).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Bereite Eingaben für das Modell vor
    X_standard = test_sequences['Standard'].to(device)
    X_group1 = test_sequences['Indicators_Group_1'].to(device)
    X_group2 = test_sequences['Indicators_Group_2'].to(device)

    # Führe die Vorhersage durch
    with torch.no_grad():
        output = model(X_standard, X_group1, X_group2)
        close_index = list(scaler.feature_names_in_).index('Close')
        predicted_price_scaled = output[0, -1, close_index].item()
        predicted_price = close_scaler.inverse_transform([[predicted_price_scaled]])[0, 0]

    # Lade tatsächliche Preise
    try:
        actual_start_price_df = yf.download(crypto_symbol, start="2023-02-01", end="2023-02-02")
        actual_start_price = actual_start_price_df['Close'].iloc[0]

        actual_end_price_df = yf.download(crypto_symbol, start=forecast_date, end="2023-03-03")
        actual_end_price = actual_end_price_df['Close'].iloc[0]
    except Exception as e:
        print(f"Fehler beim Abrufen der tatsächlichen Preisdaten: {e}")
        exit(1)

    # Berechne Gewinn und Fehlermaße
    actual_gain = actual_end_price - actual_start_price
    predicted_gain = predicted_price - actual_start_price
    mse_price_error = F.mse_loss(torch.tensor([predicted_price]), torch.tensor([actual_end_price])).item()
    rmse_price_error = mse_price_error ** 0.5
    absolute_error = abs(predicted_price - actual_end_price)
    percent_error = (absolute_error / actual_end_price) * 100
    # Berechne den Durchschnitt der tatsächlichen Zielwerte
    mean_actual = torch.tensor([actual_start_price + actual_end_price]).mean()

    # Berechne SS_total und SS_residual korrekt
    ss_total = ((torch.tensor([actual_end_price]) - mean_actual) ** 2).sum().item()
    ss_residual = ((torch.tensor([actual_end_price]) - torch.tensor([predicted_price])) ** 2).sum().item()

    # R^2 berechnen
    r2_score = 1 - (ss_residual / ss_total)

    # Drucke Ergebnisse
    print(f"Kaufpreis am 2023-02-01: {actual_start_price}")
    print(f"Tatsächlicher Preis am {forecast_date}: {actual_end_price}")
    print(f"Vorhergesagter Preis: {predicted_price}")
    print(f"Tatsächlicher Gewinn: {actual_gain}")
    print(f"Vorhergesagter Gewinn: {predicted_gain}")
    print(f"MSE im Preis: {mse_price_error}")
    print(f"RMSE im Preis: {rmse_price_error}")
    print(f"Absoluter Fehler: {absolute_error}")
    print(f"Prozentualer Fehler: {percent_error:.4f}%")
    print(f"R² (Bestimmtheitsmaß): {r2_score:.4f}")

    # Plot des Kursverlaufs
    try:
        start_plot_date = (pd.to_datetime("2023-02-01") - pd.DateOffset(days=seq_length)).strftime('%Y-%m-%d')
        historical_prices = yf.download(crypto_symbol, start=start_plot_date,
                                        end=pd.to_datetime(forecast_date) + pd.DateOffset(days=1))['Close']

        plt.figure(figsize=(10, 6))
        plt.plot(historical_prices.index, historical_prices.values, label='Tatsächlicher Preisverlauf', color='blue')
        plt.scatter(pd.to_datetime(forecast_date), predicted_price, color='red', label='Vorhergesagter Preis', zorder=5)
        plt.scatter(pd.to_datetime("2023-02-01"), actual_start_price, color='green', label='Kaufpreis am 1. Februar',
                    zorder=5)
        plt.title(f"Kursverlauf von {crypto_symbol} mit Vorhersage am {forecast_date}")
        plt.xlabel("Datum")
        plt.ylabel("Preis in USD")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Fehler beim Plotten des Kursverlaufs: {e}")
