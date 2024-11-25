import os
import numpy as np
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn.functional as F
from src.Experiment_2.experiment_2_model import FusionModel
import talib as ta
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

# Funktionen zur Berechnung technischer Indikatoren
def calculate_indicators(df):
    df = df.copy()

    # Bisherige Indikatoren
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
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['obv'] = ta.OBV(df['Close'], df['Volume'])

    df = df.dropna()
    return df

# Funktion zum Laden der Testdaten
def load_test_data(symbol, end_date):
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(days=seq_length + 1)).strftime('%Y-%m-%d')
    test_df = yf.download(symbol, start=start_date, end=end_date)
    test_df.reset_index(inplace=True)
    test_df = test_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    indicators = calculate_indicators(test_df)
    test_df = test_df.rename(columns={'Close': 'Adj Close'})  # 'Adj Close' aus 'Close' erstellen
    test_df['Close'] = test_df['Adj Close']  # Sicherstellen, dass 'Close' vorhanden bleibt
    test_data = pd.concat([test_df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']], indicators], axis=1)
    test_data = test_data.loc[:, ~test_data.columns.duplicated()].dropna()
    return test_data

# Funktion zur Vorbereitung der Testsequenzen
def prepare_test_sequences(test_data, scaler, group_features):
    expected_features = scaler.feature_names_in_
    test_data = test_data.reindex(columns=expected_features)  # Spaltenreihenfolge anpassen
    scaled_data = scaler.transform(test_data)
    scaled_df = pd.DataFrame(scaled_data, columns=expected_features)

    sequences = {}
    for group_name, features in group_features.items():
        group_data = scaled_df[features].copy()
        sequence = group_data.values[-seq_length:]
        sequences[group_name] = torch.tensor(sequence).unsqueeze(0).float()
    return sequences

# Definiere die Feature-Gruppen
group_features = {
    'Standard': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'Indicators_Group_1': ['rsi', 'macd', 'macd_signal', 'momentum', 'stochastic', 'adx', 'willr', 'cci20', 'mfi'],
    'Indicators_Group_2': ['cci', 'roc', 'bb_upper', 'bb_lower', 'sma20', 'ema50', 'upperband', 'lowerband', 'obv']
}

# Hauptskript zur Ausführung der Vorhersage
if __name__ == "__main__":
    models_dir = "./Data/Models/"
    model_path = os.path.join(models_dir, "fusion_model_final.pth")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    # Setze Vorhersage- und Analysezeiträume
    forecast_date = "2023-03-02"
    seq_length = 50
    hidden_dim = 64
    crypto_symbol = "BTC-USD"

    # Lade den Scaler und initialisiere einen neuen Scaler nur für das `Close`-Feature
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    close_scaler = StandardScaler()

    # Lade und bereite die Testdaten vor
    try:
        test_data = load_test_data(crypto_symbol, "2023-02-01")
        close_scaler.fit(test_data[['Close']])  # Nur den Close-Wert skalieren
        test_sequences = prepare_test_sequences(test_data, scaler, group_features)
    except Exception as e:
        print(f"Fehler beim Laden oder Vorbereiten der Testdaten: {e}")
        exit(1)

    # Lade das Modell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(hidden_dim=hidden_dim, seq_length=seq_length).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Bereite die Eingabedaten für das Modell vor
    X_standard = test_sequences['Standard'].to(device)
    X_group1 = test_sequences['Indicators_Group_1'].to(device)
    X_group2 = test_sequences['Indicators_Group_2'].to(device)

    # Vorhersage durchführen
    with torch.no_grad():
        output = model(X_standard, X_group1, X_group2)
        close_index = list(scaler.feature_names_in_).index('Close')
        predicted_price_scaled = output[0, -1, close_index].item()
        predicted_price = close_scaler.inverse_transform([[predicted_price_scaled]])[0, 0]

    # Tatsächlicher Preis am Kaufdatum (1. März 2023) und Verkaufsdatum (30. März 2023)
    try:
        actual_start_price_df = yf.download(crypto_symbol, start="2023-02-01", end="2023-02-02")
        if actual_start_price_df.empty:
            raise ValueError("Keine Daten für den Startpreis verfügbar.")
        actual_start_price = actual_start_price_df['Close'].iloc[0]

        actual_end_price_df = yf.download(crypto_symbol, start="2023-03-02", end="2023-03-03")
        if actual_end_price_df.empty:
            raise ValueError("Keine Daten für den Endpreis verfügbar.")
        actual_end_price = actual_end_price_df['Close'].iloc[0]
    except Exception as e:
        print(f"Fehler beim Abrufen der tatsächlichen Preisdaten: {e}")
        exit(1)

    # Berechne tatsächlichen und vorhergesagten Gewinn
    actual_gain = actual_end_price - actual_start_price
    predicted_gain = predicted_price - actual_start_price

    # MSE und RMSE für die Preisvorhersage berechnen
    mse_price_error = F.mse_loss(torch.tensor([predicted_price]), torch.tensor([actual_end_price])).item()
    rmse_price_error = mse_price_error ** 0.5

    # Berechne absoluten und prozentualen Fehler
    absolute_error = abs(predicted_price - actual_end_price)
    percent_error = (absolute_error / actual_end_price) * 100
    # Berechne den Durchschnitt der tatsächlichen Zielwerte
    mean_actual = torch.tensor([actual_start_price + actual_end_price]).mean()
    # Berechne SS_total und SS_residual korrekt
    ss_total = ((torch.tensor([actual_end_price]) - mean_actual) ** 2).sum().item()
    ss_residual = ((torch.tensor([actual_end_price]) - torch.tensor([predicted_price])) ** 2).sum().item()
    # R^2 berechnen
    r2_score = 1 - (ss_residual / ss_total)

    # Ergebnisse anzeigen
    print(f"Kaufpreis am 01.02.2023: {actual_start_price}")
    print(f"Tatsächlicher Preis am 02.03.2023: {actual_end_price}")
    print(f"Vorhergesagter Preis: {predicted_price}")
    print(f"Tatsächlicher Gewinn: {actual_gain}")
    print(f"Vorhergesagter Gewinn: {predicted_gain}")
    print(f"MSE im Preis: {mse_price_error}")
    print(f"RMSE im Preis: {rmse_price_error}")
    print(f"R² (Bestimmtheitsmaß): {r2_score:.4f}")
    print(f"Absoluter Fehler: {absolute_error}")
    print(f"Prozentualer Fehler: {percent_error:.4f}%")

    # Kursverlauf und Vorhersage-Plot
    try:
        start_plot_date = (pd.to_datetime("2023-02-01") - pd.DateOffset(days=seq_length)).strftime('%Y-%m-%d')
        historical_prices = yf.download(crypto_symbol, start=start_plot_date, end="2023-03-03")['Close']

        plt.figure(figsize=(10, 6))
        plt.plot(historical_prices.index, historical_prices.values, label='Tatsächlicher Preisverlauf', color='blue')
        plt.scatter(pd.to_datetime("2023-03-02"), predicted_price, color='red',
                    label='Vorhergesagter Preis am 02.03.2023', zorder=5)
        plt.scatter(pd.to_datetime("2023-02-01"), actual_start_price, color='green', label='Kaufpreis am 01.02.2023',
                    zorder=5)
        plt.title(f"Kursverlauf von {crypto_symbol} mit Vorhersage am 02.03.2023")
        plt.xlabel("Datum")
        plt.ylabel("Preis in USD")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Fehler beim Plotten des Kursverlaufs: {e}")
