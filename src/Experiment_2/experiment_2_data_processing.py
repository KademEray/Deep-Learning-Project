import os

import talib as ta
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
import pickle
import torch

# Funktionen zur Berechnung technischer Indikatoren
def calculate_indicators(df):
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

    # Zusätzliche Indikatoren mit TA-Lib für Gruppe 1 und 2
    # Für Gruppe 1
    df['adx'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['willr'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['cci20'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
    df['mfi'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)

    # Für Gruppe 2
    df['sma20'] = ta.SMA(df['Close'], timeperiod=20)
    df['ema50'] = ta.EMA(df['Close'], timeperiod=50)
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['obv'] = ta.OBV(df['Close'], df['Volume'])

    df = df.dropna()
    return df

# Funktion zur Sequenzierung der Daten
def generate_sequences(data, sequence_length=50, forecast_steps=30, target_feature_dim=15):
    X, Y = [], []
    for i in range(len(data) - sequence_length - forecast_steps):
        x_seq = data[i:i + sequence_length].values
        y_seq = data.iloc[i + sequence_length:i + sequence_length + forecast_steps].values

        # Zusicherung der richtigen Spaltenanzahl
        x_seq = x_seq[:, :target_feature_dim]
        y_seq = y_seq[:, :target_feature_dim]

        X.append(x_seq)
        Y.append(y_seq)

    # Rückgabe von Debug-Informationen und die Tensoren
    X, Y = np.array(X), np.array(Y)
    print(f"Debug: Final X shape {X.shape}, Y shape {Y.shape}")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def clean_data(df):
    df = df.drop_duplicates().interpolate()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Hauptskript zur Verarbeitung und Gruppierung der Daten
if __name__ == "__main__":
    output_dir = "./Data/"
    samples_dir = "./Data/Samples/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    crypto_symbol = "BTC-USD"
    start_date, end_date = "2010-01-01", "2023-02-01"

    df = yf.download(crypto_symbol, start=start_date, end=end_date, progress=False)
    df.reset_index(inplace=True)
    df = clean_data(df)

    # Berechnung der Indikatoren und Kombination mit Standard-Features
    indicators = calculate_indicators(df)
    df = pd.concat([df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']], indicators], axis=1).dropna()
    print(f"Debug: {crypto_symbol} dataset size after indicators: {df.shape}")

    # Skalierung der Daten
    scaler = StandardScaler()
    # Skalierung mit allen relevanten numerischen Spalten (ohne 'Date')
    scaled_data = scaler.fit_transform(df.drop(columns=['Date']))

    # Überprüfen der Spaltenanzahl vor Erstellung des DataFrames
    if scaled_data.shape[1] == df.shape[1] - 1:
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns[1:])  # Spalten ohne 'Date' verwenden
    else:
        print(f"Warnung: Erwartete Spaltenanzahl {df.shape[1] - 1}, aber erhalten {scaled_data.shape[1]}")
        scaled_df = pd.DataFrame(scaled_data, columns=df.drop(columns=['Date']).columns)

    print(f"Debug: Scaled data shape: {scaled_df.shape}")

    # Feature-Gruppen definieren
    standard_features = scaled_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    indicator_group_1 = scaled_df[
        ['rsi', 'macd', 'macd_signal', 'momentum', 'stochastic', 'adx', 'willr', 'cci20', 'mfi']]
    indicator_group_2 = scaled_df[
        ['cci', 'roc', 'bb_upper', 'bb_lower', 'sma20', 'ema50', 'upperband', 'lowerband', 'obv']]

    # Sequenzen für jede Gruppe generieren und speichern
    groups = {
        'Standard': standard_features,
        'Indicators_Group_1': indicator_group_1,
        'Indicators_Group_2': indicator_group_2,
    }

    for group_name, group_data in groups.items():
        target_feature_dim = group_data.shape[1]
        X, Y = generate_sequences(group_data, sequence_length=50, forecast_steps=30,
                                  target_feature_dim=target_feature_dim)

        # Speichern der Daten als .pkl
        with open(os.path.join(samples_dir, f"{group_name}_X.pkl"), 'wb') as f:
            pickle.dump(X, f)
        with open(os.path.join(samples_dir, f"{group_name}_Y.pkl"), 'wb') as f:
            pickle.dump(Y, f)
        print(f"Debug: Saved {group_name} sequences X shape: {X.shape}, Y shape: {Y.shape}")

    # Speichern des Scalers
    scaler_path = "./Data/Models/scaler.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler wurde gespeichert unter", scaler_path)
