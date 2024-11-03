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
from src.standard_model.standard_model import FusionModel

# Funktionen zur Berechnung technischer Indikatoren
def calculate_indicators(df):
    df = df.copy()
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
    df = df.dropna()
    return df

# Funktion zum Laden der Testdaten
def load_test_data(symbol, end_date):
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(days=seq_length + 1)).strftime('%Y-%m-%d')
    test_df = yf.download(symbol, start=start_date, end=end_date)
    test_df.reset_index(inplace=True)
    test_df = test_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    indicators = calculate_indicators(test_df)
    test_df = test_df.rename(columns={'Close': 'Adj Close'})
    test_df['Close'] = test_df['Adj Close']
    test_data = pd.concat([test_df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']], indicators], axis=1)
    test_data = test_data.loc[:, ~test_data.columns.duplicated()].dropna()
    return test_data

# Funktion zur Vorbereitung der Testsequenzen
def prepare_test_sequences(test_data, scaler, group_features):
    expected_features = scaler.feature_names_in_
    test_data = test_data.reindex(columns=expected_features)
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
    'Indicators_Group_1': ['rsi', 'macd', 'macd_signal', 'momentum', 'stochastic'],
    'Indicators_Group_2': ['cci', 'roc', 'bb_upper', 'bb_lower']
}

# Hauptskript zur Ausführung der Vorhersage
if __name__ == "__main__":
    models_dir = "../../Data/Models/"
    model_path = os.path.join(models_dir, "fusion_model_final.pth")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    # Vorhersagedatum und Symbole
    forecast_date = "2023-01-30"
    start_date = "2023-01-01"
    crypto_symbol = "BTC-USD"
    seq_length = 50
    hidden_dim = 64

    # Lade den Scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Lade und bereite die Testdaten vor
    test_data = load_test_data(crypto_symbol, start_date)
    test_sequences = prepare_test_sequences(test_data, scaler, group_features)

    # Lade das Modell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(hidden_dim=hidden_dim, seq_length=seq_length).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Bereite die Eingabedaten für das Modell vor
    X_standard = test_sequences['Standard'].to(device)
    X_group1 = test_sequences['Indicators_Group_1'].to(device)
    X_group2 = test_sequences['Indicators_Group_2'].to(device)

    # Vorhersage durchführen
    with torch.no_grad():
        output = model(X_standard, X_group1, X_group2)
        predicted_price_scaled = output.item()
        dummy_scaled = np.zeros((1, scaler.n_features_in_))
        close_index = list(scaler.feature_names_in_).index('Close')
        dummy_scaled[0, close_index] = predicted_price_scaled
        predicted_price = scaler.inverse_transform(dummy_scaled)[0, close_index]

    # Tatsächlicher Preis am Startdatum und Vorhersagedatum
    actual_start_price_df = yf.download(crypto_symbol, start=start_date, end="2023-01-02")
    actual_start_price = actual_start_price_df['Close'].iloc[0]

    actual_end_price_df = yf.download(crypto_symbol, start=forecast_date, end="2023-01-31")
    actual_end_price = actual_end_price_df['Close'].iloc[0]

    # Berechne tatsächlichen und vorhergesagten Gewinn
    actual_gain = actual_end_price - actual_start_price
    predicted_gain = predicted_price - actual_start_price

    # MSE und RMSE für die Preisvorhersage berechnen
    mse_price_error = F.mse_loss(torch.tensor([predicted_price]), torch.tensor([actual_end_price])).item()
    rmse_price_error = mse_price_error ** 0.5

    # Berechne absoluten und prozentualen Fehler
    absolute_error = abs(predicted_price - actual_end_price)
    percent_error = (absolute_error / actual_end_price) * 100

    # Ergebnisse anzeigen
    print(f"Kaufpreis am {start_date}: {actual_start_price}")
    print(f"Tatsächlicher Preis am {forecast_date}: {actual_end_price}")
    print(f"Vorhergesagter Preis: {predicted_price}")
    print(f"Tatsächlicher Gewinn: {actual_gain}")
    print(f"Vorhergesagter Gewinn: {predicted_gain}")
    print(f"MSE im Preis: {mse_price_error}")
    print(f"RMSE im Preis: {rmse_price_error}")
    print(f"Absoluter Fehler: {absolute_error}")
    print(f"Prozentualer Fehler: {percent_error:.2f}%")

    # Kursverlauf und Vorhersage-Plot
    start_plot_date = (pd.to_datetime(forecast_date) - pd.DateOffset(days=30)).strftime('%Y-%m-%d')
    historical_prices = yf.download(crypto_symbol, start=start_plot_date, end=pd.to_datetime(forecast_date) + pd.DateOffset(days=1))['Close']

    plt.figure(figsize=(10, 6))
    plt.plot(historical_prices.index, historical_prices.values, label='Tatsächlicher Preisverlauf', color='blue')
    plt.scatter(pd.to_datetime(forecast_date), predicted_price, color='red', label='Vorhergesagter Preis', zorder=5)
    plt.title(f"Kursverlauf von {crypto_symbol} mit Vorhersage am {forecast_date}")
    plt.xlabel("Datum")
    plt.ylabel("Preis in USD")
    plt.legend()
    plt.grid(True)
    plt.show()
