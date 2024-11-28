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
from src.Experiment_6.experiment_6_model import FusionModelWithCNN
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
    df['adx'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['willr'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['cci20'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
    df['mfi'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    df['sma20'] = ta.SMA(df['Close'], timeperiod=20)
    df['ema50'] = ta.EMA(df['Close'], timeperiod=50)
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2,
                                                                   matype=0)
    df['obv'] = ta.OBV(df['Close'], df['Volume'])

    df = df.dropna()
    return df


def load_test_data(symbol, end_date, seq_length):
    """
    Lädt die Testdaten bis zum `end_date` (exklusive) und berechnet technische Indikatoren.
    Behebt Typfehler und stellt sicher, dass der Index und die Spalten einheitlich sind.
    """
    # Verlängere den Zeitraum für die Berechnung der Indikatoren
    extended_start_date = (pd.to_datetime(end_date) - pd.DateOffset(days=seq_length + 50)).strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(days=seq_length)).strftime('%Y-%m-%d')

    # Laden der Hauptsymbol-Daten
    test_df = yf.download(symbol, start=extended_start_date, end=end_date)
    if test_df.empty:
        raise ValueError(f"Keine Daten für das Symbol {symbol} in der Zeitspanne {extended_start_date} bis {end_date}.")

    print(f"Hauptsymbol-Daten ({symbol}) von {start_date} bis {end_date}:\n{test_df.head()}\n")

    # Sicherstellen, dass der Index einheitlich ist
    test_df.index = pd.to_datetime(test_df.index, errors='coerce')

    # Sicherstellen, dass die Spalten numerisch sind
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # Berechnung der technischen Indikatoren
    indicators = calculate_indicators(test_df)
    print(f"Berechnete Indikatoren für das Hauptsymbol ({symbol}):\n{indicators.head()}\n")

    # Zusätzliche Symbole einfügen
    additional_symbols = [
        "GC=F", "SI=F", "CL=F", "BZ=F", "HG=F", "PL=F",
        "DX-Y.NYB", "^GSPC", "^IXIC", "ETH-USD", "BNB-USD",
        "SOL-USD", "ADA-USD", "XRP-USD", "DOGE-USD",
        "DOT-USD", "LTC-USD", "MATIC-USD", "AVAX-USD"
    ]
    for sym in additional_symbols:
        try:
            additional_data = yf.download(sym, start=extended_start_date, end=end_date)
            if additional_data.empty:
                print(f"WARNUNG: Keine Daten für {sym} in der Zeitspanne {extended_start_date} bis {end_date}.")
                test_df[f"{sym}_close"] = np.nan
            else:
                additional_data = additional_data[['Close']].rename(columns={'Close': f"{sym}_close"})
                additional_data.index = pd.to_datetime(additional_data.index)
                test_df = test_df.merge(additional_data, left_index=True, right_index=True, how='left')
        except Exception as e:
            print(f"Fehler beim Herunterladen von Daten für {sym}: {e}")
            test_df[f"{sym}_close"] = np.nan

    # Kombiniere Daten
    test_data = pd.concat([test_df, indicators], axis=1)

    # Entferne doppelte Spalten
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]
    print("Anzahl fehlender Werte pro Spalte vor dropna():\n", test_data.isna().sum())

    # Fehlende Werte auffüllen
    test_data.fillna(method='ffill', inplace=True)
    test_data.fillna(method='bfill', inplace=True)

    print("Anzahl fehlender Werte pro Spalte nach Auffüllen:\n", test_data.isna().sum())

    # Filter auf den gewünschten Zeitraum
    test_data = test_data.loc[test_data.index >= start_date]

    if test_data.empty:
        raise ValueError("Keine gültigen Daten nach dem Entfernen von fehlenden Werten.")

    return test_data


# Funktion zur Vorbereitung der Testsequenzen
def prepare_test_sequences(test_data, scaler, group_features, seq_length):
    """
    Bereitet die Testsequenzen vor, die ausschließlich auf Daten vor dem `end_date` basieren.
    """
    # Skalierung der Daten
    expected_features = scaler.feature_names_in_
    test_data = test_data.reindex(columns=expected_features)
    scaled_data = scaler.transform(test_data)
    scaled_df = pd.DataFrame(scaled_data, columns=expected_features)

    # Erstellen der Sequenzen
    sequences = {}
    for group_name, features in group_features.items():
        group_data = scaled_df[features].copy()
        sequence = group_data.values[-seq_length:]  # Nur die letzten `seq_length` Tage
        sequences[group_name] = torch.tensor(sequence).unsqueeze(0).float()
    return sequences



# Definiere die Feature-Gruppen basierend auf den definierten Gruppen im Hauptskript
group_features = {
    'Standard': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'Indicators_Group_1': [
        'rsi', 'macd', 'macd_signal', 'momentum', 'stochastic', 'adx', 'willr', 'cci20', 'mfi',
        'cci', 'roc', 'bb_upper', 'bb_lower', 'sma20', 'ema50', 'upperband', 'lowerband', 'obv'
    ],
    'Indicators_Group_2': [
        'GC=F_close', 'SI=F_close', 'CL=F_close', 'BZ=F_close', 'HG=F_close', 'PL=F_close',
        'DX-Y.NYB_close', '^GSPC_close', '^IXIC_close', 'ETH-USD_close', 'BNB-USD_close',
        'SOL-USD_close', 'ADA-USD_close', 'XRP-USD_close', 'DOGE-USD_close',
        'DOT-USD_close', 'LTC-USD_close', 'MATIC-USD_close', 'AVAX-USD_close'
    ]
}



# Funktion zum Laden des Modells mit passenden Parametern
def load_model_with_matching_keys(model, model_path):
    # Lade den gespeicherten Zustand des Modells
    saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Erhalte den aktuellen Zustand des Modells
    model_state_dict = model.state_dict()

    # Neue `state_dict`, die nur kompatible Parameter enthält
    new_state_dict = {}

    for name, param in saved_state_dict.items():
        if name in model_state_dict and model_state_dict[name].size() == param.size():
            new_state_dict[name] = param
        else:
            print(f"Parameter {name} ignoriert. Erwartete Größe {model_state_dict.get(name, 'nicht vorhanden')}, "
                  f"aber erhalten {param.size()}.")

    # Lade den gefilterten Zustand
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model

def calculate_baseline_from_independent_data(symbol, start_date, end_date):
    """
    Berechnet den MSE und Durchschnitt der Close-Werte aus unabhängigen Daten.

    Parameters:
        symbol (str): Tickersymbol der Kryptowährung.
        start_date (str): Startdatum für die historischen Daten.
        end_date (str): Enddatum für die historischen Daten.

    Returns:
        dict: Enthält den Durchschnitt der unabhängigen Close-Werte und den MSE basierend darauf.
    """
    try:
        # Lade unabhängige Daten
        independent_data = yf.download(symbol, start=start_date, end=end_date)
        independent_close_prices = independent_data['Close'].dropna()  # Bereinige fehlende Werte

        if independent_close_prices.empty:
            raise ValueError("Unabhängige Daten enthalten keine Werte.")

        # Berechne den Durchschnitt der unabhängigen Close-Werte
        mean_baseline = independent_close_prices.mean()

        return {"mean_baseline": mean_baseline}
    except Exception as e:
        print(f"Fehler beim Berechnen der Baseline: {e}")
        return {"mean_baseline": None}


# Hauptskript zur Ausführung der Vorhersage
if __name__ == "__main__":
    models_dir = "./Data/Models/"
    model_path = os.path.join(models_dir, "fusion_model_final.pth")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    # Datumskonfiguration
    forecast_date = "2023-03-02"
    purchase_date = "2023-02-01"
    seq_length = 50
    hidden_dim = 64
    crypto_symbol = "BTC-USD"

    # Unabhängige Baseline-Daten
    baseline_start_date = "2022-01-01"
    baseline_end_date = "2023-02-01"

    # Lade den Scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    close_scaler = StandardScaler()

    # Lade und bereite die Testdaten vor
    try:
        test_data = load_test_data(crypto_symbol, purchase_date, seq_length)
        close_scaler.fit(test_data[['Close']])  # Fit auf `Close`-Preis
        test_sequences = prepare_test_sequences(test_data, scaler, group_features, seq_length)
    except Exception as e:
        print(f"Fehler beim Laden oder Vorbereiten der Testdaten: {e}")
        exit(1)

    # Lade das Modell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModelWithCNN(hidden_dim=hidden_dim, seq_length=seq_length, group_features=group_features).to(device)
    model = load_model_with_matching_keys(model, model_path)
    model.eval()

    # Eingabedaten für das Modell
    X_standard = test_sequences['Standard'].to(device)
    X_group1 = test_sequences['Indicators_Group_1'].to(device)
    X_group2 = test_sequences['Indicators_Group_2'].to(device)

    # Vorhersage durchführen
    with torch.no_grad():
        output = model(X_standard, X_group1, X_group2)
        close_index = list(scaler.feature_names_in_).index('Close')
        predicted_price_scaled = output[0, -1, close_index].item()
        predicted_price = close_scaler.inverse_transform([[predicted_price_scaled]])[0, 0]

    # Tatsächlicher Preis am Kauf- und Vorhersagedatum abrufen
    try:
        actual_start_price_df = yf.download(crypto_symbol, start=purchase_date,
                                            end=pd.to_datetime(purchase_date) + pd.DateOffset(days=1))
        if actual_start_price_df.empty:
            raise ValueError(f"Keine Daten für den Kaufpreis am {purchase_date} verfügbar.")
        actual_start_price = actual_start_price_df['Close'].iloc[0]

        actual_end_price_df = yf.download(crypto_symbol, start=forecast_date,
                                          end=pd.to_datetime(forecast_date) + pd.DateOffset(days=1))
        if actual_end_price_df.empty:
            raise ValueError(f"Keine Daten für den Vorhersagepreis am {forecast_date} verfügbar.")
        actual_end_price = actual_end_price_df['Close'].iloc[0]
    except Exception as e:
        print(f"Fehler beim Abrufen der tatsächlichen Preisdaten: {e}")
        exit(1)

    # Berechnungen
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
    baseline_result = calculate_baseline_from_independent_data(
        crypto_symbol, baseline_start_date, baseline_end_date
    )

    # Ergebnisse anzeigen
    print(f"Kaufpreis am {purchase_date}: {actual_start_price}")
    print(f"Tatsächlicher Preis am {forecast_date}: {actual_end_price}")
    print(f"Vorhergesagter Preis: {predicted_price}")
    print(f"Tatsächlicher Gewinn: {actual_gain}")
    print(f"Vorhergesagter Gewinn: {predicted_gain}")
    print(f"Baseline Durchschnitt: {baseline_result['mean_baseline']}")
    print(f"MSE im Preis: {mse_price_error}")
    print(f"RMSE im Preis: {rmse_price_error}")
    print(f"R² (Bestimmtheitsmaß): {r2_score:.4f}")
    print(f"Absoluter Fehler: {absolute_error}")
    print(f"Prozentualer Fehler: {percent_error:.6f}%")

    # Kursverlauf und Vorhersage-Plot
    try:
        start_plot_date = (pd.to_datetime(purchase_date) - pd.DateOffset(days=seq_length)).strftime('%Y-%m-%d')
        historical_prices = yf.download(crypto_symbol, start=start_plot_date,
                                         end=pd.to_datetime(forecast_date) + pd.DateOffset(days=1))['Close']

        plt.figure(figsize=(10, 6))
        plt.plot(historical_prices.index, historical_prices.values, label='Tatsächlicher Preisverlauf', color='blue')
        plt.scatter(pd.to_datetime(forecast_date), predicted_price, color='red', label='Vorhergesagter Preis', zorder=5)
        plt.scatter(pd.to_datetime("2023-02-01"), actual_start_price, color='green', label='Kaufpreis am 1. Februar',
                    zorder=5)
        plt.axhline(y=baseline_result['mean_baseline'], color='orange', linestyle='--',
                    label=f"Baseline (Durchschnitt: {baseline_result['mean_baseline']:.2f})")
        plt.title(f"Kursverlauf von {crypto_symbol} mit Vorhersage am {forecast_date}")
        plt.xlabel("Datum")
        plt.ylabel("Preis in USD")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Fehler beim Plotten des Kursverlaufs: {e}")
