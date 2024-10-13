import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import ta
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scipy.signal import argrelextrema

# Datenabruf
def fetch_test_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Technische Indikatoren hinzufügen
def add_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(close=df['Close'])
    df['bb_mavg'] = bollinger.bollinger_mavg()
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()

    stochastic = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['stoch'] = stochastic.stoch()
    df['stoch_signal'] = stochastic.stoch_signal()

    df.fillna(0, inplace=True)
    return df

# Fehlende Spalten ergänzen und Reihenfolge sicherstellen
def add_missing_columns(df, required_columns):
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[required_columns]
    return df

# Inverse Transformation
def inverse_transform(predictions, scaler, feature_columns):
    dummy = np.zeros((len(predictions), len(feature_columns)))
    close_index = feature_columns.index('Close')
    dummy[:, close_index] = predictions.flatten()
    inverted = scaler.inverse_transform(dummy)
    return inverted[:, close_index]

# Beste Buy- und Sell-Signale finden und optimieren
def find_best_trades(predicted_prices, actual_prices, date_values, min_distance=10, volatility_threshold=0.01):
    buy_signals = argrelextrema(predicted_prices, np.less)[0]
    sell_signals = argrelextrema(predicted_prices, np.greater)[0]

    best_trades = []
    last_trade_end = -1  # Zum Verhindern von überschneidenden Trades

    # Berechne die Volatilität als Standardabweichung der Preisänderungen
    price_volatility = np.std(np.diff(actual_prices)) / np.mean(actual_prices)

    # Dynamischer Mindestabstand basierend auf Volatilität
    dynamic_min_distance = max(min_distance, int(volatility_threshold * len(actual_prices)))

    for buy in buy_signals:
        if buy < last_trade_end:
            continue  # Überspringe, wenn sich der Buy innerhalb eines laufenden Trades befindet

        best_sell = None
        best_profit = -np.inf

        # Suche nach dem besten Sell-Signal nach dem Buy-Signal
        for sell in sell_signals:
            if sell > buy and (sell - buy) > dynamic_min_distance:
                profit = actual_prices[sell] - actual_prices[buy]
                if profit > best_profit:
                    best_sell = sell
                    best_profit = profit

        # Füge den besten Trade zur Liste hinzu, falls er einen Gewinn bringt
        if best_sell is not None and best_profit > 0:
            best_trades.append((buy, best_sell, best_profit))
            last_trade_end = best_sell  # Setze das Ende des aktuellen Trades

            # Entferne Signale in der aktuellen Buy-Sell-Phase, um Überschneidungen zu verhindern
            buy_signals = buy_signals[buy_signals > best_sell]
            sell_signals = sell_signals[sell_signals > best_sell]

    # Sortiere die Trades nach Gewinn und behalte maximal 3 Trades
    best_trades = sorted(best_trades, key=lambda x: x[2], reverse=True)[:3]
    return best_trades

# Genauigkeit berechnen
def calculate_accuracy(actual_prices, predicted_prices):
    correct_trend = 0
    total_predictions = len(actual_prices) - 1

    for i in range(1, len(actual_prices)):
        actual_trend = actual_prices[i] - actual_prices[i - 1]
        predicted_trend = predicted_prices[i] - predicted_prices[i - 1]

        if (actual_trend >= 0 and predicted_trend >= 0) or (actual_trend < 0 and predicted_trend < 0):
            correct_trend += 1

    accuracy = correct_trend / total_predictions
    return accuracy

# Modell evaluieren
def evaluate_model(model_info, data, asset_name):
    model = load_model(model_info["model_path"])
    with open(model_info["scaler_path"], 'rb') as f:
        scaler = pickle.load(f)
    feature_columns = model_info["feature_columns"]

    date_values = data['Date'].values

    data = add_missing_columns(data, feature_columns)
    data = data[feature_columns]

    data_scaled = scaler.transform(data)
    sequences = np.array([data_scaled[i:i + 50] for i in range(len(data_scaled) - 50)])

    predictions = model.predict(sequences)
    predictions_inv = inverse_transform(predictions, scaler, feature_columns)

    actual_prices = data['Close'].values[50:]
    mae = mean_absolute_error(actual_prices, predictions_inv)
    accuracy = calculate_accuracy(actual_prices, predictions_inv)

    print(f"{asset_name} MAE: {mae}")
    print(f"{asset_name} Genauigkeit: {accuracy:.2%}")

    best_trades = find_best_trades(predictions_inv, actual_prices, date_values[50:], min_distance=10)

    plt.figure(figsize=(14, 7))
    plt.plot(date_values[50:], actual_prices, label='Tatsächlicher Preis')
    plt.plot(date_values[50:], predictions_inv, label='Vorhergesagter Preis')

    for trade in best_trades:
        buy, sell, profit = trade
        plt.scatter(date_values[50:][buy], predictions_inv[buy], marker='^', color='g', label=f'Buy ({date_values[50:][buy]})')
        plt.scatter(date_values[50:][sell], predictions_inv[sell], marker='v', color='r', label=f'Sell ({date_values[50:][sell]})')
        print(f"Buy am {date_values[50:][buy]} für {predictions_inv[buy]:.2f} und Sell am {date_values[50:][sell]} für {predictions_inv[sell]:.2f} mit Profit von {profit:.2f}")

    plt.title(f'{asset_name} - {model_info["model_path"]}')
    plt.legend()

    # Genauigkeit im Diagramm anzeigen
    plt.text(0.02, 0.95, f"Genauigkeit: {accuracy:.2%}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.show()

if __name__ == "__main__":
    eth_data = fetch_test_data('ETH-USD', '2022-01-01', '2022-12-01')
    eth_data = add_technical_indicators(eth_data)

    model_info = {
        "model_path": "../models/standard_model/eth_standard_model",
        "scaler_path": "../models/scaler/eth_scaler_with_indicators.pkl",
        "feature_columns": ['Open', 'High', 'Low', 'Close', 'Volume', 'Fed_Rate', 'Inflation', 'rsi', 'macd',
                            'macd_signal', 'bb_mavg', 'bb_high', 'bb_low', 'stoch', 'stoch_signal']
    }

    evaluate_model(model_info, eth_data, "Etherium")

