import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import ta
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# Fehlende Spalten ergänzen
def add_missing_columns(df, required_columns):
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[required_columns]
    return df

# Inverse Transformation der Vorhersagen
def inverse_transform(predictions, scaler, feature_columns):
    dummy = np.zeros((len(predictions), len(feature_columns)))
    close_index = feature_columns.index('Close')
    predictions = predictions.reshape(-1, 1)
    dummy[:, close_index] = predictions.flatten()
    inverted = scaler.inverse_transform(dummy)
    return inverted[:, close_index]

# Berechnung der Genauigkeit (Accuracy)
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

# Buy- und Sell-Signale basierend auf den vorhergesagten Preisen
def find_best_single_trade(predicted_prices, actual_prices, date_values):
    buy_signals = argrelextrema(predicted_prices, np.less)[0]
    sell_signals = argrelextrema(predicted_prices, np.greater)[0]

    best_buy = None
    best_sell = None
    best_profit = -np.inf

    for buy in buy_signals:
        for sell in sell_signals:
            if sell > buy:
                profit = actual_prices[sell] - actual_prices[buy]
                if profit > best_profit:
                    best_buy = buy
                    best_sell = sell
                    best_profit = profit

    return best_buy, best_sell, best_profit

# Monte Carlo Simulation
def monte_carlo_simulation(model, input_sequence, scaler, feature_columns, num_simulations=100, forecast_steps=30):
    simulations = []
    for i in range(num_simulations):
        noisy_input_sequence = input_sequence + np.random.normal(0, 0.01, input_sequence.shape)
        predictions = model.predict(noisy_input_sequence[np.newaxis, :, :])
        predictions_inv = inverse_transform(predictions.flatten(), scaler, feature_columns)
        simulations.append(predictions_inv[:forecast_steps])

    return np.array(simulations)

# Modell evaluieren mit Monte Carlo Simulation
def evaluate_model(model_info, data, actual_data_2022, asset_name, prediction_steps=30, num_simulations=100):
    model = load_model(model_info["model_path"])
    with open(model_info["scaler_path"], 'rb') as f:
        scaler = pickle.load(f)
    feature_columns = model_info["feature_columns"]

    actual_dates_2022 = actual_data_2022['Date'].values
    actual_prices_2022 = actual_data_2022['Close'].values

    prediction_start_idx = len(data) - prediction_steps
    data = add_missing_columns(data, feature_columns)
    data = data[feature_columns]
    data_scaled = scaler.transform(data)

    input_sequence = data_scaled[prediction_start_idx - 50:prediction_start_idx]

    # Monte Carlo Simulation
    simulations = monte_carlo_simulation(model, input_sequence, scaler, feature_columns, num_simulations, prediction_steps)

    mean_prediction = np.mean(simulations, axis=0)
    std_prediction = np.std(simulations, axis=0)

    prediction_steps = min(len(actual_dates_2022), len(mean_prediction), prediction_steps)

    # Berechnung der Genauigkeit mit RMSE und MAE
    rmse = np.sqrt(mean_squared_error(actual_prices_2022[:prediction_steps], mean_prediction[:prediction_steps]))
    mae = mean_absolute_error(actual_prices_2022[:prediction_steps], mean_prediction[:prediction_steps])
    accuracy = calculate_accuracy(actual_prices_2022[:prediction_steps], mean_prediction[:prediction_steps])

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Genauigkeit (Steigung/Senkung): {accuracy:.2%}")

    buy_signal, sell_signal, profit = find_best_single_trade(mean_prediction[:prediction_steps],
                                                             actual_prices_2022[:prediction_steps],
                                                             actual_dates_2022[:prediction_steps])

    prediction_df = pd.DataFrame({
        'Datum': actual_dates_2022[:prediction_steps],
        'Tatsächlicher Preis': actual_prices_2022[:prediction_steps],
        'Vorhergesagter Preis (Monte Carlo Mean)': mean_prediction[:prediction_steps]
    })
    prediction_df.to_csv('btc_predictions_monte_carlo.csv', index=False)
    print("Vorhersagen und tatsächliche Preise wurden in 'btc_predictions_monte_carlo.csv' gespeichert.")

    # Plotten der Vorhersagen und tatsächlichen Preise mit Konfidenzintervall
    plt.figure(figsize=(14, 7))
    plt.plot(actual_dates_2022[:prediction_steps], actual_prices_2022[:prediction_steps], label='Tatsächlicher Preis')
    plt.plot(actual_dates_2022[:prediction_steps], mean_prediction[:prediction_steps], label='Vorhergesagter Preis (Monte Carlo Mean)')
    plt.fill_between(actual_dates_2022[:prediction_steps],
                     mean_prediction[:prediction_steps] - 1.96 * std_prediction[:prediction_steps],
                     mean_prediction[:prediction_steps] + 1.96 * std_prediction[:prediction_steps],
                     color='gray', alpha=0.3, label='95% Confidence Interval')

    if buy_signal is not None and sell_signal is not None:
        plt.scatter(actual_dates_2022[buy_signal], mean_prediction[buy_signal], marker='^', color='g', label='Buy Signal')
        plt.scatter(actual_dates_2022[sell_signal], mean_prediction[sell_signal], marker='v', color='r', label='Sell Signal')
        plt.plot([], [], ' ', label=f'Profit: {profit:.2f}')

    plt.text(0.02, 0.95, f"Genauigkeit: {accuracy:.2%}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.title(f'{asset_name} - Monte Carlo Vorhersage für 1 Monat in 2022')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Lade Daten bis Ende 2021
    btc_data = fetch_test_data('BTC-USD', '2010-01-01', '2021-12-31')
    btc_data_with_indicators = add_technical_indicators(btc_data.copy())

    # Lade nur die Daten für 2022 zur Evaluierung (nicht für Vorhersage)
    btc_data_2022 = fetch_test_data('BTC-USD', '2022-01-01', '2022-01-31')
    btc_data_2022 = add_technical_indicators(btc_data_2022)

    model_info = {
        "model_path": "../../models/standard_model/btc_standard_model.h5",  # Korrigierter Pfad
        "scaler_path": "../../models/scaler/btc_scaler_with_indicators.pkl",
        "feature_columns": ['Open', 'High', 'Low', 'Close', 'Volume', 'Fed_Rate', 'Inflation', 'rsi', 'macd',
                            'macd_signal', 'bb_mavg', 'bb_high', 'bb_low', 'stoch', 'stoch_signal']
    }

    evaluate_model(model_info, btc_data_with_indicators, btc_data_2022, "Bitcoin", prediction_steps=30)
