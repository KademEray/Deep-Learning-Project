# experiment_nvidia.py

import os
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Verwende CUDA, wenn verfügbar
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Laden des trainierten Modells
model_directory = "../models/trained"
model_path = os.path.join(model_directory, "ppo_trading_model_100.zip")
model = PPO.load(model_path, device=device)


# Funktion zum Laden der Testdaten von Yahoo Finance
def load_test_data(symbol, start_date):
    df = yf.download(symbol, start=start_date)
    if not df.empty:
        df.reset_index(inplace=True)
        return df
    else:
        raise ValueError(f"Keine Daten für {symbol} ab {start_date} vorhanden.")


# Funktion zur Vorbereitung der Beobachtung
def prepare_observation(df_row):
    relevant_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    obs = df_row[relevant_columns].values.astype(np.float32)
    padding = np.zeros(27 - len(obs), dtype=np.float32)
    return np.concatenate([obs, padding])


# Funktion zur Durchführung des Experiments
def run_experiment(symbol, start_date):
    df = load_test_data(symbol, start_date)
    df['Date'] = pd.to_datetime(df['Date'])

    initial_balance = 10000
    balance = initial_balance
    position = 0
    entry_price = 0
    net_worths = []
    dates = []
    actual_prices = []
    predicted_actions = []
    accuracies = []

    for index, row in df.iterrows():
        obs = prepare_observation(row)
        action, _ = model.predict(obs, deterministic=True)
        current_price = row['Close']
        dates.append(row['Date'])
        actual_prices.append(current_price)
        predicted_actions.append(action)

        # Handlungslogik basierend auf der Vorhersage
        if action == 1 and balance >= current_price:  # Kaufen
            position = balance // current_price
            balance -= position * current_price
            entry_price = current_price
        elif action == 2 and position > 0:  # Verkaufen
            balance += position * current_price
            position = 0

        # Stop-Loss bei 2% Verlust
        if position > 0 and current_price <= entry_price * 0.98:
            balance += position * current_price
            position = 0

        net_worth = balance + position * current_price
        net_worths.append(net_worth)

        # Berechne die Genauigkeit (einfaches Beispiel)
        accuracy = 1 if (action == 1 and current_price < entry_price) or (
                    action == 2 and current_price > entry_price) else 0
        accuracies.append(accuracy)

    # Plot der Ergebnisse
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label="Tatsächlicher Kurs")
    plt.plot(dates, net_worths, label="Nettovermögen")
    plt.title(f"Prognose für {symbol} ab dem {start_date}")
    plt.xlabel("Datum")
    plt.ylabel("Preis / Nettovermögen")
    plt.legend()
    plt.show()

    # Genauigkeit der Vorhersage
    overall_accuracy = np.mean(accuracies) * 100
    print(f"Die Genauigkeit der Prognose beträgt {overall_accuracy:.2f}%")


# Durchführung des Experiments für Nvidia
run_experiment("NVDA", "2023-01-01")
