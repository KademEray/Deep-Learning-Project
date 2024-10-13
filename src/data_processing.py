import pandas as pd
import os
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tqdm import tqdm
import pickle
import pandas_datareader.data as web
import datetime


# Makroökonomische Daten laden: Leitzins (Fed Funds Rate) und Inflation
def fetch_macro_data(start_date, end_date):
    try:
        # Leitzinsdaten von FRED (Federal Reserve)
        fed_rates = web.DataReader('DFF', 'fred', start_date, end_date)
        fed_rates.rename(columns={'DFF': 'Fed_Rate'}, inplace=True)

        # Inflationsdaten (CPI - Verbraucherpreisindex) von FRED
        cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
        cpi.rename(columns={'CPIAUCSL': 'CPI'}, inplace=True)
        cpi['Inflation'] = cpi['CPI'].pct_change(periods=12) * 100  # Jahr-über-Jahr-Inflation

        # Daten auf tägliche Frequenz resamplen und fehlende Werte auffüllen
        fed_rates_daily = fed_rates.resample('D').ffill()
        cpi_daily = cpi[['Inflation']].resample('D').ffill()

        # Zusammenführen der makroökonomischen Daten
        macro_data = fed_rates_daily.join(cpi_daily, how='left')
        return macro_data
    except Exception as e:
        print(f"Fehler beim Abrufen der makroökonomischen Daten: {e}")
        return None


# Finanzdaten von Yahoo Finance laden
def fetch_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"Keine Daten für {symbol} im Zeitraum {start_date} bis {end_date} verfügbar.")
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except Exception as e:
        print(f"Fehler beim Herunterladen der Daten für {symbol}: {e}")
        return None


def clean_data(df):
    basis_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    df = df.dropna(subset=basis_columns)
    df = df.drop_duplicates()
    return df


def add_technical_indicators(df):
    try:
        close = df['Close']
        if len(close) < 50:
            print(f"Nicht genügend Daten für technische Indikatoren vorhanden.")
            return df

        df['rsi'] = RSIIndicator(close=close, window=14).rsi()
        macd = MACD(close=close)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        bollinger = ta.volatility.BollingerBands(close=close)
        df['bb_mavg'] = bollinger.bollinger_mavg()
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()

        stochastic = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=close, window=14)
        df['stoch'] = stochastic.stoch()
        df['stoch_signal'] = stochastic.stoch_signal()

    except Exception as e:
        print(f"Fehler beim Hinzufügen der technischen Indikatoren: {e}")

    return df


def scale_data(df):
    if df.empty:
        print("Keine Daten zum Skalieren vorhanden.")
        return df, None

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if df[col].notna().any()]

    if not numeric_cols:
        print("Keine numerischen Spalten zum Skalieren vorhanden.")
        return df, None

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler


if __name__ == "__main__":
    symbols = ["BTC-USD"]
    start_date = "2010-01-01"
    end_date = "2022-01-01"

    raw_data_directory = "../data/raw_data"
    processed_data_directory = "../data/processed_data"
    processed_no_indicators_directory = os.path.join(processed_data_directory, "without_indicators")
    processed_with_indicators_directory = os.path.join(processed_data_directory, "with_indicators")

    os.makedirs(raw_data_directory, exist_ok=True)
    os.makedirs(processed_no_indicators_directory, exist_ok=True)
    os.makedirs(processed_with_indicators_directory, exist_ok=True)

    combined_data = pd.DataFrame()

    # Makroökonomische Daten laden
    macro_data = fetch_macro_data(start_date, end_date)

    for symbol in symbols:
        raw_data = fetch_data(symbol, start_date, end_date)
        if raw_data is not None:
            # Makroökonomische Daten in das Finanz-Dataset einfügen
            raw_data = raw_data.set_index('Date').join(macro_data, how='left').reset_index()
            combined_data = pd.concat([combined_data, raw_data], ignore_index=True)

    if not combined_data.empty:
        cleaned_data = clean_data(combined_data)

        chunk_size = 150
        chunks_no_indicators = [cleaned_data.iloc[i:i + chunk_size] for i in range(0, len(cleaned_data), chunk_size)]
        for idx, chunk in enumerate(chunks_no_indicators):
            chunk_file = os.path.join(processed_no_indicators_directory,
                                      f"cleaned_data_no_indicators_part_{idx + 1}.csv")
            chunk.to_csv(chunk_file)

        cleaned_data_with_indicators = add_technical_indicators(cleaned_data)
        scaled_data, scaler = scale_data(cleaned_data_with_indicators)

        # Speichern des Skalers
        scaler_filename = "../models/scaler.pkl"
        os.makedirs(os.path.dirname(scaler_filename), exist_ok=True)
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)

        chunks_with_indicators = [scaled_data.iloc[i:i + chunk_size] for i in range(0, len(scaled_data), chunk_size)]
        for idx, chunk in enumerate(chunks_with_indicators):
            chunk_file = os.path.join(processed_with_indicators_directory,
                                      f"cleaned_data_with_indicators_part_{idx + 1}.csv")
            chunk.to_csv(chunk_file)

        print("Daten erfolgreich gesammelt, gesäubert, mit technischen Indikatoren und Makrodaten erweitert, skaliert und gespeichert.")
