import os

import numpy as np
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import talib
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas_datareader.data as web
import shutil


# Makroökonomische Daten abrufen
def fetch_macro_data(start_date, end_date):
    try:
        fed_rates = web.DataReader('DFF', 'fred', start_date, end_date).rename(columns={'DFF': 'Fed_Rate'})
        cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
        cpi['Inflation'] = cpi['CPIAUCSL'].pct_change(periods=12) * 100
        fed_rates_daily = fed_rates.resample('D').ffill()
        cpi_daily = cpi[['Inflation']].resample('D').ffill()
        return fed_rates_daily.join(cpi_daily, how='left')
    except Exception as e:
        print(f"Fehler beim Abrufen makroökonomischer Daten: {e}")
        return None

# Finanzdaten abrufen
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        print(f"Keine Daten für {symbol}")
        return None
    data.reset_index(inplace=True)
    return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Bereinigung der Daten
def clean_data(df):
    df = df.dropna().drop_duplicates()
    return df


# Primärindikatoren hinzufügen, aber Spalten nicht entfernen
def add_primary_indicators(df):
    close = df['Close']

    # Füge nur die notwendigen Indikatoren hinzu
    df['rsi'] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['cci'] = talib.CCI(df['High'], df['Low'], close)
    df['roc'] = ROCIndicator(close=close, window=12).roc()
    df['mom'] = close.diff(1)  # Momentum
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=close, window=14)
    df['stoch'] = stoch.stoch()
    df['stoch_signal'] = stoch.stoch_signal()

    return df


# Erweiterte Indikatoren hinzufügen, ohne primary-Indikatoren zu duplizieren
def add_advanced_indicators(df):
    # Überprüfen, ob die notwendigen Spalten vorhanden sind
    required_columns = ['Close', 'High', 'Low', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Die Spalte '{col}' ist im DataFrame nicht vorhanden.")

    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

    # Volatilitätsindikatoren
    df['atr'] = AverageTrueRange(high=high, low=low, close=close).average_true_range()
    bollinger = BollingerBands(close=close)
    df['bb_mavg'] = bollinger.bollinger_mavg()
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()

    # Trendindikatoren
    df['sma'] = SMAIndicator(close=close, window=20).sma_indicator()
    df['ema'] = EMAIndicator(close=close, window=20).ema_indicator()
    df['adx'] = ADXIndicator(high=high, low=low, close=close).adx()

    # Volumenindikatoren
    df['obv'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df['mfi'] = MFIIndicator(high=high, low=low, close=close, volume=volume).money_flow_index()

    # Zyklusindikatoren
    df['ht_dcperiod'] = talib.HT_DCPERIOD(close)
    df['ht_dphase'] = talib.HT_DCPHASE(close)
    df['ht_sine'], df['ht_lead_sine'] = talib.HT_SINE(close)
    df['ht_trendmode'] = talib.HT_TRENDMODE(close)

    # Preistransformationen
    df['avg_price'] = (high + low + close) / 3
    df['wcl_price'] = (high + low + close * 2) / 4

    # Statistische Indikatoren
    df['correlation'] = close.rolling(window=20).corr(volume)
    df['stddev'] = close.rolling(window=20).std()

    return df


# Daten skalieren und in den spezifischen Ordnern speichern
def scale_and_save_data(df, directory, scaler_filename):
    # Numeric columns auswählen und NaN- und inf-Werte entfernen
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Skalieren und Scaler speichern
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)

    # Speichern in Teile
    chunk_size = 150
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    for idx, chunk in enumerate(chunks):
        chunk.to_csv(os.path.join(directory, f"part_{idx + 1}.csv"))


if __name__ == "__main__":
    symbols = {"BTC-USD": "btc", "ETH-USD": "eth", "SOL-USD": "sol"}
    start_date = "2010-01-01"
    end_date = "2022-01-01"
    data_directory = "../data"
    scaler_directory = "../data/scaler"

    os.makedirs(scaler_directory, exist_ok=True)
    macro_data = fetch_macro_data(start_date, end_date)

    for symbol, abbr in symbols.items():
        base_dir = os.path.join(data_directory, abbr)
        dirs = {
            "no_indicators": os.path.join(base_dir, "no_indicators"),
            "primary_indicators": os.path.join(base_dir, "primary_indicators"),
            "advanced_indicators": os.path.join(base_dir, "advanced_indicators")
        }
        for d in dirs.values():
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        # Basisdaten laden und bereinigen
        raw_data = fetch_data(symbol, start_date, end_date)
        if raw_data is not None:
            cleaned_data = clean_data(raw_data)

            # Basisdaten ohne Indikatoren skalieren und speichern
            scale_and_save_data(cleaned_data, dirs["no_indicators"], f"{scaler_directory}/{abbr}_no_indicators.pkl")

            # Primärindikatoren hinzufügen und unerwünschte Spalten entfernen
            data_primary = cleaned_data.set_index("Date").join(macro_data, how="left").reset_index()
            data_primary = add_primary_indicators(data_primary)
            data_primary = data_primary.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            scale_and_save_data(data_primary, dirs["primary_indicators"], f"{scaler_directory}/{abbr}_primary.pkl")

            # Erweiterte Indikatoren hinzufügen und unerwünschte Spalten entfernen
            data_advanced = add_advanced_indicators(cleaned_data)

            # Entferne nur vorhandene Spalten
            columns_to_drop = [
                'Open', 'High', 'Low', 'Close', 'Volume', 'Fed_Rate', 'Inflation',
                'rsi', 'macd', 'macd_signal', 'cci', 'roc', 'mom', 'stoch', 'stoch_signal'
            ]
            data_advanced = data_advanced.drop(columns=[col for col in columns_to_drop if col in data_advanced.columns])

            scale_and_save_data(data_advanced, dirs["advanced_indicators"], f"{scaler_directory}/{abbr}_advanced.pkl")

        print(f"Daten für {symbol} erfolgreich verarbeitet und in {base_dir} gespeichert.")
