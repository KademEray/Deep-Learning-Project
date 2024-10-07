# Code zum Sammeln von Daten mithilfe von yfinance und zum Säubern der Daten
import yfinance as yf
import pandas as pd
import os
import delete_data
import ta  # Technische Analyse Bibliothek
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from datetime import datetime, timedelta


def fetch_data(tickers, start_date, end_date):
    """
    Holt historische Daten von Yahoo Finance für mehrere Ticker und gibt sie als DataFrame zurück.
    :param tickers: Liste der Symbole der Aktien oder Kryptowährungen (z.B. ['AAPL', 'MSFT', 'BTC-USD'])
    :param start_date: Startdatum der Daten (im Format 'YYYY-MM-DD')
    :param end_date: Enddatum der Daten (im Format 'YYYY-MM-DD')
    :return: DataFrame mit den erhaltenen Daten für alle Ticker
    """
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    if not data.empty:
        return data
    else:
        print(f"Keine Daten für die angegebenen Ticker im Zeitraum von {start_date} bis {end_date} gefunden.")
        return None


def clean_data(df):
    """
    Säubert die erhaltenen Daten.
    :param df: DataFrame mit den rohen Daten
    :return: Gesäuberter DataFrame
    """
    # Fülle alle fehlenden Werte mit NaN, um sicherzustellen, dass keine Zeilen gelöscht werden
    df = df.fillna(value=pd.NA)
    # Entferne Duplikate
    df = df.drop_duplicates()
    return df


def add_technical_indicators(df, ticker):
    """
    Fügt technische Indikatoren zum DataFrame hinzu.
    :param df: DataFrame mit den rohen oder gesäuberten Daten
    :param ticker: das Symbol der Aktie oder Kryptowährung
    :return: DataFrame mit technischen Indikatoren
    """
    try:
        close = df[(ticker, 'Close')]
        high = df[(ticker, 'High')]
        low = df[(ticker, 'Low')]
        volume = df[(ticker, 'Volume')]

        if len(close) < 50:
            print(f"Nicht genügend Daten für technische Indikatoren für {ticker} vorhanden.")
            return df

        # Erstelle einen temporären DataFrame für die neuen Indikatoren
        indicators = pd.DataFrame(index=close.index)

        # Hinzufügen des RSI (Relative Strength Index)
        indicators['rsi'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # Hinzufügen des MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close=close)
        indicators['macd'] = macd.macd()
        indicators['macd_signal'] = macd.macd_signal()
        indicators['macd_diff'] = macd.macd_diff()

        # Hinzufügen von Bollinger Bändern
        bollinger = ta.volatility.BollingerBands(close=close, window=20)
        indicators['bollinger_mavg'] = bollinger.bollinger_mavg()
        indicators['bollinger_hband'] = bollinger.bollinger_hband()
        indicators['bollinger_lband'] = bollinger.bollinger_lband()

        # Hinzufügen von gleitenden Durchschnitten
        indicators['sma_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        indicators['sma_200'] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
        indicators['ema_50'] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
        indicators['ema_200'] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

        # Hinzufügen des Average True Range (ATR)
        indicators['atr'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close,
                                                           window=14).average_true_range()

        # Hinzufügen des Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14)
        indicators['stoch_k'] = stoch.stoch()
        indicators['stoch_d'] = stoch.stoch_signal()

        # Hinzufügen des On-Balance Volume (OBV)
        indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

        # Hinzufügen des Money Flow Index (MFI)
        indicators['mfi'] = ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume,
                                                   window=14).money_flow_index()

        # Füge die Indikatoren zum ursprünglichen DataFrame hinzu (mit pd.concat, um Fragmentierung zu vermeiden)
        df = pd.concat([df, indicators], axis=1)

    except Exception as e:
        print(f"Fehler beim Hinzufügen der technischen Indikatoren für {ticker}: {e}")

    return df


if __name__ == "__main__":
    # Beispiel für das Sammeln von Daten und deren Säuberung
    tickers = [
        "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX", "NVDA", "JPM", "V", "DIS", "INTC", "CSCO", "WMT", "PFE",
        "XOM", "CVX", "BA", "MMM", "NKE", "PEP", "KO", "T", "VZ", "BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD",
        "DOT-USD", "BABA", "ORCL", "SAP", "TM", "NSRGY", "RY", "BNS", "HSBC", "SNY", "GSK", "LMT", "RTX", "MRNA", "ABT", "UNH",
        "JNJ", "PG", "HD", "GS", "MS", "C", "BAC", "WFC", "ADBE", "PYPL", "CRM", "ZM", "SNAP", "SQ", "SPOT", "SHOP", "ZM",
        "BIDU", "NTES", "JD", "TME", "BHP", "RIO", "VALE", "GLNCY", "BP", "SHEL", "ENB", "SU", "EQNR", "MCD",
        "SBUX", "YUM", "CMG", "DPZ", "PINS", "TWLO", "F", "GM", "HMC", "RACE", "NSC", "UNP", "CSX", "PGR", "ALL", "TRV"
    ]  # Erweiterte Liste der gewünschten Aktien oder Kryptowährungen
    start_date = "2014-01-01"
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Löschen des Speicherorts für alte Dateien
    raw_data_directory = "../data/raw_data"
    processed_data_directory = "../data/processed_data"

    delete_data.delete_old_data(raw_data_directory, processed_data_directory)

    # Erstellen neuer Verzeichnisse
    os.makedirs(raw_data_directory, exist_ok=True)
    os.makedirs(processed_data_directory, exist_ok=True)

    raw_data = fetch_data(tickers, start_date, end_date)
    if raw_data is not None:
        # Aufteilen der Daten in kleinere Teile von jeweils 150 Zeilen und speichern
        chunk_size = 150
        chunks = [raw_data.iloc[i:i + chunk_size] for i in range(0, len(raw_data), chunk_size)]
        for idx, chunk in enumerate(chunks):
            chunk_file = os.path.join(raw_data_directory, f"raw_data_part_{idx + 1}.csv")
            chunk.to_csv(chunk_file)

        print("Rohdaten erfolgreich in Teile aufgeteilt und gespeichert.")

        # Säubern und Speichern der einzelnen Teile
        for idx in range(len(chunks)):
            chunk_file = os.path.join(raw_data_directory, f"raw_data_part_{idx + 1}.csv")
            chunk_data = pd.read_csv(chunk_file, header=[0, 1], index_col=0)
            cleaned_chunk = clean_data(chunk_data)

            # Speichern der gesäuberten Daten
            cleaned_chunk.to_csv(os.path.join(processed_data_directory, f"cleaned_data_part_{idx + 1}.csv"))

        print("Daten erfolgreich gesammelt, gesäubert und in Teilen gespeichert.")