# Code zum Sammeln von Daten mithilfe der Kaggle API und zum Säubern der Daten
import kaggle
import pandas as pd
import os
import delete_data
import ta  # Technische Analyse Bibliothek
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import time

# Setze den Pfad für kaggle.json, falls dieser nicht im Standardpfad liegt
kaggle_json_path = "../kaggle.json"
os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath(os.path.dirname(kaggle_json_path))

# Initialisieren der Kaggle API mit der kaggle.json Datei
api = KaggleApi()
api.authenticate()

def fetch_data(dataset_name, raw_data_directory, retries=3):
    """
    Holt historische Daten von Kaggle und speichert sie als CSV-Dateien.
    :param dataset_name: Der Name des Kaggle-Datensatzes (z.B. 'jacksoncrow/stock-market-dataset')
    :param raw_data_directory: Verzeichnis zum Speichern der rohen Daten
    :param retries: Anzahl der Wiederholungsversuche bei Fehlern
    """
    for attempt in range(retries):
        try:
            api.dataset_download_files(dataset_name, path=raw_data_directory, unzip=True)
            csv_files = []
            for root, _, files in os.walk(raw_data_directory):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            if csv_files:
                dataframes = [pd.read_csv(file) for file in tqdm(csv_files, desc="Lade CSV-Dateien")]  # Ladeleiste für das Lesen der CSV-Dateien
                combined_data = pd.concat(dataframes, ignore_index=True)
                # Filtere Daten ab 2020, um die Datenmenge zu reduzieren
                if 'Date' in combined_data.columns:
                    combined_data['Date'] = pd.to_datetime(combined_data['Date'], errors='coerce')
                    combined_data = combined_data[combined_data['Date'] >= '2010-01-01']
                return combined_data
            else:
                print(f"Keine CSV-Dateien in {raw_data_directory} gefunden.")
                return None
        except Exception as e:
            print(f"Fehler beim Herunterladen von {dataset_name}: {e}. Versuch {attempt + 1} von {retries}.")
            time.sleep(5)  # Wartezeit zwischen den Versuchen
    print(f"Download von {dataset_name} nach {retries} Versuchen fehlgeschlagen.")
    return None

def clean_data(df):
    """
    Säubert die erhaltenen Daten.
    :param df: DataFrame mit den rohen Daten
    :return: Gesäuberter DataFrame
    """
    # Entferne Zeilen, bei denen Basiswerte fehlen
    basis_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    df = df.dropna(subset=basis_columns)
    # Entferne Duplikate
    df = df.drop_duplicates()
    return df

def add_technical_indicators(df):
    """
    Fügt technische Indikatoren zum DataFrame hinzu.
    :param df: DataFrame mit den rohen oder gesäuberten Daten
    :return: DataFrame mit technischen Indikatoren
    """
    try:
        close = df['Close']

        if len(close) < 50:
            print(f"Nicht genügend Daten für technische Indikatoren vorhanden.")
            return df

        # Erstelle einen temporären DataFrame für die neuen Indikatoren
        indicators = pd.DataFrame(index=close.index)

        # Hinzufügen des RSI (Relative Strength Index)
        indicators['rsi'] = RSIIndicator(close=close, window=14).rsi()

        # Hinzufügen des MACD (Moving Average Convergence Divergence)
        macd = MACD(close=close)
        indicators['macd'] = macd.macd()
        indicators['macd_signal'] = macd.macd_signal()

        # Hinzufügen von Bollinger Bändern
        bollinger = ta.volatility.BollingerBands(close=close)
        df['bb_mavg'] = bollinger.bollinger_mavg()
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()

        # Korrigiertes Hinzufügen des Stochastic Oscillator
        stochastic = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=close, window=14)
        df['stoch'] = stochastic.stoch()
        df['stoch_signal'] = stochastic.stoch_signal()

        # Entferne Indikatoren, die komplett leer sind
        indicators = indicators.dropna(axis=1, how='all')

        # Füge die Indikatoren zum ursprünglichen DataFrame hinzu (mit pd.concat, um Fragmentierung zu vermeiden)
        df = pd.concat([df, indicators], axis=1)

    except Exception as e:
        print(f"Fehler beim Hinzufügen der technischen Indikatoren: {e}")

    return df


def scale_data(df):
    """
    Skaliert die numerischen Daten des DataFrames.
    :param df: DataFrame mit den gesäuberten Daten
    :return: Skalierter DataFrame
    """
    if df.empty:
        print("Keine Daten zum Skalieren vorhanden.")
        return df

    # Nur Spalten auswählen, die numerische Werte enthalten und keine reinen NaN-Werte sind
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if df[col].notna().any()]

    if not numeric_cols:
        print("Keine numerischen Spalten zum Skalieren vorhanden.")
        return df

    # NaN-Werte in den ausgewählten numerischen Spalten durch den Mittelwert ersetzen
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Skaliere die ausgewählten numerischen Spalten
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


if __name__ == "__main__":
    # Beispiel für das Sammeln von Daten und deren Säuberung
    dataset_names = ["jacksoncrow/stock-market-dataset", "jessevent/all-crypto-currencies"]  # Beispiel für zwei Kaggle-Datensätze
    start_date = "2018-01-01"
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Löschen des Speicherorts für alte Dateien
    raw_data_directory = "../data/raw_data"
    processed_data_directory = "../data/processed_data"

    delete_data.delete_old_data(raw_data_directory, processed_data_directory)

    # Erstellen neuer Verzeichnisse
    os.makedirs(raw_data_directory, exist_ok=True)
    os.makedirs(processed_data_directory, exist_ok=True)

    # Daten von beiden Datensätzen herunterladen und kombinieren
    combined_data = pd.DataFrame()
    for dataset_name in dataset_names:
        raw_data = fetch_data(dataset_name, raw_data_directory)
        if raw_data is not None:
            combined_data = pd.concat([combined_data, raw_data], ignore_index=True)

    if not combined_data.empty:
        # Säubern der kombinierten Daten
        cleaned_data = clean_data(combined_data)

        # Hinzufügen von technischen Indikatoren
        cleaned_data = add_technical_indicators(cleaned_data)

        # Skalieren der Daten
        scaled_data = scale_data(cleaned_data)

        # Aufteilen der Daten in kleinere Teile von jeweils 150 Zeilen und speichern
        chunk_size = 150
        chunks = [scaled_data.iloc[i:i + chunk_size] for i in range(0, len(scaled_data), chunk_size)]
        for idx, chunk in enumerate(chunks):
            chunk_file = os.path.join(processed_data_directory, f"cleaned_data_part_{idx + 1}.csv")
            chunk.to_csv(chunk_file)

        print("Daten erfolgreich gesammelt, gesäubert, mit technischen Indikatoren erweitert, skaliert und in Teilen gespeichert.")