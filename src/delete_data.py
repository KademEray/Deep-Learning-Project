import os
import shutil
from datetime import datetime, timedelta

def delete_old_data(raw_data_directory, processed_data_directory):
    """
    Löscht alte Dateien in den angegebenen Verzeichnissen.
    :param raw_data_directory: Verzeichnis, das die rohen Daten enthält
    :param processed_data_directory: Verzeichnis, das die gesäuberten Daten enthält
    """
    # Löschen der alten rohen Daten
    if os.path.exists(raw_data_directory):
        for filename in os.listdir(raw_data_directory):
            file_path = os.path.join(raw_data_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Fehler beim Löschen von {file_path}. Grund: {e}')
        print(f"Alter Speicherort '{raw_data_directory}' wurde gelöscht.")

    # Löschen der alten gesäuberten Daten
    if os.path.exists(processed_data_directory):
        for filename in os.listdir(processed_data_directory):
            file_path = os.path.join(processed_data_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Fehler beim Löschen von {file_path}. Grund: {e}')
        print(f"Alter Speicherort '{processed_data_directory}' wurde gelöscht.")

if __name__ == "__main__":
    # Verzeichnisse für rohe und gesäuberte Daten
    raw_data_directory = "../data/raw_data"
    processed_data_directory = "../data/processed_data"

    # Alte Daten löschen
    delete_old_data(raw_data_directory, processed_data_directory)
