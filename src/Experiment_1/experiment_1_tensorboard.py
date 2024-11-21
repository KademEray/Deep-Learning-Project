import os
import shutil
import torch
import pickle
from subprocess import Popen
from torch.utils.tensorboard import SummaryWriter
from src.Experiment_1.experiment_1_model import FusionModel

# Prüfe, ob CUDA verfügbar ist, und setze das Gerät
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modellparameter
hidden_dim = 64
seq_length = 50

# Speicherort des Modells und der Daten
model_path = "./Data/Models/fusion_model_final.pth"
samples_dir = "./Data/Samples/"
log_dir = "./logs"  # Verzeichnis für TensorBoard-Dateien

# Funktion zum Bereinigen des Log-Verzeichnisses
def clean_logs(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Altes Log-Verzeichnis '{directory}' gelöscht.")

# TensorBoard starten
def start_tensorboard(logdir):
    print("Starte TensorBoard...")
    return Popen(["tensorboard", "--logdir", logdir])

# Hauptfunktion
def main():
    # Bereinige Logs
    clean_logs(log_dir)

    # Lade die echten Daten
    with open(os.path.join(samples_dir, 'Standard_X.pkl'), 'rb') as f:
        X_standard = pickle.load(f)
    with open(os.path.join(samples_dir, 'Indicators_Group_1_X.pkl'), 'rb') as f:
        X_group1 = pickle.load(f)
    with open(os.path.join(samples_dir, 'Indicators_Group_2_X.pkl'), 'rb') as f:
        X_group2 = pickle.load(f)

    # Wandle die Daten in Tensoren um und lade sie auf das Gerät
    Y = torch.tensor(X_standard[:1], dtype=torch.float32).to(device)  # Nur ein Batch
    X1 = torch.tensor(X_group1[:1], dtype=torch.float32).to(device)
    X2 = torch.tensor(X_group2[:1], dtype=torch.float32).to(device)

    # Initialisiere Modell
    model = FusionModel(hidden_dim=hidden_dim, seq_length=seq_length).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print("Modell-Datei nicht gefunden.")
        return

    # Initialisiere TensorBoard Writer
    writer = SummaryWriter(log_dir)

    # Übergebe den Modellgraphen an TensorBoard mit den echten Daten
    writer.add_graph(model, (Y, X1, X2))
    print(f"Graph gespeichert in: {log_dir}")

    # Schließe den Writer
    writer.close()

    # Starte TensorBoard
    tensorboard_process = start_tensorboard(log_dir)
    try:
        print("TensorBoard läuft. Drücke Strg+C, um zu beenden.")
        tensorboard_process.wait()
    except KeyboardInterrupt:
        print("Beende TensorBoard...")
        tensorboard_process.terminate()

# Ausführung
if __name__ == "__main__":
    main()
