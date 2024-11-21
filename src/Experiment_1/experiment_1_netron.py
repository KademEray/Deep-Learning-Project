import os
import shutil
import torch
import pickle
from subprocess import Popen
from src.Experiment_1.experiment_1_model import FusionModel

# Prüfe, ob CUDA verfügbar ist, und setze das Gerät
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modellparameter
hidden_dim = 64
seq_length = 50

# Speicherort des Modells und der Daten
model_path = "./Data/Models/fusion_model_final.pth"
samples_dir = "./Data/Samples/"
onnx_path = "./fusion_model.onnx"  # Speicherort für die ONNX-Datei

# Funktion zum Löschen eines Modells oder ONNX-Dateien
def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Datei gelöscht: {path}")

# Funktion zum Löschen des Modells und ONNX-Datei
def clean_model_and_onnx(model_path, onnx_path):
    delete_if_exists(model_path)
    delete_if_exists(onnx_path)

# Netron starten
def start_netron(onnx_path):
    print("Starte Netron...")
    return Popen(["netron", onnx_path])

# Funktion zur Erstellung und Speicherung des Modells
def create_and_save_model(model_path, hidden_dim, seq_length):
    print("Erstelle neues Modell...")
    model = FusionModel(hidden_dim=hidden_dim, seq_length=seq_length).to(device)
    torch.save(model.state_dict(), model_path)
    print(f"Modell gespeichert: {model_path}")
    return model

# Hauptfunktion
def main():
    # Lösche bestehende Dateien
    clean_model_and_onnx(model_path, onnx_path)

    # Lade die Daten
    with open(os.path.join(samples_dir, 'Standard_X.pkl'), 'rb') as f:
        X_standard = pickle.load(f)
    with open(os.path.join(samples_dir, 'Indicators_Group_1_X.pkl'), 'rb') as f:
        X_group1 = pickle.load(f)
    with open(os.path.join(samples_dir, 'Indicators_Group_2_X.pkl'), 'rb') as f:
        X_group2 = pickle.load(f)

    # Wandle die Daten in Tensoren um und lade sie auf das Gerät
    Y = torch.tensor(X_standard[:1]).float().to(device)  # Nur ein Batch
    X1 = torch.tensor(X_group1[:1]).float().to(device)
    X2 = torch.tensor(X_group2[:1]).float().to(device)

    # Prüfe, ob Modell existiert, oder erstelle es
    if os.path.exists(model_path):
        print(f"Lade existierendes Modell: {model_path}")
        model = FusionModel(hidden_dim=hidden_dim, seq_length=seq_length).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = create_and_save_model(model_path, hidden_dim, seq_length)

    model.eval()

    # Exportiere das Modell in ONNX-Format
    torch.onnx.export(
        model,
        (Y, X1, X2),  # Beispiel-Eingaben
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["Y", "X1", "X2"],
        output_names=["output"],
        dynamic_axes={
            "Y": {0: "batch_size", 1: "seq_length"},
            "X1": {0: "batch_size", 1: "seq_length"},
            "X2": {0: "batch_size", 1: "seq_length"},
            "output": {0: "batch_size", 1: "seq_length"},
        },
    )
    print(f"Modell in ONNX-Format exportiert: {onnx_path}")

    # Starte Netron
    netron_process = start_netron(onnx_path)
    try:
        print("Netron läuft. Drücke Strg+C, um zu beenden.")
        netron_process.wait()
    except KeyboardInterrupt:
        print("Beende Netron...")
        netron_process.terminate()

# Ausführung
if __name__ == "__main__":
    main()
