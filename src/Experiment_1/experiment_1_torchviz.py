import os
import shutil
import torch
import pickle
from torchviz import make_dot
from src.Experiment_1.experiment_1_model import FusionModel

# Prüfe, ob CUDA verfügbar ist, und setze das Gerät
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modellparameter
hidden_dim = 64
seq_length = 50

# Speicherort des Modells und der Daten
model_path = "./Data/Models/fusion_model_final.pth"
samples_dir = "./Data/Samples/"
output_png = "./fusion_model_graph"  # Basisname für die PNG-Datei

# Funktion zum Bereinigen des Modells
def clean_model(model_path):
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Datei gelöscht: {model_path}")

# Hauptfunktion
def main():
    # Bereinige altes Modell
    clean_model(model_path)

    # Lade die echten Daten
    with open(os.path.join(samples_dir, 'Standard_X.pkl'), 'rb') as f:
        X_standard = pickle.load(f)
    with open(os.path.join(samples_dir, 'Indicators_Group_1_X.pkl'), 'rb') as f:
        X_group1 = pickle.load(f)
    with open(os.path.join(samples_dir, 'Indicators_Group_2_X.pkl'), 'rb') as f:
        X_group2 = pickle.load(f)

    # Wandle die Daten in Tensoren um und lade sie auf das Gerät
    Y = torch.tensor(X_standard[:1]).clone().detach().float().to(device)  # Nur ein Batch
    X1 = torch.tensor(X_group1[:1]).clone().detach().float().to(device)
    X2 = torch.tensor(X_group2[:1]).clone().detach().float().to(device)

    # Initialisiere Modell
    model = FusionModel(hidden_dim=hidden_dim, seq_length=seq_length).to(device)
    print("Erstelle neues Modell...")
    # Speichere Modell
    torch.save(model.state_dict(), model_path)
    print(f"Modell gespeichert: {model_path}")

    # Visualisiere das Modell mit torchviz
    output = model(Y, X1, X2)  # Führe eine Vorhersage aus, um den Graphen zu erstellen
    dot = make_dot(output, params=dict(model.named_parameters()))  # Erstelle den Graphen
    dot.format = 'png'  # Setze das Ausgabeformat auf PNG
    dot.attr(dpi="300")  # Erhöhe die Auflösung
    dot.render(output_png, format="png", cleanup=True)  # Speichere die PNG-Datei
    print(f"Modellgraph als PNG gespeichert: {output_png}.png")

# Ausführung
if __name__ == "__main__":
    main()
