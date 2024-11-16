import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from experiment_4_model_layer import CustomLSTM, MultiInputLSTMWithGates, DualAttention
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# Prüfen, ob CUDA verfügbar ist, und das Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicLSTMModel(nn.Module):
    def __init__(self, seq_length, hidden_dim, input_sizes, num_layers=3, dropout=0.3):
        super(DynamicLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Erstelle dynamische LSTM-Schichten für jede Eingabedimension
        self.lstm_layers = nn.ModuleList([CustomLSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout)
                                          for input_size in input_sizes])

        # Multi-Input LSTM und Attention
        self.multi_input_lstm = MultiInputLSTMWithGates(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.dual_attention_layer = DualAttention(hidden_dim, hidden_dim)

        # Fully Connected Layer für jeden Zeitschritt
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # Endausgabe für 10 Features
        )

    def forward(self, *inputs):
        # Sicherstellen, dass die Eingabegrößen die LSTM-Schichten korrekt füttern
        outputs = [lstm_layer(input_data) for lstm_layer, input_data in zip(self.lstm_layers, inputs)]

        # Extrahiere das gelernte Signal aus jedem LSTM und füge es zusammen
        processed_signals = [output[0] for output in outputs]

        # Berechne self_gate und übergebe es zusammen mit den anderen Eingaben an MultiInputLSTMWithGates
        self_gate = torch.sigmoid(self.dual_attention_layer(processed_signals[0]))  # Beispielhaft aus den verarbeiteten Signalen
        Index = processed_signals[1]  # Beispielweise ein weiteres Signal

        # Multi-Input LSTM zur Fusion
        combined_input, _ = self.multi_input_lstm(*processed_signals, Index, self_gate)

        # Attention und finale Transformation für jeden Zeitschritt
        attended = self.dual_attention_layer(combined_input)

        # Vollständige Sequenz als Ausgabe für jeden Zeitschritt durchlaufen
        output = self.fc(attended)  # Ausgabe mit Form [Batch, Seq, Features]

        return output


def train_group_model(group_name, X_samples, Y_samples, seq_length, hidden_dim, batch_size, learning_rate, epochs, model_dir):
    """
    Trainiert ein DynamicLSTMModel für eine spezifische Gruppe von Daten (z.B., Standard, Indicators_Group_1).
    Das Modell wird für eine Anzahl von Epochen mit dem Mean Squared Error (MSE) als Verlustfunktion und Adam als Optimierer trainiert.
    """
    # Bestimme die Eingabedimensionen für jede Gruppe
    input_sizes = [X.shape[2] for X in [X_samples]]  # Hier werden alle Gruppen mit ihren Dimensionen berücksichtigt

    # Initialisiere das Modell für die spezifische Gruppe und bewege es auf das Gerät (GPU, wenn verfügbar)
    model = DynamicLSTMModel(seq_length, hidden_dim, input_sizes).to(device)

    # Konvertiere X_samples und Y_samples in Tensors, falls sie es nicht sind, und verschiebe sie auf das richtige Gerät
    if not isinstance(X_samples, torch.Tensor):
        X_samples = torch.tensor(X_samples, dtype=torch.float32)
    if not isinstance(Y_samples, torch.Tensor):
        Y_samples = torch.tensor(Y_samples, dtype=torch.float32)

    # Übertrage die Daten auf das Gerät
    X_samples = X_samples.to(device)
    Y_samples = Y_samples.to(device)

    # Definiere die Verlustfunktion und den Optimierer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Erstelle einen Dataset- und Dataloader für die Eingabe- und Ziel-Daten
    dataset = torch.utils.data.TensorDataset(X_samples, Y_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setze das Modell in den Trainingsmodus
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)  # Übertrage die Daten auf das Gerät

            optimizer.zero_grad()  # Setze die Gradienten des Optimierers auf Null

            # Vorhersage durchführen
            output = model(X, X, X)  # Passen Sie die Eingaben je nach Gruppe an

            # Berechne den Verlust und führe Backpropagation durch
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Durchschnittlicher Verlust pro Epoche
        epoch_loss = running_loss / len(dataloader)
        print(f"Gruppe {group_name} - Epoch {epoch + 1}/{epochs}, Durchschnittlicher Verlust: {epoch_loss:.6f}")

    # Speichere das trainierte Modell
    model_path = os.path.join(model_dir, f"{group_name}_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modell für Gruppe {group_name} gespeichert unter {model_path}")

    return model_path


class FusionModel(nn.Module):
    def __init__(self, hidden_dim=64, seq_length=30, output_features=10, group_features=None):
        super(FusionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.output_features = output_features

        # Bestimme die Eingabedimensionen für jede Gruppe
        input_sizes = [len(group_features[group_name]) for group_name in group_features]

        # Initialisiert drei DynamicLSTMModel-Instanzen für die jeweiligen Gruppen
        self.standard_model = DynamicLSTMModel(seq_length, hidden_dim, input_sizes).to(device)
        self.indicators_1_model = DynamicLSTMModel(seq_length, hidden_dim, input_sizes).to(device)
        self.indicators_2_model = DynamicLSTMModel(seq_length, hidden_dim, input_sizes).to(device)

        # Definiert eine Fully Connected-Schicht für die finale Fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(output_features * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_features)
        )

    def forward(self, Y, X1, X2):
        # Generiert die Vorhersagen der einzelnen Gruppenmodelle
        standard_pred = self.standard_model(Y, X1, X2)  # Erwartete Form: [Batch, Seq, Features]
        indicators_1_pred = self.indicators_1_model(Y, X1, X2)  # Erwartete Form: [Batch, Seq, Features]
        indicators_2_pred = self.indicators_2_model(Y, X1, X2)  # Erwartete Form: [Batch, Seq, Features]

        # Sicherstellen, dass die Ausgaben die Sequenz-Dimension enthalten
        if standard_pred.dim() == 2:
            standard_pred = standard_pred.unsqueeze(1)
        if indicators_1_pred.dim() == 2:
            indicators_1_pred = indicators_1_pred.unsqueeze(1)
        if indicators_2_pred.dim() == 2:
            indicators_2_pred = indicators_2_pred.unsqueeze(1)

        # Kombiniere die Features entlang der letzten Dimension
        combined = torch.cat((standard_pred, indicators_1_pred, indicators_2_pred), dim=2)  # [Batch, Seq, Features*3]

        # Wende die Fusion-Schicht für jeden Zeitschritt der Sequenz an
        output = self.fusion_fc(combined)  # Erwartet [Batch, Seq, Features]

        return output  # Gibt eine Ausgabe in der Form [Batch, Seq, Features]


def train_fusion_model(X_standard, X_group1, X_group2, Y_samples, hidden_dim, batch_size, learning_rate, epochs, model_dir):
    input_sizes = [X_standard.shape[2], X_group1.shape[2], X_group2.shape[2]]  # Dynamische Bestimmung der Eingabedimensionen

    model = DynamicLSTMModel(seq_length=50, hidden_dim=hidden_dim, input_sizes=input_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(X_standard, X_group1, X_group2, Y_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (X_standard, X_group1, X_group2, y) in enumerate(dataloader):
            X_standard, X_group1, X_group2, y = (
                X_standard.float().to(device),
                X_group1.float().to(device),
                X_group2.float().to(device),
                y.float().to(device),
            )

            optimizer.zero_grad()

            output = model(X_standard, X_group1, X_group2)

            # Verwende nur die letzten 30 Zeitschritte in der Loss-Berechnung
            output = output[:, -30:, :]
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Durchschnittlicher Verlust: {epoch_loss:.6f}")

    model_path = os.path.join(model_dir, "fusion_model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Fusion Model gespeichert unter {model_path}")
    return model_path


# Hauptfunktion
if __name__ == '__main__':

    # Setzen der Zufallszahlenseeds für Python, NumPy und PyTorch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Falls Sie CUDA verwenden, setzen Sie auch den Seed für CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # Für Multi-GPU, falls verwendet

    # Zusätzliche Einstellungen, um deterministisches Verhalten sicherzustellen
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Definiert die Verzeichnisse für gespeicherte und trainierte Daten
    samples_dir = "./Data/Samples"  # Verzeichnis mit den vorbereiteten Trainingssequenzen
    model_dir = "./Data/Models"  # Verzeichnis zum Speichern des trainierten Modells

    # Wichtige Modell- und Trainingsparameter
    seq_length = 50  # Länge der Eingabesequenzen in Tagen
    hidden_dim = 64  # Dimension der versteckten Schicht im Modell
    batch_size = 256  # Anzahl der Samples pro Batch
    learning_rate = 0.001  # Lernrate für den Optimizer
    epochs = 50  # Anzahl der Trainingsdurchläufe (Epochen)

    with open(os.path.join(samples_dir, 'Standard_X.pkl'), 'rb') as f:
        X_standard = pickle.load(f)  # Standard-Feature-Gruppe
    with open(os.path.join(samples_dir, 'Indicators_Group_1_X.pkl'), 'rb') as f:
        X_group1 = pickle.load(f)  # Momentum-Indikatorgruppe
    with open(os.path.join(samples_dir, 'Indicators_Group_2_X.pkl'), 'rb') as f:
        X_group2 = pickle.load(f)  # Volatilität und Trend-Indikatorgruppe
    with open(os.path.join(samples_dir, 'Standard_Y.pkl'), 'rb') as f:
        Y_samples = pickle.load(f)  # Zielwerte für die Vorhersage

    print(
        f"Geladene Y_samples Form: {Y_samples.shape}")  # Ausgabe der Form, um zu bestätigen, dass die Daten korrekt geladen wurden (sollte [Anzahl Samples, 1] sein)

    # Starte das Training des Fusion Models mit den geladenen Daten und Parametern
    train_fusion_model(
        X_standard,  # Eingabesequenzen für die Standard-Feature-Gruppe
        X_group1,  # Eingabesequenzen für die Momentum-Indikatorgruppe
        X_group2,  # Eingabesequenzen für die Volatilität/Trend-Indikatorgruppe
        Y_samples,  # Zielwerte für die Vorhersage
        hidden_dim,  # Dimension der versteckten Schicht des Modells
        batch_size,  # Größe der Batches für das Training
        learning_rate,  # Lernrate für den Optimizer
        epochs,  # Anzahl der Epochen für das Training
        model_dir  # Verzeichnis zum Speichern des trainierten Modells
    )