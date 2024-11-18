import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from experiment_7_model_layer import CustomLSTM, MultiInputLSTMWithGates, DualAttention
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# Prüfen, ob CUDA verfügbar ist, und das Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicLSTMModel(nn.Module):
    def __init__(self, seq_length, hidden_dim, input_sizes, num_layers=3, dropout=0.3):
        super(DynamicLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Überprüfen und sicherstellen, dass input_sizes eine Liste aus int-Werten ist
        if not all(isinstance(input_size, int) for input_size in input_sizes):
            raise ValueError("input_sizes muss eine Liste aus ganzzahligen Werten sein.")

        # Dynamische LSTM-Schichten für jede Eingabedimension
        self.lstm_layers = nn.ModuleList([
            CustomLSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout)
            for input_size in input_sizes
        ])

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
        outputs = [lstm_layer(input_data) for lstm_layer, input_data in zip(self.lstm_layers, inputs)]
        processed_signals = [output[0] for output in outputs]

        self_gate = torch.sigmoid(self.dual_attention_layer(processed_signals[0]))
        Index = processed_signals[1] if len(processed_signals) > 1 else processed_signals[0]

        # Multi-Input LSTM zur Fusion
        if len(processed_signals) == 1:
            combined_input, _ = self.multi_input_lstm(processed_signals[0], processed_signals[0], processed_signals[0],
                                                      Index, self_gate)
        else:
            combined_input, _ = self.multi_input_lstm(processed_signals[0], processed_signals[1], processed_signals[2],
                                                      Index, self_gate)

        attended = self.dual_attention_layer(combined_input)
        output = self.fc(attended)
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


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.output_dim = output_dim

        # Dynamische Berechnung der Flattened-Dimension
        self.flattened_dim = None
        with torch.no_grad():
            temp_input = torch.zeros(1, input_dim, seq_length)
            temp_output = self.relu(self.conv2(self.relu(self.conv1(temp_input))))
            self.flattened_dim = temp_output.numel()

        # Fully Connected Layer
        self.fc = nn.Linear(self.flattened_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # Transponiere für Conv1d (Batch, Features, Seq)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        print(f"Shape vor Flattening: {x.shape}")  # Debugging
        x = x.view(x.size(0), -1)  # Dynamisches Flattening
        print(f"Shape vor Fully-Connected Layer: {x.shape}")  # Debugging
        x = self.fc(x)
        return x


class FusionModelWithCNN(nn.Module):
    def __init__(self, hidden_dim=64, seq_length=30, output_features=10, group_features=None):
        super(FusionModelWithCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.output_features = output_features

        # Extrahiere die Eingabedimensionen aus den Gruppen
        self.input_sizes = {
            key: len(value) if isinstance(value, list) else value
            for key, value in group_features.items()
        }

        # Initialisiere die LSTM-Modelle mit den Dimensionen
        self.standard_model = DynamicLSTMModel(
            seq_length, hidden_dim, [self.input_sizes["Standard"]]
        ).to(device)

        self.indicators_1_model = DynamicLSTMModel(
            seq_length, hidden_dim, [self.input_sizes["Indicators_Group_1"]]
        ).to(device)

        self.indicators_2_model = DynamicLSTMModel(
            seq_length, hidden_dim, [self.input_sizes["Indicators_Group_2"]]
        ).to(device)

        # Initialisiere das CNN-Modell für die Fusion
        total_input_dim = sum(self.input_sizes.values())
        self.cnn_model = SimpleCNN(
            input_dim=total_input_dim, seq_length=seq_length, output_dim=hidden_dim
        )

        # Fully Connected Layer zur finalen Fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(output_features * 3 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_features)
        )

    def forward(self, Y, X1, X2):
        # Vorhersagen mit den LSTM-Modellen
        standard_pred = self.standard_model(Y)
        indicators_1_pred = self.indicators_1_model(X1)
        indicators_2_pred = self.indicators_2_model(X2)

        # CNN-Verarbeitung
        combined_input = torch.cat((Y, X1, X2), dim=2)
        cnn_output = self.cnn_model(combined_input)

        # Kombiniere alle Ergebnisse
        cnn_output = cnn_output.unsqueeze(1).repeat(1, combined_input.size(1), 1)
        combined = torch.cat((standard_pred, indicators_1_pred, indicators_2_pred, cnn_output), dim=2)

        # Finale Ausgabe durch Fusion Layer
        output = self.fusion_fc(combined)
        return output


def train_fusion_model_with_cnn(X_standard, X_group1, X_group2, Y_samples, hidden_dim, batch_size, learning_rate, epochs, model_dir):
    print(f"X_standard Shape: {X_standard.shape}")
    print(f"X_group1 Shape: {X_group1.shape}")
    print(f"X_group2 Shape: {X_group2.shape}")
    print(f"Y_samples Shape: {Y_samples.shape}")

    group_features = {
        "Standard": X_standard.shape[-1],  # Einzelne Eingabedimension
        "Indicators_Group_1": X_group1.shape[-1],
        "Indicators_Group_2": X_group2.shape[-1],
    }

    input_sizes = [group_features["Standard"], group_features["Indicators_Group_1"],
                   group_features["Indicators_Group_2"]]

    model = FusionModelWithCNN(
        hidden_dim=hidden_dim,
        seq_length=X_standard.shape[1],  # Die Sequenzlänge aus den Eingabedaten extrahieren
        output_features=Y_samples.size(-1),
        group_features=group_features,
    ).to(device)

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
    print(f"Fusion Model mit CNN gespeichert unter {model_path}")
    return model_path


# Hauptfunktion
if __name__ == '__main__':

    # Setzen der Zufallszahlenseeds für Python, NumPy und PyTorch
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # Falls Sie CUDA verwenden, setzen Sie auch den Seed für CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)  # Für Multi-GPU, falls verwendet

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
    train_fusion_model_with_cnn(
        X_standard,
        X_group1,
        X_group2,
        Y_samples,
        hidden_dim,
        batch_size,
        learning_rate,
        epochs,
        model_dir
    )