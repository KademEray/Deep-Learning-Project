import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from standard_model_layer import CustomLSTM, MultiInputLSTMWithGates, DualAttention

# Prüfen, ob CUDA verfügbar ist, und das Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ThreeGroupLSTMModel(nn.Module):
    def __init__(self, seq_length, hidden_dim, num_layers=3, dropout=0.3):
        """
        Initialisiert das ThreeGroupLSTMModel, das darauf ausgelegt ist, drei separate Datenquellen (Y, X1, X2) zu verarbeiten,
        ihre relevanten Merkmale zu extrahieren und mithilfe eines Gated LSTM und Attention-Mechanismen zu fusionieren.

        Parameter:
        - seq_length (int): Länge der Sequenz für die Eingaben X1 und X2.
        - hidden_dim (int): Dimension der versteckten Zustände der LSTM-Schichten.
        - num_layers (int): Anzahl der Schichten im LSTM.
        - dropout (float): Dropout-Rate zur Vermeidung von Überanpassung.
        """
        super(ThreeGroupLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Custom LSTM für die Hauptdatenquelle Y mit einer Eingabedimension von 10
        self.Y_layer = CustomLSTM(10, hidden_dim, num_layers=num_layers, dropout=dropout)

        # Liste von LSTMs für die erste Gruppe (X1), eine Schicht pro Zeitschritt der Sequenz (z.B., 5 Eingabe-Features)
        self.X1_layers = nn.ModuleList(
            [CustomLSTM(5, hidden_dim, num_layers=num_layers, dropout=dropout) for _ in range(seq_length)]
        )

        # Liste von LSTMs für die zweite Gruppe (X2), ebenfalls eine Schicht pro Zeitschritt (z.B., 4 Eingabe-Features)
        self.X2_layers = nn.ModuleList(
            [CustomLSTM(4, hidden_dim, num_layers=num_layers, dropout=dropout) for _ in range(seq_length)]
        )

        # Multi-Input LSTM-Schicht mit Gating zur Fusion der Daten
        self.multi_input_lstm = MultiInputLSTMWithGates(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        # Dual Attention Layer zur Auswahl der relevanten Features und Zeitpunkte
        self.dual_attention_layer = DualAttention(hidden_dim, hidden_dim)

        # Final Fully Connected Layer, um das Modell auf eine einzelne Prognose zu reduzieren
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Endausgabe ist eine einzelne Vorhersage
        )

    def forward(self, Y, X1, X2):
        """
        Führt den Vorwärtsdurchlauf des Modells aus, bei dem die Hauptquelle Y sowie die Gruppen X1 und X2 verarbeitet werden.
        Die relevanten Informationen werden extrahiert, durch Attention und Gating integriert und schließlich durch eine
        Fully Connected Layer in eine finale Vorhersage überführt.

        Parameter:
        - Y (Tensor): Hauptsignal-Tensor der Dimension [Batch, Seq, Features].
        - X1 (Tensor): Eingabedaten der ersten Gruppe von Nebensignalen (z.B., Momentum-Indikatoren).
        - X2 (Tensor): Eingabedaten der zweiten Gruppe von Nebensignalen (z.B., Trend-Indikatoren).

        Rückgabe:
        - output (Tensor): Finale Ausgabe des Modells.
        """

        # 1. Verarbeitung des Hauptsignals Y, um relevante Merkmale zu extrahieren
        Y_tilde, _ = self.Y_layer(Y)

        # 2. Gated Signal für die Kontrolle der Nebensignale durch das Hauptsignal
        self_gate = torch.sigmoid(self.dual_attention_layer(Y_tilde)).unsqueeze(1)

        # 3. Verarbeitung von X1 durch eine LSTM-Schicht pro Zeitschritt, um relevante versteckte Zustände zu berechnen
        X1_hidden = torch.stack([self.X1_layers[i](X1)[1] for i in range(len(self.X1_layers))], dim=0)

        # 4. Gleiches Verfahren für X2, wobei für jeden Zeitschritt eine separate LSTM-Schicht verwendet wird
        X2_hidden = torch.stack([self.X2_layers[i](X2)[1] for i in range(len(self.X2_layers))], dim=0)

        # 5. Mittelung der versteckten Zustände über die Zeitschritte hinweg für X1 und X2
        X1_tilde = torch.mean(X1_hidden, dim=0)
        X2_tilde = torch.mean(X2_hidden, dim=0)

        # 6. Multi-Input LSTM mit Gating zur Fusion von Y_tilde, X1_tilde und X2_tilde
        combined_input, _ = self.multi_input_lstm(Y_tilde, X1_tilde, X2_tilde, Y_tilde, self_gate)

        # 7. Anwendung des Dual Attention Layers auf das fusionierte Signal, um wichtige Informationen zu fokussieren
        attended = self.dual_attention_layer(combined_input)

        # 8. Transformation der fokussierten Information durch den finalen Fully Connected Layer
        output = self.fc(attended)

        return output


def train_group_model(group_name, X_samples, Y_samples, seq_length, hidden_dim, batch_size, learning_rate, epochs,
                      model_dir):
    """
    Trainiert ein ThreeGroupLSTMModel für eine spezifische Gruppe von Daten (z.B., Standard, Indicators_Group_1).
    Das Modell wird für eine Anzahl von Epochen mit dem Mean Squared Error (MSE) als Verlustfunktion und Adam als Optimierer trainiert.

    Parameter:
    - group_name (str): Name der Gruppe (z.B., "Standard", "Indicators_Group_1"), wird zum Speichern verwendet.
    - X_samples (Tensor): Eingabedaten (z.B., technische Indikatoren oder Preisdaten) für die Gruppe.
    - Y_samples (Tensor): Zielwerte für die Vorhersage.
    - seq_length (int): Länge der Sequenz der Eingabedaten.
    - hidden_dim (int): Dimension der versteckten Schicht des Modells.
    - batch_size (int): Anzahl der Datenproben in jedem Batch.
    - learning_rate (float): Lernrate des Optimierers.
    - epochs (int): Anzahl der Trainingsdurchläufe.
    - model_dir (str): Verzeichnis, in dem das trainierte Modell gespeichert wird.
    """

    # Initialisiere das Modell für die spezifische Gruppe und bewege es auf das Gerät (GPU, wenn verfügbar)
    model = ThreeGroupLSTMModel(seq_length, hidden_dim).to(device)

    # Definiere die Verlustfunktion und den Optimierer
    criterion = nn.MSELoss()  # Mean Squared Error (MSE) als Verlustfunktion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam-Optimierer

    # Erstelle einen Dataset- und Dataloader für die Eingabe- und Ziel-Daten
    dataset = torch.utils.data.TensorDataset(X_samples, Y_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setze das Modell in den Trainingsmodus
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0  # Variable zum Speichern des kumulierten Verlusts für die Epoche

        # Iteriere über jeden Batch im Dataloader
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.float().to(device), y.float().to(device)  # Übertrage die Daten auf das Gerät

            optimizer.zero_grad()  # Setze die Gradienten des Optimierers auf Null

            # Beispielhafte Eingabe: das Modell benötigt drei Eingaben (hier alle als X für Platzhalter)
            output = model(X, X, X)

            # Berechne den Verlust zwischen den Modellvorhersagen und den Zielwerten
            loss = criterion(output, y)
            loss.backward()  # Rückwärtsdurchlauf für die Gradientenberechnung
            optimizer.step()  # Aktualisiere die Modellparameter

            running_loss += loss.item()  # Addiere den Batch-Verlust zur laufenden Verlustsumme

        # Berechne den durchschnittlichen Verlust der Epoche
        epoch_loss = running_loss / len(dataloader)
        print(f"Gruppe {group_name} - Epoch {epoch + 1}/{epochs}, Durchschnittlicher Verlust: {epoch_loss:.6f}")

    # Speichere das trainierte Modell in das angegebene Verzeichnis mit dem Gruppennamen
    model_path = os.path.join(model_dir, f"{group_name}_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modell für Gruppe {group_name} gespeichert unter {model_path}")

    return model_path


class FusionModel(nn.Module):
    def __init__(self, hidden_dim=64, seq_length=50):
        """
        Initialisiert das FusionModel, welches die Vorhersagen von drei verschiedenen
        ThreeGroupLSTMModels kombiniert, um eine finale Vorhersage zu erzeugen.

        Parameter:
        - hidden_dim (int): Dimension der versteckten Schicht der Modelle.
        - seq_length (int): Länge der Eingabesequenzen für jedes Modell.
        """
        super(FusionModel, self).__init__()

        # Versteckte Dimension speichern
        self.hidden_dim = hidden_dim

        # Initialisiert drei ThreeGroupLSTMModel-Instanzen für die jeweiligen Gruppen
        self.standard_model = ThreeGroupLSTMModel(seq_length, hidden_dim)  # Modell für die Standard-Gruppe
        self.indicators_1_model = ThreeGroupLSTMModel(seq_length, hidden_dim)  # Modell für Indicators_Group_1
        self.indicators_2_model = ThreeGroupLSTMModel(seq_length, hidden_dim)  # Modell für Indicators_Group_2

        # Definiert eine Fully Connected-Schicht für die finale Fusion
        # - 3 Eingaben (eine pro Modell), werden auf die versteckte Dimension reduziert
        # - Relu-Aktivierung und eine finale Ausgabe mit einer einzigen Prognose
        self.fusion_fc = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # Kombiniert die drei Eingaben in eine versteckte Dimension
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Finale Ausgabe als eine einzelne Prognose
        )

    def forward(self, Y, X1, X2):
        """
        Führt die Vorwärtsausführung durch. Kombiniert die Vorhersagen der drei Modelle und generiert eine finale Ausgabe.

        Parameter:
        - Y, X1, X2: Eingabesequenzen für die Modelle. Diese repräsentieren jeweils die Daten der drei Gruppen.

        Rückgabewert:
        - output: Die kombinierte finale Vorhersage des FusionModels.
        """

        # Generiert die Vorhersagen der einzelnen Gruppenmodelle
        standard_pred = self.standard_model(Y, X1, X2)  # Vorhersage des Standard-Modells
        indicators_1_pred = self.indicators_1_model(Y, X1, X2)  # Vorhersage des Indicators_Group_1-Modells
        indicators_2_pred = self.indicators_2_model(Y, X1, X2)  # Vorhersage des Indicators_Group_2-Modells

        # Kombiniert die Vorhersagen zu einem einzigen Tensor
        # - Jede Vorhersage repräsentiert eine Dimension im Batch
        # - Die kombinierten Vorhersagen haben die Dimension [batch_size, 3]
        combined = torch.cat((standard_pred, indicators_1_pred, indicators_2_pred), dim=1)

        # Durchläuft die Fully Connected-Schicht zur Fusion und erzeugt die finale Ausgabe
        output = self.fusion_fc(combined)  # [batch_size, 1]

        return output  # Gibt die finale Vorhersage aus


def train_fusion_model(X_standard, X_group1, X_group2, Y_samples, hidden_dim, batch_size, learning_rate, epochs, model_dir):
    """
        Trainiert das FusionModel auf Basis der drei Eingabegruppen (Standard, Indicators_Group_1, Indicators_Group_2).

        Parameter:
        - X_standard (torch.Tensor): Sequenzierte Eingabedaten der Standard-Gruppe.
        - X_group1 (torch.Tensor): Sequenzierte Eingabedaten der ersten Indikatorengruppe.
        - X_group2 (torch.Tensor): Sequenzierte Eingabedaten der zweiten Indikatorengruppe.
        - Y_samples (torch.Tensor): Zielwerte für die Vorhersage.
        - hidden_dim (int): Dimension der versteckten Schicht des FusionModels.
        - batch_size (int): Größe des Batches für das Training.
        - learning_rate (float): Lernrate für den Optimizer.
        - epochs (int): Anzahl der Trainingsdurchläufe.
        - model_dir (str): Verzeichnis zum Speichern des trainierten Modells.

        Rückgabewert:
        - model_path (str): Pfad zum gespeicherten Modell.
        """
    # Initialisiert das FusionModel mit der gegebenen versteckten Dimension und überträgt es auf das Gerät (CPU oder GPU)
    model = FusionModel(hidden_dim).to(device)

    # Verlustfunktion (Mean Squared Error) für die Regressionsaufgabe
    criterion = nn.MSELoss()

    # Adam-Optimizer mit der spezifizierten Lernrate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader erstellen, um die Eingabedaten und Zielwerte in Batches zu laden
    # Jede Batch enthält X_standard, X_group1, X_group2 und die Zielwerte (Y_samples)
    dataset = torch.utils.data.TensorDataset(X_standard, X_group1, X_group2, Y_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setzt das Modell in den Trainingsmodus
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0 # Initialisiert die Laufvariable für den Verlust pro Epoche
        # Iteriert durch jeden Batch im DataLoader
        for batch_idx, (X_standard, X_group1, X_group2, y) in enumerate(dataloader):
            # Überträgt Eingaben und Zielwerte in die korrekten Datentypen und auf das Gerät (CPU oder GPU)
            X_standard, X_group1, X_group2, y = (
                X_standard.float().to(device),
                X_group1.float().to(device),
                X_group2.float().to(device),
                y.float().to(device),
            )

            # Rückwärtsgradienten auf Null setzen
            optimizer.zero_grad()

            # Vorhersage mit dem FusionModel
            output = model(X_standard, X_group1, X_group2)

            # Verlust berechnen; erwartet wird, dass y und output die gleiche Form haben [Batch, 1]
            loss = criterion(output, y)
            # Backpropagation des Verlusts
            loss.backward()
            # Aktualisiert die Modellparameter basierend auf den Gradienten
            optimizer.step()
            # Summiert den Verlust für diesen Batch
            running_loss += loss.item()
        # Durchschnittlicher Verlust für die Epoche
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Durchschnittlicher Verlust: {epoch_loss:.6f}")
    # Speichert das trainierte Modell
    model_path = os.path.join(model_dir, "fusion_model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Fusion Model gespeichert unter {model_path}")
    return model_path


# Hauptfunktion
if __name__ == '__main__':
    # Definiert die Verzeichnisse für gespeicherte und trainierte Daten
    samples_dir = "../../Data/Samples/"  # Verzeichnis mit den vorbereiteten Trainingssequenzen
    model_dir = "../../Data/Models/"  # Verzeichnis zum Speichern des trainierten Modells

    # Wichtige Modell- und Trainingsparameter
    seq_length = 50  # Länge der Eingabesequenzen in Tagen
    hidden_dim = 64  # Dimension der versteckten Schicht im Modell
    batch_size = 256  # Anzahl der Samples pro Batch
    learning_rate = 0.001  # Lernrate für den Optimizer
    epochs = 50  # Anzahl der Trainingsdurchläufe (Epochen)

    # Lade vorbereitete Trainingsdaten aus Pickle-Dateien und übertrage sie auf das Gerät (CPU oder GPU)
    with open(os.path.join(samples_dir, 'Standard_X.pkl'), 'rb') as f:
        X_standard = pickle.load(f).to(device)  # Standard-Feature-Gruppe
    with open(os.path.join(samples_dir, 'Indicators_Group_1_X.pkl'), 'rb') as f:
        X_group1 = pickle.load(f).to(device)  # Momentum-Indikatorgruppe
    with open(os.path.join(samples_dir, 'Indicators_Group_2_X.pkl'), 'rb') as f:
        X_group2 = pickle.load(f).to(device)  # Volatilität und Trend-Indikatorgruppe
    with open(os.path.join(samples_dir, 'Standard_Y.pkl'), 'rb') as f:
        Y_samples = pickle.load(f).to(device)  # Zielwerte für die Vorhersage
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
