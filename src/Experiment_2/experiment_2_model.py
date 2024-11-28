import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
from experiment_2_model_layer import CustomLSTM, MultiInputLSTMWithGates, DualAttention
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# Prüfen, ob CUDA verfügbar ist, und das Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ThreeGroupLSTMModel(nn.Module):
    """
    A PyTorch neural network model that processes three groups of input sequences using custom LSTM layers,
    applies multi-input LSTM with gating and dual attention mechanisms, and produces an output sequence.
    Args:
        seq_length (int): The length of the input sequences.
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        num_layers (int, optional): The number of recurrent layers in the LSTM. Default is 3.
        dropout (float, optional): The dropout probability for the LSTM layers. Default is 0.3.
    Attributes:
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        Y_layer (CustomLSTM): Custom LSTM layer for the main data source Y with an input dimension of 5.
        X1_layer (CustomLSTM): Custom LSTM layer for the group X1 with an input dimension of 9.
        X2_layer (CustomLSTM): Custom LSTM layer for the group X2 with an input dimension of 9.
        multi_input_lstm (MultiInputLSTMWithGates): Multi-input LSTM with gating mechanism.
        dual_attention_layer (DualAttention): Dual attention mechanism.
        fc (nn.Sequential): Fully connected layer for each time step.
    Methods:
        forward(Y, X1, X2):
            Processes the input sequences Y, X1, and X2 through the model and returns the output sequence.
            Args:
                Y (torch.Tensor): The main input sequence with shape [Batch, Seq, Features].
                X1 (torch.Tensor): The input sequence for group X1 with shape [Batch, Seq, Features].
                X2 (torch.Tensor): The input sequence for group X2 with shape [Batch, Seq, Features].
            Returns:
                torch.Tensor: The output sequence with shape [Batch, Seq, Features].
    """
    def __init__(self, seq_length, hidden_dim, num_layers=3, dropout=0.3):
        super(ThreeGroupLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Custom LSTM für die Hauptdatenquelle Y mit einer Eingabedimension von 10
        self.Y_layer = CustomLSTM(5, hidden_dim, num_layers=num_layers, dropout=dropout)

        # Angepasste Eingabedimensionen für die LSTMs der Gruppen X1 und X2
        self.X1_layer = CustomLSTM(9, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.X2_layer = CustomLSTM(9, hidden_dim, num_layers=num_layers, dropout=dropout)

        # Multi-Input LSTM und Attention
        self.multi_input_lstm = MultiInputLSTMWithGates(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.dual_attention_layer = DualAttention(hidden_dim, hidden_dim)

        # Fully Connected Layer für jeden Zeitschritt
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # Endausgabe für 10 Features
        )

    def forward(self, Y, X1, X2):
        # Verarbeitung des Hauptsignals Y
        Y_tilde, _ = self.Y_layer(Y)

        # Gated Signal
        self_gate = torch.sigmoid(self.dual_attention_layer(Y_tilde))

        X1_tilde, _ = self.X1_layer(X1)
        X2_tilde, _ = self.X2_layer(X2)

        # Multi-Input LSTM zur Fusion
        combined_input, _ = self.multi_input_lstm(Y_tilde, X1_tilde, X2_tilde, Y_tilde, self_gate)

        # Attention und finale Transformation für jeden Zeitschritt
        attended = self.dual_attention_layer(combined_input)

        # Vollständige Sequenz als Ausgabe für jeden Zeitschritt durchlaufen
        output = self.fc(attended)  # Ausgabe mit Form [Batch, Seq, Features]

        return output


class FusionModel(nn.Module):
    """
    A PyTorch model that fuses predictions from three separate LSTM-based models.
    Args:
        hidden_dim (int): The number of features in the hidden state of the LSTM. Default is 64.
        seq_length (int): The length of the input sequences. Default is 30.
        output_features (int): The number of output features. Default is 5.
    Attributes:
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        seq_length (int): The length of the input sequences.
        output_features (int): The number of output features.
        standard_model (ThreeGroupLSTMModel): LSTM model for the standard group.
        indicators_1_model (ThreeGroupLSTMModel): LSTM model for the first indicators group.
        indicators_2_model (ThreeGroupLSTMModel): LSTM model for the second indicators group.
        fusion_fc (nn.Sequential): Fully connected layers for fusing the outputs of the three models.
    Methods:
        forward(Y, X1, X2):
            Forward pass of the model. Takes three input tensors and returns the fused output.
            Args:
                Y (torch.Tensor): Input tensor for the standard model.
                X1 (torch.Tensor): Input tensor for the first indicators model.
                X2 (torch.Tensor): Input tensor for the second indicators model.
            Returns:
                torch.Tensor: Fused output tensor.
    """
    def __init__(self, hidden_dim=64, seq_length=30, output_features=5):
        super(FusionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.output_features = output_features

        # Initialisiert drei ThreeGroupLSTMModel-Instanzen für die jeweiligen Gruppen
        self.standard_model = ThreeGroupLSTMModel(seq_length, hidden_dim)  # Modell für die Standard-Gruppe
        self.indicators_1_model = ThreeGroupLSTMModel(seq_length, hidden_dim)  # Modell für Indicators_Group_1
        self.indicators_2_model = ThreeGroupLSTMModel(seq_length, hidden_dim)  # Modell für Indicators_Group_2

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
    """
    Trains a fusion model using the provided datasets and parameters.
    Args:
        X_standard (torch.Tensor): Standard input features.
        X_group1 (torch.Tensor): Group 1 input features.
        X_group2 (torch.Tensor): Group 2 input features.
        Y_samples (torch.Tensor): Target output samples.
        hidden_dim (int): Dimension of the hidden layers in the model.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        model_dir (str): Directory to save the trained model and plots.
    Returns:
        str: Path to the saved model file.
    The function performs the following steps:
    1. Initializes the fusion model, loss criterion, and optimizer.
    2. Splits the dataset into training and validation sets.
    3. Trains the model for the specified number of epochs, recording training and validation losses.
    4. Saves the trained model and plots of the loss and metrics over epochs.
    """
    
    model = FusionModel(hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(X_standard, X_group1, X_group2, Y_samples)

    # Aufteilen in Trainings- und Validierungsdaten
    val_split = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [val_split, len(dataset) - val_split])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    scaler = torch.amp.GradScaler('cuda')

    model.train()
    train_loss_values = []
    val_loss_values = []
    val_mse_values = []
    val_rmse_values = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (X_standard, X_group1, X_group2, y) in enumerate(train_loader):
            X_standard, X_group1, X_group2, y = (
                X_standard.float().to(device),
                X_group1.float().to(device),
                X_group2.float().to(device),
                y.float().to(device),
            )

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                output = model(X_standard, X_group1, X_group2)
                output = output[:, -30:, :]  # Kürzt die Ausgabe auf die letzten 30 Zeitschritte

                loss = criterion(output, y)

            scaler.scale(loss).backward()  # Gradient scaling
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Durchschnittlicher Trainingsverlust
        epoch_loss = running_loss / len(train_loader)
        train_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.6f}")

        # Validierungsmodus
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for X_standard, X_group1, X_group2, y in val_loader:
                X_standard, X_group1, X_group2, y = (
                    X_standard.float().to(device),
                    X_group1.float().to(device),
                    X_group2.float().to(device),
                    y.float().to(device),
                )

                output = model(X_standard, X_group1, X_group2)
                output = output[:, -30:, :]  # Kürzt die Ausgabe auf die letzten 30 Zeitschritte

                loss = criterion(output, y)
                val_loss += loss.item()

                # MSE berechnen
                mse = nn.MSELoss()(output, y).item()
                val_mse += mse

        # Durchschnittlicher Validierungsverlust und Metriken
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_rmse = val_mse ** 0.5

        val_loss_values.append(val_loss)
        val_mse_values.append(val_mse)
        val_rmse_values.append(val_rmse)

        print(f"Validation Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, RMSE: {val_rmse:.6f}")

        model.train()  # Zurück in den Trainingsmodus

    # Modell speichern
    model_path = os.path.join(model_dir, "fusion_model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Fusion Model gespeichert unter {model_path}")

    # Plots erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_values, label="Training Loss", color="blue")
    plt.plot(range(1, epochs + 1), val_loss_values, label="Validation Loss", color="orange")
    plt.title("Training und Validation Loss über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(model_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss-Plot gespeichert unter {loss_plot_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), val_mse_values, label="Validation MSE", color="green")
    plt.plot(range(1, epochs + 1), val_rmse_values, label="Validation RMSE", color="red")
    plt.title("Validation MSE und RMSE über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("Fehler")
    plt.legend()
    plt.grid(True)
    metrics_plot_path = os.path.join(model_dir, "metrics_plot.png")
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"Metriken-Plot gespeichert unter {metrics_plot_path}")

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
    samples_dir = "./Data/Samples/"  # Verzeichnis mit den vorbereiteten Trainingssequenzen
    model_dir = "./Data/Models/"  # Verzeichnis zum Speichern des trainierten Modells

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
