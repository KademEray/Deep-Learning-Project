import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
from experiment_3_model_layer import CustomLSTM, MultiInputLSTMWithGates, DualAttention
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# Prüfen, ob CUDA verfügbar ist, und das Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicLSTMModel(nn.Module):
    """
    A dynamic LSTM model that processes multiple input sequences with different dimensions, 
    applies attention mechanisms, and outputs a sequence of features.
    Args:
        seq_length (int): The length of the input sequences.
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        input_sizes (list of int): A list of input dimensions for each LSTM layer.
        num_layers (int, optional): The number of recurrent layers in each LSTM. Default is 3.
        dropout (float, optional): The dropout probability for the LSTM layers. Default is 0.3.
    Attributes:
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        lstm_layers (nn.ModuleList): A list of custom LSTM layers for each input dimension.
        multi_input_lstm (MultiInputLSTMWithGates): A multi-input LSTM with gating mechanism.
        dual_attention_layer (DualAttention): A dual attention layer for processing combined inputs.
        fc (nn.Sequential): A fully connected layer for transforming the attended output.
    Methods:
        forward(*inputs):
            Processes the input sequences through the LSTM layers, applies attention mechanisms, 
            and outputs a sequence of features.
            Args:
                *inputs: Variable length input list, where each element is a tensor of shape 
                         [batch_size, seq_length, input_size].
            Returns:
                torch.Tensor: The output tensor of shape [batch_size, seq_length, 10].
    """
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


class FusionModel(nn.Module):
    """
    A PyTorch model that fuses predictions from three DynamicLSTMModel instances.
    Args:
        hidden_dim (int): The number of features in the hidden state of the LSTM. Default is 64.
        seq_length (int): The length of the input sequences. Default is 30.
        output_features (int): The number of output features. Default is 10.
        group_features (dict): A dictionary where keys are group names and values are lists of feature names for each group.
    Attributes:
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        seq_length (int): The length of the input sequences.
        output_features (int): The number of output features.
        standard_model (DynamicLSTMModel): The LSTM model for the standard group.
        indicators_1_model (DynamicLSTMModel): The LSTM model for the first indicators group.
        indicators_2_model (DynamicLSTMModel): The LSTM model for the second indicators group.
        fusion_fc (nn.Sequential): A fully connected layer for fusing the outputs of the three models.
    Methods:
        forward(Y, X1, X2):
            Forward pass of the model.
            Args:
                Y (torch.Tensor): Input tensor for the standard model.
                X1 (torch.Tensor): Input tensor for the first indicators model.
                X2 (torch.Tensor): Input tensor for the second indicators model.
            Returns:
                torch.Tensor: The fused output tensor with shape [Batch, Seq, Features].
    """
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
    """
    Trains a fusion model using dynamic LSTM layers on three different input groups and saves the trained model.
    Args:
        X_standard (torch.Tensor): Standard input tensor of shape (batch_size, seq_length, input_dim_standard).
        X_group1 (torch.Tensor): Group 1 input tensor of shape (batch_size, seq_length, input_dim_group1).
        X_group2 (torch.Tensor): Group 2 input tensor of shape (batch_size, seq_length, input_dim_group2).
        Y_samples (torch.Tensor): Target output tensor of shape (batch_size, seq_length, output_dim).
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        batch_size (int): The number of samples per batch to load.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        model_dir (str): Directory to save the trained model and plots.
    Returns:
        str: Path to the saved model file.
    """
    input_sizes = [X_standard.shape[2], X_group1.shape[2], X_group2.shape[2]]  # Dynamische Bestimmung der Eingabedimensionen

    model = DynamicLSTMModel(seq_length=50, hidden_dim=hidden_dim, input_sizes=input_sizes).to(device)
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